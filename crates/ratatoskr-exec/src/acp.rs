//! ACP client: drive an external coding CLI (Claude Code, via its ACP adapter subprocess) to make
//! edits in a worktree. Ratatoskr is the ACP *Client*; the CLI adapter is the *Agent*.
//!
//! Shape follows `agent-client-protocol`'s own `yolo_one_shot_client` example: build a `Client`,
//! auto-approve permission requests (the run is already sandboxed), then per turn initialize →
//! open a session rooted at the worktree → send the prompt. Session updates stream in as
//! notifications; we accumulate them as the (optional) narrative and return the stop reason.

use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
    SessionNotification, TextContent,
};
use agent_client_protocol::{AcpAgent, Agent, ConnectionTo};

/// The result of one ACP turn.
#[derive(Debug, Clone)]
pub struct AcpTurnResult {
    /// Why the agent stopped (debug form of the protocol's stop reason).
    pub stop_reason: String,
    /// Accumulated session updates — the agent's narrative of what it did (best-effort).
    pub output: String,
}

/// Errors driving the ACP agent.
#[derive(Debug, thiserror::Error)]
pub enum AcpError {
    #[error("invalid agent command {0:?}: {1}")]
    Command(String, String),
    #[error("ACP protocol error: {0}")]
    Protocol(String),
}

/// Spawn `command` as an ACP agent, open a session at `cwd`, send `prompt`, and return the result.
pub async fn drive(command: &str, cwd: &Path, prompt: &str) -> Result<AcpTurnResult, AcpError> {
    let agent = AcpAgent::from_str(command)
        .map_err(|e| AcpError::Command(command.to_string(), e.to_string()))?;

    let updates = Arc::new(Mutex::new(String::new()));
    let updates_notif = updates.clone();
    let updates_end = updates.clone();
    let cwd = cwd.to_path_buf();
    let prompt = prompt.to_string();

    agent_client_protocol::Client
        .builder()
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                if let Ok(mut buf) = updates_notif.lock() {
                    buf.push_str(&format!("{:?}\n", notification.update));
                }
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _connection| {
                // The run is already sandboxed, so auto-APPROVE. Options come as
                // [reject, allow, allow_always]; picking the first would DENY, so prefer an
                // option whose `kind` starts with "allow" (fall back to the last, usually the
                // most permissive). Match on the serialized `kind` to stay independent of the
                // schema's exact enum type.
                let allow = request
                    .options
                    .iter()
                    .find(|opt| {
                        serde_json::to_value(opt)
                            .ok()
                            .and_then(|v| {
                                v.get("kind")
                                    .and_then(|k| k.as_str())
                                    .map(|s| s.starts_with("allow"))
                            })
                            .unwrap_or(false)
                    })
                    .or_else(|| request.options.last());
                match allow.map(|opt| opt.option_id.clone()) {
                    Some(id) => responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id)),
                    )),
                    None => responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Cancelled,
                    )),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, move |connection: ConnectionTo<Agent>| async move {
            connection
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let session = connection
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;

            let response = connection
                .send_request(PromptRequest::new(
                    session.session_id.clone(),
                    vec![ContentBlock::Text(TextContent::new(prompt))],
                ))
                .block_task()
                .await?;

            let output = updates_end.lock().map(|b| b.clone()).unwrap_or_default();
            Ok(AcpTurnResult {
                stop_reason: format!("{:?}", response.stop_reason),
                output,
            })
        })
        .await
        .map_err(|e| AcpError::Protocol(e.to_string()))
}
