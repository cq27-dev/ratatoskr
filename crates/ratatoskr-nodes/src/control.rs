//! The run process's half of the operator controls: pause, stop and steer.
//!
//! A run executes here, and the operator is at a dashboard in another process. The dashboard holds
//! what they asked for; this asks it, at each node's turn boundaries, what that node should do
//! now. The direction is deliberate and matches the clarification rendezvous next door: the run
//! reaches out, so nothing outside it can reach in — into its worktree, its conversation, or the
//! store only it may write.
//!
//! **Every failure means carry on.** An unreachable dashboard, a reply that will not parse, a run
//! nobody is watching: all of them answer `Continue`. A control channel that could stall a run by
//! breaking would be worse than having none, because it would fail at exactly the moment the
//! operator was trying to intervene.

use std::sync::Arc;
use std::time::Duration;

use ratatoskr_agent::Controller;
use ratatoskr_core::Control;

/// Ceiling on one control request. Short: this is a loopback call between two local processes, and
/// a node is waiting on it at every turn boundary.
const POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// Asks the dashboard that started this run what its nodes should do.
struct DashboardControl {
    dashboard: String,
    project: String,
    run_id: String,
    /// One client, so the connection to the dashboard is kept alive across a run's many polls
    /// rather than rebuilt at every turn boundary.
    client: reqwest::Client,
}

impl Controller for DashboardControl {
    fn poll<'a>(
        &'a self,
        node: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Control> + Send + 'a>> {
        Box::pin(async move {
            let asked = self
                .client
                .post(format!("{}/internal/control", self.dashboard))
                .json(&serde_json::json!({
                    "project": self.project,
                    "run_id": self.run_id,
                    "node": node,
                }))
                .timeout(POLL_TIMEOUT)
                .send()
                .await;
            match asked {
                Ok(reply) => reply.json::<Control>().await.unwrap_or_else(|e| {
                    // Worth a line: a reply this side cannot read means the two processes disagree
                    // about the protocol, and the operator's buttons will look broken.
                    tracing::debug!("could not read the dashboard's control reply: {e}");
                    Control::carry_on()
                }),
                Err(e) => {
                    tracing::debug!("could not reach the dashboard for control: {e}");
                    Control::carry_on()
                }
            }
        })
    }
}

/// Point this run's nodes at the dashboard that started it, if one did.
///
/// Does nothing for a run started from the command line: there is no dashboard holding commands,
/// and a node that cannot be paused by anyone should not pay for asking.
pub fn install(run_id: &str) {
    let Ok(dashboard) = std::env::var("RATATOSKR_DASHBOARD") else {
        return;
    };
    ratatoskr_agent::configure_control(Arc::new(DashboardControl {
        dashboard,
        // Empty unless the dashboard spawned this run, which is also the only case in which it has
        // anything to say.
        project: std::env::var("RATATOSKR_PROJECT").unwrap_or_default(),
        run_id: run_id.to_string(),
        client: reqwest::Client::new(),
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_unreachable_dashboard_lets_the_node_carry_on() {
        // The defect this guards is a run that stalls because the thing watching it went away.
        // Port 1 is not listening, so this is the unreachable case without a fixture.
        let control = DashboardControl {
            dashboard: "http://127.0.0.1:1".to_string(),
            project: "p".to_string(),
            run_id: "r".to_string(),
            client: reqwest::Client::new(),
        };
        assert_eq!(control.poll("analyst").await, Control::carry_on());
    }
}
