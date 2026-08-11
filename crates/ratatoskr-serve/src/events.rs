//! Following a run's activity between checkpoints.
//!
//! Checkpoint arrival is a coarse signal — a node can work for minutes without producing one. The
//! structured log records every tool call and every piece of model text as it happens, so the
//! dashboard tails that instead of waiting.
//!
//! The log file is per process and per day, and concurrent runs interleave in it, so attribution
//! has to happen here: each record carries its `run_id` either as a field or through the enclosing
//! spans, and only records belonging to the requested run are forwarded.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::mpsc;

/// How often the log is checked for new lines. Fast enough to feel live, slow enough that an idle
/// dashboard costs nothing measurable.
const POLL: Duration = Duration::from_millis(300);

/// The most recent events replayed when a client attaches, so opening the dashboard mid-run shows
/// what has happened rather than an empty pane until the next event.
const REPLAY_LIMIT: usize = 200;

/// Longest `detail` forwarded. Model text is already truncated at the logging site; this is a
/// backstop so one enormous record can't stall a stream.
const DETAIL_LIMIT: usize = 2000;

/// One thing a run did, normalised for display.
///
/// The messy part — that `run_id` and `node` live in different places depending on which crate
/// logged the record — is resolved here rather than in the client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveEvent {
    pub at: String,
    /// `tool_call`, `model_text`, `checkpoint`, or whatever else the log carried.
    pub kind: String,
    pub node: Option<String>,
    /// The useful part: the tool name, the model's text, or the message.
    pub detail: String,
    /// Set on a `question` event: what a viewer's answer has to be posted against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question_id: Option<String>,
    /// An optional producer-provided summary for a tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// The bounded JSON arguments supplied to a tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    /// How long a tool took, on its `tool_result`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Model calls the attempt took, off a `checkpoint`. Not derivable from tool calls: a turn
    /// that answers without calling anything still cost a call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turns: Option<u64>,
    /// Why the node failed, off a `checkpoint`. Present is what makes a node render as failed
    /// rather than done — the two are indistinguishable from the fact of a checkpoint alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Which attempt this was, off a `checkpoint`. The implementer checkpoints once per converge
    /// iteration, so without it repeated attempts collapse into one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iteration: Option<u64>,
    /// Set on a `usage` event: what the node's attempt cost.
    ///
    /// Carried so a node's box can be rebuilt from the stream alone. Without it the numbers exist
    /// only on the checkpoint, which is the run's FINAL state — fine while following live, wrong
    /// the moment you look at where a run was rather than where it ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<LiveUsage>,
    /// Set on a `node_start` event: what the node is about to run on.
    ///
    /// A checkpoint carries the same facts, but only once the node has finished — and the moment
    /// a viewer most wants them is while it is still working.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facts: Option<LiveNodeFacts>,
}

/// What one attempt of a node cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub duration_ms: u64,
}

impl LiveUsage {
    /// Read them off a `usage` record. `None` for every other kind.
    ///
    /// The keys are dotted OpenTelemetry names (`gen_ai.usage.input_tokens`), which are flat keys
    /// in the JSON rather than nested objects — a `pointer()` lookup would find nothing.
    fn of(record: &Value) -> Option<Self> {
        if !matches!(
            record.get("kind").and_then(Value::as_str)?,
            "usage" | "checkpoint"
        ) {
            return None;
        }
        let n = |k: &str| record.get(k).and_then(Value::as_u64).unwrap_or(0);
        Some(LiveUsage {
            input_tokens: n("gen_ai.usage.input_tokens"),
            output_tokens: n("gen_ai.usage.output_tokens"),
            cached_input_tokens: n("gen_ai.usage.cached_input_tokens"),
            cache_creation_input_tokens: n("gen_ai.usage.cache_creation_input_tokens"),
            reasoning_tokens: n("gen_ai.usage.reasoning_tokens"),
            duration_ms: n("duration_ms"),
        })
    }
}

/// What a node announced about itself when it started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveNodeFacts {
    pub model: String,
    pub tools: Vec<String>,
    pub thinking: bool,
    pub reuses_session: bool,
}

impl LiveNodeFacts {
    /// Read them off a `node_start` record. `None` for every other kind.
    fn of(record: &Value) -> Option<Self> {
        // Both ends of an attempt carry them: `node_start` announces what the node was given,
        // and `checkpoint` records what it turned out to have. A viewer moving through a run needs
        // whichever came last.
        if !matches!(
            record.get("kind").and_then(Value::as_str)?,
            "node_start" | "checkpoint"
        ) {
            return None;
        }
        // A checkpoint for a node that ran no model has no route to report.
        if record
            .get("model")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return None;
        }
        let flag = |k: &str| record.get(k).and_then(Value::as_bool).unwrap_or(false);
        Some(LiveNodeFacts {
            model: record.get("model").and_then(Value::as_str)?.to_string(),
            // Joined for the log line, because a comma-separated list reads better there than a
            // JSON array does; split back here.
            tools: record
                .get("tools")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .split(',')
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect(),
            thinking: flag("thinking"),
            reuses_session: flag("reuses_session"),
        })
    }
}

/// The newest daily log file, if any. `tracing-appender` suffixes the date (`ratatoskr.jsonl.
/// 2026-08-05`), so there is never a bare `ratatoskr.jsonl`, and the dates sort lexicographically.
pub async fn newest_log(dir: &Path) -> Option<PathBuf> {
    daily_logs(dir).await.pop()
}

/// Every daily log file, oldest first.
async fn daily_logs(dir: &Path) -> Vec<PathBuf> {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return Vec::new();
    };
    let mut found = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let is_jsonl = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("ratatoskr.jsonl."));
        if is_jsonl {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Where a connecting viewer starts reading: the file before the newest, when there is one.
///
/// A run that was live across midnight has its beginning in the previous day's file, and replaying
/// only the newest would show a run that has been going for hours as though it had just started —
/// the more misleading answer, because the pane looks populated and is missing the half that
/// explains the run. One file back covers a run spanning one rollover, which is every run that is
/// not already pathological.
async fn replay_from(dir: &Path) -> Option<PathBuf> {
    let mut logs = daily_logs(dir).await;
    logs.pop();
    logs.pop()
}

/// Pull `run_id` out of a record: a plain field first (how the dashboard's own launch and reap
/// lines carry it, since they are emitted outside any run), then the enclosing spans.
fn run_id_of(record: &Value) -> Option<&str> {
    if let Some(id) = record.get("run_id").and_then(Value::as_str) {
        return Some(id);
    }
    record
        .get("spans")?
        .as_array()?
        .iter()
        .find_map(|span| span.get("run_id").and_then(Value::as_str))
}

/// Same idea for the node: a field on checkpoint records, the `agent` span for everything an
/// agent emits.
fn node_of(record: &Value) -> Option<&str> {
    let raw = record.get("node").and_then(Value::as_str).or_else(|| {
        record
            .get("spans")?
            .as_array()?
            .iter()
            .find_map(|span| span.get("node").and_then(Value::as_str))
    })?;
    Some(stage_name(raw))
}

/// The name the pipeline knows a node by.
///
/// The red team checkpoints as `red_team` and runs as `redteam` — a split the run itself relies
/// on. Everything downstream of here keys on the stage name, so an event arriving under the other
/// spelling belongs to no stage at all: it groups on its own in the feed, and the node it came
/// from never lights up while it is working. Normalising once here is cheaper than every consumer
/// remembering which side of the split it is on.
fn stage_name(node: &str) -> &str {
    match node {
        "redteam" => "red_team",
        other => other,
    }
}

/// Normalise one log record, keeping only what a viewer can act on.
fn to_event(record: &Value) -> LiveEvent {
    let kind = record
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("event")
        .to_string();
    let str_field = |key: &str| record.get(key).and_then(Value::as_str);
    let detail = match kind.as_str() {
        "tool_call" => str_field("tool").unwrap_or("tool"),
        "model_text" => str_field("text").unwrap_or_default(),
        // The question itself is the point of a question event, not its log message.
        "question" => str_field("question").unwrap_or_default(),
        _ => str_field("message").unwrap_or(&kind),
    };
    let detail = match detail.char_indices().nth(DETAIL_LIMIT) {
        Some((cut, _)) => format!("{}…", &detail[..cut]),
        None => detail.to_string(),
    };

    let args = (kind == "tool_call")
        .then(|| record.get("args")?.as_str())
        .flatten()
        .and_then(|args| serde_json::from_str(args).ok());

    LiveEvent {
        at: str_field("timestamp").unwrap_or_default().to_string(),
        subject: (kind == "tool_call")
            .then(|| str_field("tool_subject").map(str::to_string))
            .flatten(),
        args,
        duration_ms: record.get("duration_ms").and_then(Value::as_u64),
        question_id: str_field("question_id").map(str::to_string),
        facts: LiveNodeFacts::of(record),
        usage: LiveUsage::of(record),
        turns: record
            .get("turns")
            .and_then(Value::as_u64)
            .filter(|t| *t > 0),
        error: str_field("error")
            .filter(|e| !e.is_empty())
            .map(str::to_string),
        iteration: record.get("iteration").and_then(Value::as_u64),
        kind,
        node: node_of(record).map(str::to_string),
        detail,
    }
}

/// Parse a batch of lines into this run's events, dropping anything unparseable or belonging to
/// another run.
fn events_for(run_id: &str, lines: &[&str]) -> Vec<LiveEvent> {
    lines
        .iter()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| run_id_of(record) == Some(run_id))
        .map(|record| to_event(&record))
        .collect()
}

/// Every event this run produced, oldest first.
///
/// The scrubbing counterpart to [`follow`]: `follow` replays a tail and then tails, which is what
/// a viewer watching a live run wants and useless for moving through a finished one. Untrimmed —
/// a run is a few hundred events (a 42-minute run produced 654), so paging would cost more than
/// it saves.
///
/// Stored events first, log files as the fallback. A run whose history was ingested — an imported
/// run with no logs here, or one old enough to have lost them — is read from the store; a live run
/// whose events reached the log before anything ingested them is read from disk. The store keeps
/// each event's raw record in `payload_json`, so it parses back to the same [`LiveEvent`] the log
/// walk produces. Read-only: `history` never writes the store.
pub async fn history(store: &ratatoskr_store::Store, dir: &Path, run_id: &str) -> Vec<LiveEvent> {
    if let Ok(rows) = store.events_for_run(run_id).await
        && !rows.is_empty()
    {
        return rows
            .iter()
            .filter_map(|row| serde_json::from_str::<Value>(&row.payload_json).ok())
            .map(|record| to_event(&record))
            .collect();
    }

    let mut out = Vec::new();
    for path in daily_logs(dir).await {
        let Ok(text) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        out.extend(events_for(run_id, &text.lines().collect::<Vec<_>>()));
    }
    out
}

/// A run's log lines as store rows, ready to be made durable.
///
/// The same parse the dashboard reads with, so what is stored is what was shown — and the payload
/// is the raw record, so storing it loses nothing that a later reader might want.
pub async fn rows_for_run(dir: &Path, run_id: &str) -> Vec<ratatoskr_store::EventRow> {
    let mut out = Vec::new();
    for path in daily_logs(dir).await {
        let Ok(text) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        for line in text.lines() {
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if run_id_of(&record) != Some(run_id) {
                continue;
            }
            let event = to_event(&record);
            out.push(ratatoskr_store::EventRow {
                seq: out.len() as i64,
                at: event.at,
                kind: event.kind,
                node: event.node,
                payload_json: line.to_string(),
            });
        }
    }
    out
}

/// Cap what a newly attached viewer is replayed.
///
/// The tail, not the whole history — but never at the cost of a question. A run blocked on a
/// human is the one thing a viewer must see, and on a busy run its event is easily older than the
/// last few hundred lines.
fn trim_replay(events: &mut Vec<LiveEvent>) {
    if events.len() <= REPLAY_LIMIT {
        return;
    }
    let cut = events.len() - REPLAY_LIMIT;
    let mut seen = 0;
    events.retain(|event| {
        seen += 1;
        seen > cut || event.kind.starts_with("question")
    });
}

/// Read whatever has been appended since `pos`. Returns the new text and the new position.
async fn read_since(path: &Path, pos: u64) -> std::io::Result<(String, u64)> {
    let mut file = tokio::fs::File::open(path).await?;
    let len = file.metadata().await?.len();
    // A shorter file than last time means it was replaced under us; start over rather than
    // seeking past the end and reading garbage.
    let pos = if len < pos { 0 } else { pos };
    if len == pos {
        return Ok((String::new(), pos));
    }
    file.seek(std::io::SeekFrom::Start(pos)).await?;
    let mut buf = Vec::with_capacity((len - pos) as usize);
    file.take(len - pos).read_to_end(&mut buf).await?;
    Ok((String::from_utf8_lossy(&buf).into_owned(), len))
}

/// Where a tail has got to in the log.
#[derive(Default)]
struct Tail {
    current: Option<PathBuf>,
    pos: u64,
    /// A trailing fragment with no newline yet — the writer is mid-line.
    partial: String,
    replayed: bool,
}

impl Tail {
    /// Read and forward whatever is new. `Err(())` means the client is gone.
    async fn drain(&mut self, run_id: &str, tx: &mpsc::Sender<LiveEvent>) -> Result<(), ()> {
        let Some(path) = self.current.clone() else {
            return Ok(());
        };
        let Ok((chunk, next)) = read_since(&path, self.pos).await else {
            return Ok(());
        };
        self.pos = next;
        if chunk.is_empty() {
            return Ok(());
        }

        self.partial.push_str(&chunk);
        // Keep the unterminated tail: forwarding half a line would drop the event.
        let Some(end) = self.partial.rfind('\n') else {
            return Ok(());
        };
        let complete: String = self.partial.drain(..end + 1).collect();
        let lines: Vec<&str> = complete.lines().collect();

        let mut events = events_for(run_id, &lines);
        if !self.replayed {
            trim_replay(&mut events);
        }
        // Only once a complete line has actually been seen, so a connect that races a
        // half-written line still caps the backlog that follows.
        self.replayed = true;

        for event in events {
            tx.send(event).await.map_err(|_| ())?;
        }
        Ok(())
    }
}

/// Follow the log for one run until the receiver goes away.
///
/// One task per connected client. That is more polling than a single shared tailer broadcasting to
/// everyone, but the task dies with its channel — no shared state to own, and no lifecycle to get
/// wrong — and this is a local dashboard with a handful of viewers at most.
pub async fn follow(dir: PathBuf, run_id: String, tx: mpsc::Sender<LiveEvent>) {
    // Starts one file back, so a run that began before the last rollover is replayed whole. The
    // loop below walks forward to the newest file on its first pass.
    let mut tail = Tail {
        current: replay_from(&dir).await,
        ..Default::default()
    };

    loop {
        if tx.is_closed() {
            return;
        }

        // Follow the rollover: at midnight the run keeps writing, just to a new file.
        let newest = newest_log(&dir).await;
        if newest != tail.current {
            // Drain what we are leaving first. The rollover is noticed up to one poll after it
            // happens, so the old file's last lines would otherwise never be read.
            if tail.current.is_some() && tail.drain(&run_id, &tx).await.is_err() {
                return;
            }
            tail.current = newest;
            tail.pos = 0;
            tail.partial.clear();
        }

        if tail.drain(&run_id, &tx).await.is_err() {
            return;
        }

        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatoskr_store::{EventRow, Store};

    /// A record shaped exactly like the agent's, where attribution lives in the span list.
    fn agent_line(run: &str, node: &str) -> String {
        serde_json::json!({
            "timestamp": "2026-08-05T19:02:08Z",
            "kind": "tool_call",
            "tool": "semantic_search",
            "spans": [{"name": "run", "run_id": run}, {"name": "agent", "node": node}]
        })
        .to_string()
    }

    #[test]
    fn the_red_teams_two_names_arrive_as_the_one_the_pipeline_knows() {
        // It checkpoints as `red_team` and runs as `redteam`. Unnormalised, its events belong to
        // no stage: the node stays dark while it is plainly working, which is what a live run
        // showed with 36 events under the other spelling.
        let running: Value = serde_json::from_str(&agent_line("r1", "redteam")).unwrap();
        assert_eq!(to_event(&running).node.as_deref(), Some("red_team"));

        // The checkpoint side already uses the stage name and must pass through untouched.
        let done: Value = serde_json::from_str(
            r#"{"timestamp":"t","kind":"checkpoint","node":"red_team","spans":[{"run_id":"r1"}]}"#,
        )
        .unwrap();
        assert_eq!(to_event(&done).node.as_deref(), Some("red_team"));
    }

    #[test]
    fn an_acceptance_step_is_attributed_to_the_node_running_it() {
        // A suite takes minutes. Unattributed, the node running it reads as idle for the whole of
        // it, and a run rebuilt from the stream shows nothing happening while the tests run.
        let record: Value = serde_json::from_str(
            r#"{"timestamp":"t","kind":"acceptance_step","node":"red_team","step":"tests",
                "exit_code":0,"spans":[{"run_id":"r1"}]}"#,
        )
        .unwrap();
        assert_eq!(to_event(&record).node.as_deref(), Some("red_team"));
    }

    #[test]
    fn a_checkpoint_carries_everything_its_stored_row_does() {
        // The store keeps only each node's LATEST state, so a viewer reconstructing where a run
        // WAS has to read the log. Anything the row records and the event omits is a number that
        // would have to be back-filled from the present — which is how a historical view comes to
        // show final figures against a past position.
        let record: Value = serde_json::from_str(
            r#"{"timestamp":"t","kind":"checkpoint","node":"implementer","bytes":21286,
                "iteration":2,"model":"anthropic/claude-opus-4-8","tools":"Read,Bash",
                "tools_used":"Bash","thinking":true,"reuses_session":true,"turns":31,"error":"",
                "duration_ms":339000,"gen_ai.usage.input_tokens":7,
                "gen_ai.usage.output_tokens":396,"gen_ai.usage.cached_input_tokens":1065945,
                "gen_ai.usage.cache_creation_input_tokens":38998,
                "gen_ai.usage.reasoning_tokens":0,"spans":[{"run_id":"r1"}]}"#,
        )
        .unwrap();
        let e = to_event(&record);
        assert_eq!(e.turns, Some(31));
        assert_eq!(e.iteration, Some(2));
        assert_eq!(e.error, None, "an empty error is not a failure");

        let facts = e.facts.expect("a checkpoint reports what the node ran on");
        assert_eq!(facts.model, "anthropic/claude-opus-4-8");
        assert_eq!(facts.tools, ["Read", "Bash"]);
        assert!(facts.thinking && facts.reuses_session);

        let usage = e.usage.expect("a checkpoint reports what it cost");
        assert_eq!(usage.cached_input_tokens, 1_065_945);
        assert_eq!(usage.duration_ms, 339_000);
    }

    #[test]
    fn a_failed_node_is_told_apart_from_a_finished_one() {
        // Both write a checkpoint. Only the error distinguishes them, so it is what a box renders
        // as failed rather than done.
        let failed: Value = serde_json::from_str(
            r#"{"timestamp":"t","kind":"checkpoint","node":"verifier",
                "error":"verifier agent failed: UnknownToolCall","spans":[{"run_id":"r1"}]}"#,
        )
        .unwrap();
        assert_eq!(
            to_event(&failed).error.as_deref(),
            Some("verifier agent failed: UnknownToolCall")
        );
    }

    #[test]
    fn a_usage_event_carries_the_numbers_a_box_is_rebuilt_from() {
        // Dotted OpenTelemetry keys are flat in the JSON, not nested — read them as written or the
        // node's cost silently reads as zero everywhere it is derived from the stream.
        let record: Value = serde_json::from_str(
            r#"{"timestamp":"t","kind":"usage","node":"context",
                "gen_ai.usage.input_tokens":45,"gen_ai.usage.output_tokens":82,
                "gen_ai.usage.cached_input_tokens":853598,
                "gen_ai.usage.cache_creation_input_tokens":53807,
                "gen_ai.usage.reasoning_tokens":0,"duration_ms":226362,
                "spans":[{"run_id":"r1"}]}"#,
        )
        .unwrap();
        let usage = to_event(&record)
            .usage
            .expect("a usage event carries usage");
        assert_eq!(usage.input_tokens, 45);
        assert_eq!(usage.cached_input_tokens, 853_598);
        assert_eq!(usage.duration_ms, 226_362);

        // Every other kind carries none, so a consumer can key on its presence.
        let other: Value = serde_json::from_str(
            r#"{"timestamp":"t","kind":"tool_call","tool":"Read","spans":[{"run_id":"r1"}]}"#,
        )
        .unwrap();
        assert!(to_event(&other).usage.is_none());
    }

    #[test]
    fn tool_calls_keep_complete_arguments_and_only_explicit_subjects() {
        let args = serde_json::json!({
            "query": null,
            "symbol": "crate::T",
            "ref": "main",
            "id": "sym_1",
            "target": {"opaque_key": [{"leaf": 7}]},
        });
        let record = serde_json::json!({
            "timestamp": "t",
            "kind": "tool_call",
            "tool": "impact_surface",
            "args": args.to_string(),
            "spans": [{"run_id": "r1"}],
        });
        let event = to_event(&record);
        assert_eq!(event.detail, "impact_surface");
        let wire = serde_json::to_value(&event).unwrap();
        assert_eq!(wire["args"], args);
        assert!(wire.get("subject").is_none());
        assert_eq!(event.args, Some(args));
        assert_eq!(event.subject, None, "arguments never imply a subject");

        let record = serde_json::json!({
            "kind": "tool_call",
            "tool": "future_tool",
            "tool_subject": "provided by the tool",
            "args": r#"{"target":{"opaque_key":[{"leaf":7}]}}"#,
        });
        let event = to_event(&record);
        assert_eq!(event.subject.as_deref(), Some("provided by the tool"));
        assert_eq!(
            event.args,
            Some(serde_json::json!({"target": {"opaque_key": [{"leaf": 7}]}}))
        );
    }

    #[test]
    fn legacy_tool_arguments_fall_back_to_the_tool_name() {
        for args in [
            None,
            Some(serde_json::json!({"not": "a string"})),
            Some(serde_json::json!("{")),
        ] {
            let mut record = serde_json::json!({"kind": "tool_call", "tool": "find_callers"});
            if let Some(args) = args {
                record["args"] = args;
            }
            let event = to_event(&record);
            assert_eq!(event.detail, "find_callers");
            assert_eq!(event.args, None);
            assert_eq!(event.subject, None);
        }
    }

    #[test]
    fn attribution_comes_from_a_field_or_the_span_list() {
        let from_span: Value = serde_json::from_str(&agent_line("r1", "scout")).unwrap();
        assert_eq!(run_id_of(&from_span), Some("r1"));
        assert_eq!(node_of(&from_span), Some("scout"));

        // The dashboard's own launch/reap lines are emitted outside any run span.
        let from_field = serde_json::json!({"kind": "run_started", "run_id": "r2"});
        assert_eq!(run_id_of(&from_field), Some("r2"));
        assert_eq!(node_of(&from_field), None);

        assert_eq!(run_id_of(&serde_json::json!({"kind": "x"})), None);
    }

    #[test]
    fn only_the_requested_runs_lines_are_forwarded() {
        // Concurrent runs share one file, so this filter is what keeps streams separate.
        let lines = [
            agent_line("r1", "scout"),
            agent_line("r2", "analyst"),
            agent_line("r1", "analyst"),
            "not json at all".to_string(),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();

        let events = events_for("r1", &refs);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].node.as_deref(), Some("scout"));
        assert_eq!(events[1].node.as_deref(), Some("analyst"));
        // A malformed line is skipped rather than ending the stream.
        assert_eq!(events_for("r3", &refs).len(), 0);
    }

    #[test]
    fn replay_trimming_keeps_the_tail_and_every_question() {
        let question = LiveEvent {
            at: "t0".into(),
            kind: "question".into(),
            node: None,
            detail: "which way?".into(),
            question_id: Some("q-1".into()),
            subject: None,
            args: None,
            duration_ms: None,
            facts: None,
            usage: None,
            turns: None,
            error: None,
            iteration: None,
        };
        let noise = LiveEvent {
            at: "t1".into(),
            kind: "tool_call".into(),
            node: Some("scout".into()),
            detail: "semantic_search".into(),
            question_id: None,
            subject: None,
            args: None,
            duration_ms: None,
            facts: None,
            usage: None,
            turns: None,
            error: None,
            iteration: None,
        };

        // The question is the oldest event, well outside the replay window.
        let mut events = vec![question];
        events.extend(std::iter::repeat_n(noise.clone(), REPLAY_LIMIT + 50));
        trim_replay(&mut events);

        assert_eq!(
            events.iter().filter(|e| e.kind == "question").count(),
            1,
            "an open question survives the trim however old it is"
        );
        assert_eq!(events.len(), REPLAY_LIMIT + 1, "the rest is capped");

        // Under the limit nothing is dropped.
        let mut short = vec![noise; 3];
        trim_replay(&mut short);
        assert_eq!(short.len(), 3);
    }

    #[test]
    fn a_question_carries_its_id_and_text() {
        // Both are needed: the text to show, the id to answer against.
        let record = serde_json::json!({
            "kind": "question",
            "question_id": "q-7",
            "question": "which approach should I take?",
            "message": "waiting on the user",
            "spans": [{"name": "run", "run_id": "r1"}]
        });
        let event = to_event(&record);
        assert_eq!(event.question_id.as_deref(), Some("q-7"));
        assert_eq!(event.detail, "which approach should I take?");

        // Everything else leaves it unset, so the UI can't offer to answer a tool call.
        let tool: Value = serde_json::from_str(&agent_line("r1", "scout")).unwrap();
        assert!(to_event(&tool).question_id.is_none());
    }

    #[test]
    fn each_kind_surfaces_its_useful_part() {
        let tool: Value = serde_json::from_str(&agent_line("r", "scout")).unwrap();
        assert_eq!(to_event(&tool).detail, "semantic_search");

        let text = serde_json::json!({"kind": "model_text", "text": "thinking about it"});
        assert_eq!(to_event(&text).detail, "thinking about it");

        let checkpoint = serde_json::json!({"kind": "checkpoint", "message": "checkpoint"});
        let event = to_event(&checkpoint);
        assert_eq!(event.kind, "checkpoint");
        assert_eq!(event.detail, "checkpoint");
    }

    #[test]
    fn a_provider_pause_keeps_its_reason_in_the_activity_feed() {
        let record = serde_json::json!({
            "timestamp": "t",
            "kind": "run_paused",
            "node": "analyst",
            "message": "run paused: the API key has reached its usage limit; continue after adding capacity",
            "spans": [{"run_id": "r1"}],
        });
        let event = to_event(&record);
        assert_eq!(event.kind, "run_paused");
        assert_eq!(event.node.as_deref(), Some("analyst"));
        assert!(event.detail.contains("usage limit"));
    }

    #[test]
    fn an_enormous_record_is_truncated_not_forwarded_whole() {
        let huge = serde_json::json!({"kind": "model_text", "text": "x".repeat(9000)});
        let event = to_event(&huge);
        assert!(event.detail.ends_with('…'));
        assert_eq!(event.detail.chars().count(), DETAIL_LIMIT + 1);
    }

    #[tokio::test]
    async fn the_newest_daily_file_wins_and_a_missing_dir_is_not_an_error() {
        let dir = std::env::temp_dir().join(format!("ratatoskr-events-{}", std::process::id()));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        assert!(newest_log(&dir).await.is_none(), "no directory yet");

        tokio::fs::create_dir_all(&dir).await.unwrap();
        assert!(newest_log(&dir).await.is_none(), "no logs yet");

        for day in ["2026-08-04", "2026-08-05"] {
            tokio::fs::write(dir.join(format!("ratatoskr.jsonl.{day}")), "")
                .await
                .unwrap();
        }
        // The prose log sits in the same directory and must not be picked up.
        tokio::fs::write(dir.join("ratatoskr.log.2026-08-06"), "")
            .await
            .unwrap();

        // A viewer connecting starts one file back, so a run that was live across the rollover is
        // replayed whole rather than appearing to have just started.
        let from = replay_from(&dir).await.expect("the previous day's log");
        assert!(
            from.to_string_lossy()
                .ends_with("ratatoskr.jsonl.2026-08-04"),
            "{from:?}"
        );

        let newest = newest_log(&dir).await.expect("a log file");
        assert!(
            newest
                .to_string_lossy()
                .ends_with("ratatoskr.jsonl.2026-08-05")
        );
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn reading_resumes_from_the_last_position() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-events-read-{}", std::process::id()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let path = dir.join("ratatoskr.jsonl.2026-08-05");

        tokio::fs::write(&path, "one\n").await.unwrap();
        let (chunk, pos) = read_since(&path, 0).await.unwrap();
        assert_eq!(chunk, "one\n");

        // Nothing new: same position, no repeated content.
        let (chunk, pos) = read_since(&path, pos).await.unwrap();
        assert!(chunk.is_empty());

        tokio::fs::write(&path, "one\ntwo\n").await.unwrap();
        let (chunk, _) = read_since(&path, pos).await.unwrap();
        assert_eq!(chunk, "two\n", "only the appended part is re-read");

        // A replaced, shorter file restarts rather than seeking past its end.
        tokio::fs::write(&path, "x\n").await.unwrap();
        let (chunk, _) = read_since(&path, 999).await.unwrap();
        assert_eq!(chunk, "x\n");
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// A scratch log directory unique to this test and process — `follow` polls a real directory,
    /// and a shared path would let concurrent tests read each other's lines.
    fn scratch(case: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-follow-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn next_event(rx: &mut mpsc::Receiver<LiveEvent>) -> LiveEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("an event within five seconds")
            .expect("the stream is still open")
    }

    #[tokio::test]
    async fn follow_picks_up_appends_and_never_splits_a_line() {
        let dir = scratch("append");
        let path = dir.join("ratatoskr.jsonl.2026-08-05");
        std::fs::write(&path, format!("{}\n", agent_line("r1", "scout"))).unwrap();

        let (tx, mut rx) = mpsc::channel(16);
        let task = tokio::spawn(follow(dir.clone(), "r1".to_string(), tx));

        assert_eq!(next_event(&mut rx).await.node.as_deref(), Some("scout"));

        // Append a line in two writes, cut mid-JSON: the half-line must not be forwarded as an
        // event, and must not be lost once the writer finishes it.
        let second = format!("{}\n", agent_line("r1", "analyst"));
        let (head, tail) = second.split_at(second.len() / 2);
        std::fs::write(&path, format!("{}\n{head}", agent_line("r1", "scout"))).unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        std::fs::write(
            &path,
            format!("{}\n{head}{tail}", agent_line("r1", "scout")),
        )
        .unwrap();

        assert_eq!(next_event(&mut rx).await.node.as_deref(), Some("analyst"));

        task.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn follow_drains_the_old_file_before_moving_to_the_next_day() {
        let dir = scratch("rollover");
        let today = dir.join("ratatoskr.jsonl.2026-08-05");
        std::fs::write(&today, format!("{}\n", agent_line("r1", "scout"))).unwrap();

        let (tx, mut rx) = mpsc::channel(16);
        let task = tokio::spawn(follow(dir.clone(), "r1".to_string(), tx));
        assert_eq!(next_event(&mut rx).await.node.as_deref(), Some("scout"));

        // A line lands in the old file and the new day's file appears in the same window. The
        // straggler must still be delivered, not skipped past by the rotation.
        std::fs::write(
            &today,
            format!(
                "{}\n{}\n",
                agent_line("r1", "scout"),
                agent_line("r1", "memory")
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("ratatoskr.jsonl.2026-08-06"),
            format!("{}\n", agent_line("r1", "analyst")),
        )
        .unwrap();

        let seen = [
            next_event(&mut rx).await.node,
            next_event(&mut rx).await.node,
        ];
        assert!(
            seen.iter().any(|n| n.as_deref() == Some("memory")),
            "the old file's last line survives the rollover, got {seen:?}"
        );
        assert!(
            seen.iter().any(|n| n.as_deref() == Some("analyst")),
            "and the new file is picked up, got {seen:?}"
        );

        task.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn follow_stops_when_the_client_goes_away() {
        let dir = scratch("hangup");
        std::fs::write(
            dir.join("ratatoskr.jsonl.2026-08-05"),
            format!("{}\n", agent_line("r1", "scout")),
        )
        .unwrap();

        let (tx, rx) = mpsc::channel(16);
        let task = tokio::spawn(follow(dir.clone(), "r1".to_string(), tx));
        drop(rx); // the SSE response was dropped: the tail must not outlive it

        let ended = tokio::time::timeout(Duration::from_secs(5), task).await;
        assert!(ended.is_ok(), "the tailing task must end with its receiver");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- history: the store is read first, the log dir is the fallback (issue #125) ---

    #[tokio::test]
    async fn history_reads_the_stored_events_when_the_run_has_any() {
        // An imported or rotated-away run has no log files on this machine; its timeline can only
        // come from the store. Each returned event must equal what `to_event` makes of the row's
        // raw `payload_json`, so the same parse the log path uses applies unchanged.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();
        let dir = scratch("history-store"); // deliberately empty: no log files for this run
        let payload = agent_line("r1", "context");
        store
            .ingest_events(
                "r1",
                vec![EventRow {
                    seq: 0,
                    at: "2026-08-05T19:02:08Z".into(),
                    kind: "tool_call".into(),
                    node: Some("context".into()),
                    payload_json: payload.clone(),
                }],
            )
            .await
            .unwrap();

        let got = history(&store, &dir, "r1").await;
        let expected = to_event(&serde_json::from_str::<Value>(&payload).unwrap());
        assert_eq!(got, vec![expected]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn history_and_live_replay_normalize_tool_arguments_identically() {
        let args = serde_json::json!({"query": null, "ref": "main", "nested": {"leaf": 7}});
        let payload = serde_json::json!({
            "timestamp": "2026-08-05T19:02:08Z",
            "kind": "tool_call",
            "tool": "find_callers",
            "tool_subject": "explicit target",
            "args": args.to_string(),
            "spans": [{"run_id": "r1"}],
        })
        .to_string();
        let live = events_for("r1", &[&payload]);

        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();
        store
            .ingest_events(
                "r1",
                vec![EventRow {
                    seq: 0,
                    at: "2026-08-05T19:02:08Z".into(),
                    kind: "tool_call".into(),
                    node: None,
                    payload_json: payload,
                }],
            )
            .await
            .unwrap();
        let dir = scratch("history-tool-args");
        assert_eq!(history(&store, &dir, "r1").await, live);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn history_falls_back_to_the_logs_when_the_store_is_empty() {
        // A live run's events reach the log before anything ingests them, so with no stored rows
        // the timeline is exactly the log-only walk `history` produced before.
        let store = Store::open_in_memory().unwrap();
        let dir = scratch("history-logs");
        std::fs::write(
            dir.join("ratatoskr.jsonl.2026-08-05"),
            format!(
                "{}\n{}\n",
                agent_line("r1", "scout"),
                agent_line("r1", "analyst")
            ),
        )
        .unwrap();

        let got = history(&store, &dir, "r1").await;
        assert_eq!(
            got.iter().map(|e| e.node.clone()).collect::<Vec<_>>(),
            vec![Some("scout".into()), Some("analyst".into())],
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn history_prefers_the_store_and_leaves_the_logs_unread() {
        // Present in both: the stored events win outright and the log file for that run is not read,
        // so a run that has been ingested does not have its log lines appended on top (no duplication).
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();
        let dir = scratch("history-both");
        // Logs say "scout"; the store says "context". Different on purpose so the source is visible.
        std::fs::write(
            dir.join("ratatoskr.jsonl.2026-08-05"),
            format!("{}\n", agent_line("r1", "scout")),
        )
        .unwrap();
        store
            .ingest_events(
                "r1",
                vec![EventRow {
                    seq: 0,
                    at: "2026-08-05T19:02:08Z".into(),
                    kind: "tool_call".into(),
                    node: Some("context".into()),
                    payload_json: agent_line("r1", "context"),
                }],
            )
            .await
            .unwrap();

        let got = history(&store, &dir, "r1").await;
        assert_eq!(
            got.len(),
            1,
            "exactly the stored events, no log duplication"
        );
        assert_eq!(got[0].node.as_deref(), Some("context"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn history_of_a_run_unknown_to_both_is_empty() {
        // No rows and no logs is an empty timeline, not an error.
        let store = Store::open_in_memory().unwrap();
        let dir = scratch("history-none");
        assert!(history(&store, &dir, "nope").await.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn history_skips_a_stored_row_whose_payload_is_not_json() {
        // One unparseable row is dropped like an unparseable log line; the rest of the timeline
        // still returns rather than the whole run aborting on it.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();
        let dir = scratch("history-badrow");
        store
            .ingest_events(
                "r1",
                vec![
                    EventRow {
                        seq: 0,
                        at: "2026-08-05T19:02:08Z".into(),
                        kind: "junk".into(),
                        node: None,
                        payload_json: "not json at all".into(),
                    },
                    EventRow {
                        seq: 1,
                        at: "2026-08-05T19:02:09Z".into(),
                        kind: "tool_call".into(),
                        node: Some("context".into()),
                        payload_json: agent_line("r1", "context"),
                    },
                ],
            )
            .await
            .unwrap();

        let got = history(&store, &dir, "r1").await;
        assert_eq!(got.len(), 1, "the good row survives one bad one");
        assert_eq!(got[0].node.as_deref(), Some("context"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
