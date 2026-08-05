//! The desk where a run's questions wait for a human.
//!
//! A run is a separate process, so this is a rendezvous between two of them: the run posts a
//! question and waits, a browser posts the answer, and this hands it back. Everything here is
//! in-memory and non-durable on purpose — the store stays single-writer, owned by the run, and a
//! question that is lost because the dashboard restarted simply falls through to the analyst.
//!
//! Every failure mode has the same correct default: **nobody is watching, so the analyst answers.**
//! Unreachable dashboard, no viewer, viewer closed the tab, nobody typed in time — all of them end
//! up there, which is exactly how an unattended run behaves today.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

/// How long a question may block a node.
///
/// Deliberately well under the five-minute prompt-cache TTL: the whole point of answering through
/// the tool result is that the asking node's conversation — and its cached prefix — survives. A
/// "generous" timeout longer than the TTL would destroy the thing this design exists to protect.
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(120);

/// How long attendance may lapse before a parked question gives up. A page refresh drops and
/// remakes the event stream, and that shouldn't dump the question to the analyst.
const ATTENDANCE_GRACE: Duration = Duration::from_secs(10);

/// How often a parked question re-checks attendance and the clock.
const TICK: Duration = Duration::from_millis(500);

/// What a run asks the dashboard.
#[derive(Debug, Deserialize)]
pub struct AskRequest {
    pub run_id: String,
    pub question_id: String,
    /// Which project the asking run belongs to. A run id is unique within a project but an
    /// operator can reuse one across projects, so attendance is keyed by both.
    #[serde(default)]
    pub project: String,
}

/// What it gets back. `answer: None` means "nobody answered — use your own fallback"; the run
/// never distinguishes *why*, because its response is the same in every case.
#[derive(Debug, Serialize)]
pub struct AskReply {
    pub answer: Option<String>,
}

/// Why an answer couldn't be delivered.
#[derive(Debug, PartialEq, Eq)]
pub enum AnswerError {
    /// The question is no longer waiting: already answered, timed out, or the run moved on.
    NotPending,
}

#[derive(Default)]
struct State {
    /// Event-stream subscribers per watched run — the only signal that a human is present.
    watchers: HashMap<Watched, usize>,
    /// When a run's watcher count last hit zero, for the grace period.
    lonely_since: HashMap<Watched, Instant>,
    /// Questions waiting on a human, by question id.
    parked: HashMap<String, oneshot::Sender<String>>,
}

/// Identifies a run being watched. Scoped by project because a run id is only unique within one:
/// `ratatoskr run --run-id` lets an operator pick the same id in two projects, and a viewer of one
/// must not make the other's run look attended.
type Watched = (String, String);

fn watched(project: &str, run_id: &str) -> Watched {
    (project.to_string(), run_id.to_string())
}

/// Tracks who is watching and what is waiting to be answered.
#[derive(Default)]
pub struct Desk {
    state: Mutex<State>,
}

/// Held for as long as one client is watching a run. Dropping it decrements the count, so
/// attendance follows the event stream's lifetime without any explicit disconnect handling.
pub struct Attending {
    desk: Arc<Desk>,
    key: Watched,
}

impl Drop for Attending {
    fn drop(&mut self) {
        let mut state = self.desk.state.lock().expect("desk mutex");
        let count = state.watchers.entry(self.key.clone()).or_insert(1);
        *count = count.saturating_sub(1);
        if *count == 0 {
            state.watchers.remove(&self.key);
            state.lonely_since.insert(self.key.clone(), Instant::now());
        }
    }
}

impl Desk {
    /// Register a viewer for `run_id`.
    pub fn attend(self: &Arc<Self>, project: &str, run_id: &str) -> Attending {
        let key = watched(project, run_id);
        {
            let mut state = self.state.lock().expect("desk mutex");
            *state.watchers.entry(key.clone()).or_insert(0) += 1;
            state.lonely_since.remove(&key);
        }
        Attending {
            desk: Arc::clone(self),
            key,
        }
    }

    fn watching(&self, key: &Watched) -> bool {
        self.state
            .lock()
            .expect("desk mutex")
            .watchers
            .get(key)
            .is_some_and(|n| *n > 0)
    }

    /// How long this run has had no viewer, or `None` if one is watching.
    fn lonely_for(&self, key: &Watched) -> Option<Duration> {
        let state = self.state.lock().expect("desk mutex");
        if state.watchers.get(key).is_some_and(|n| *n > 0) {
            return None;
        }
        Some(
            state
                .lonely_since
                .get(key)
                .map_or(Duration::MAX, |since| since.elapsed()),
        )
    }

    /// Deliver an answer. Fails if the question is no longer waiting — which covers a second
    /// dashboard answering, an answer typed after the timeout, and a replayed stale question.
    pub fn answer(&self, question_id: &str, answer: String) -> Result<(), AnswerError> {
        let sender = {
            let mut state = self.state.lock().expect("desk mutex");
            state.parked.remove(question_id)
        };
        sender
            .ok_or(AnswerError::NotPending)?
            .send(answer)
            .map_err(|_| AnswerError::NotPending)
    }

    fn park<'a>(&'a self, question_id: &str) -> (Parked<'a>, oneshot::Receiver<String>) {
        let (tx, rx) = oneshot::channel();
        self.state
            .lock()
            .expect("desk mutex")
            .parked
            .insert(question_id.to_string(), tx);
        (
            Parked {
                desk: self,
                question_id: question_id.to_string(),
            },
            rx,
        )
    }

    fn unpark(&self, question_id: &str) {
        self.state
            .lock()
            .expect("desk mutex")
            .parked
            .remove(question_id);
    }

    /// Wait for a human to answer `question_id`, or give up.
    ///
    /// Returns `None` the moment it is clear no answer is coming, so an unattended run never
    /// waits at all.
    pub async fn wait_for_answer(
        &self,
        project: &str,
        run_id: &str,
        question_id: &str,
    ) -> Option<String> {
        let key = watched(project, run_id);
        if !self.watching(&key) {
            return None;
        }
        // The guard unparks on every exit, including the one this function can't see: the whole
        // future being dropped when the run's connection goes away.
        let (_parked, mut rx) = self.park(question_id);
        let deadline = Instant::now() + ANSWER_TIMEOUT;

        loop {
            tokio::select! {
                answered = &mut rx => return answered.ok(),
                _ = tokio::time::sleep(TICK) => {
                    let gave_up = Instant::now() >= deadline
                        || self.lonely_for(&key).is_some_and(|d| d > ATTENDANCE_GRACE);
                    if gave_up {
                        // An answer may have landed while we were deciding to give up. It has
                        // already been reported delivered, so take it rather than discard it.
                        return rx.try_recv().ok();
                    }
                }
            }
        }
    }
}

/// Removes a question from the waiting set when the wait ends, however it ends.
struct Parked<'a> {
    desk: &'a Desk,
    question_id: String,
}

impl Drop for Parked<'_> {
    fn drop(&mut self) {
        self.desk.unpark(&self.question_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desk() -> Arc<Desk> {
        Arc::new(Desk::default())
    }

    #[tokio::test]
    async fn an_unwatched_run_does_not_wait_at_all() {
        // The guarantee that keeps `ratatoskr run` behaving exactly as it does today.
        let desk = desk();
        let started = Instant::now();
        assert!(desk.wait_for_answer("p", "r1", "q1").await.is_none());
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn a_watched_run_receives_the_answer() {
        let desk = desk();
        let _viewer = desk.attend("p", "r1");

        let waiting = tokio::spawn({
            let desk = Arc::clone(&desk);
            async move { desk.wait_for_answer("p", "r1", "q1").await }
        });

        // Let the question park before answering it.
        tokio::time::sleep(Duration::from_millis(50)).await;
        desk.answer("q1", "use the second approach".into()).unwrap();

        assert_eq!(
            waiting.await.unwrap().as_deref(),
            Some("use the second approach")
        );
    }

    #[tokio::test]
    async fn watching_one_project_does_not_attend_another_projects_run() {
        // `ratatoskr run --run-id` lets an operator pick the same id in two projects. A viewer of
        // one must not make the other's run wait for an answer nobody is going to give.
        let desk = desk();
        let _viewer = desk.attend("alpha", "same-id");

        let started = Instant::now();
        assert!(
            desk.wait_for_answer("beta", "same-id", "q1")
                .await
                .is_none(),
            "the other project's run is unattended"
        );
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "and is not delayed by the grace period or the timeout"
        );
    }

    #[tokio::test]
    async fn only_the_first_answer_is_accepted() {
        let desk = desk();
        let _viewer = desk.attend("p", "r1");
        let waiting = tokio::spawn({
            let desk = Arc::clone(&desk);
            async move { desk.wait_for_answer("p", "r1", "q1").await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(desk.answer("q1", "first".into()).is_ok());
        // A second dashboard, or a late click: the question is no longer pending.
        assert_eq!(
            desk.answer("q1", "second".into()),
            Err(AnswerError::NotPending)
        );
        assert_eq!(waiting.await.unwrap().as_deref(), Some("first"));
    }

    #[tokio::test]
    async fn a_dropped_wait_leaves_nothing_parked() {
        // The run's connection can vanish mid-wait; the entry must not outlive the future.
        let desk = desk();
        let _viewer = desk.attend("p", "r1");
        {
            let waiting = desk.wait_for_answer("p", "r1", "q1");
            tokio::pin!(waiting);
            // Poll once so the question is actually parked, then drop the future.
            let _ = tokio::time::timeout(Duration::from_millis(50), &mut waiting).await;
        }
        assert_eq!(
            desk.answer("q1", "too late".into()),
            Err(AnswerError::NotPending),
            "the parked entry is gone once the waiter is dropped"
        );
    }

    #[tokio::test]
    async fn answering_something_that_was_never_asked_is_refused() {
        assert_eq!(
            desk().answer("nope", "hello".into()),
            Err(AnswerError::NotPending)
        );
    }

    #[tokio::test]
    async fn attendance_follows_the_viewer_guard() {
        let desk = desk();
        assert!(!desk.watching(&watched("p", "r1")));

        let first = desk.attend("p", "r1");
        let second = desk.attend("p", "r1");
        assert!(desk.watching(&watched("p", "r1")));

        // Two tabs on the same run: one closing doesn't end attendance.
        drop(first);
        assert!(desk.watching(&watched("p", "r1")));
        assert!(desk.lonely_for(&watched("p", "r1")).is_none());

        drop(second);
        assert!(!desk.watching(&watched("p", "r1")));
        assert!(desk.lonely_for(&watched("p", "r1")).is_some());
    }

    #[tokio::test]
    async fn losing_the_viewer_gives_up_after_the_grace_period() {
        let desk = desk();
        let viewer = desk.attend("p", "r1");
        let waiting = tokio::spawn({
            let desk = Arc::clone(&desk);
            async move { desk.wait_for_answer("p", "r1", "q1").await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Closing the tab shouldn't strand the node for the full timeout...
        drop(viewer);
        // ...but a refresh reconnects within the grace, so re-attending keeps it parked.
        let _reopened = desk.attend("p", "r1");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished(), "still parked while someone watches");

        desk.answer("q1", "back again".into()).unwrap();
        assert_eq!(waiting.await.unwrap().as_deref(), Some("back again"));
    }
}
