//! What an operator asks of a run that is already in flight.
//!
//! A run executes in its own process, and the dashboard is a different one. These are the words
//! the two use: the dashboard records [`Command`]s an operator issued, and each node asks — at its
//! own turn boundaries — what it should do now, which is answered with a [`Control`].
//!
//! The direction matters. Nothing here lets the dashboard reach into a run and change it; a node
//! asks, and the answer is advice it acts on at a point where acting is safe. That is what keeps a
//! pause from landing halfway through a tool call, and what keeps this out of the checkpoint store,
//! which a run process alone may write.

use serde::{Deserialize, Serialize};

/// What a node should do when it next reaches a turn boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Directive {
    /// Carry on.
    #[default]
    Continue,
    /// Hold here and ask again shortly. The node's conversation is untouched, so resuming costs
    /// nothing beyond the wait — this is a pause, not a restart.
    Hold,
    /// End this node's turn loop. The run then parks until the operator starts the node again,
    /// which re-runs it from the input its checkpoint holds.
    Stop,
}

/// The answer to one node's "what should I do now?".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Control {
    pub directive: Directive,
    /// Operator text waiting for this node, oldest first.
    ///
    /// Handed over exactly once: whoever answers this removes it, because a message repeated on
    /// every poll would be read by the model as the operator saying it again and again.
    #[serde(default)]
    pub steer: Vec<String>,
}

impl Control {
    /// Nothing to say and nothing to change — the common answer, and the one a run assumes when it
    /// has no dashboard to ask.
    pub fn carry_on() -> Self {
        Control::default()
    }
}

/// One thing an operator did to a run.
///
/// `Pause` and `Resume` are run-wide, because that is what the control beside the scrubber means.
/// The rest name a node: a run's fork has two nodes working at once, and stopping or steering "the
/// run" would be ambiguous exactly when it matters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Command {
    Pause,
    Resume,
    Stop { node: String },
    Start { node: String },
    Steer { node: String, text: String },
}

/// A run's control state, as a reader of the dashboard sees it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlView {
    pub paused: bool,
    /// Nodes stopped and waiting to be started again.
    #[serde(default)]
    pub stopped: Vec<String>,
    /// Nodes with operator text delivered but not yet picked up.
    #[serde(default)]
    pub steering: Vec<String>,
}

/// The control state of one run, and the rules for changing it.
///
/// Lives wherever the dashboard keeps it; this type owns the transitions so that the endpoint, the
/// view and the answer a node gets cannot disagree about what a command meant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunControl {
    paused: bool,
    stopped: Vec<String>,
    steer: Vec<(String, String)>,
}

impl RunControl {
    /// Apply an operator's command.
    pub fn apply(&mut self, command: Command) {
        match command {
            Command::Pause => self.paused = true,
            Command::Resume => self.paused = false,
            Command::Stop { node } => {
                if !self.is_stopped(&node) {
                    self.stopped.push(node);
                }
            }
            Command::Start { node } => self.stopped.retain(|n| !same_node(n, &node)),
            Command::Steer { node, text } => self.steer.push((node, text)),
        }
    }

    /// What `node` should do now, taking any text waiting for it.
    ///
    /// Stop wins over a pause: an operator who stopped a node while the run was paused wants the
    /// node ended, and answering `Hold` would leave it holding a conversation nobody will resume.
    /// Text is handed over either way — a node about to stop has still been spoken to, and the
    /// message belongs in the transcript its checkpoint keeps.
    pub fn poll(&mut self, node: &str) -> Control {
        let steer = self.take_steer(node);
        let directive = if self.is_stopped(node) {
            Directive::Stop
        } else if self.paused {
            Directive::Hold
        } else {
            Directive::Continue
        };
        Control { directive, steer }
    }

    /// Whether `node` is stopped and waiting to be started again.
    pub fn is_stopped(&self, node: &str) -> bool {
        self.stopped.iter().any(|n| same_node(n, node))
    }

    /// Take the text waiting for `node`, oldest first.
    fn take_steer(&mut self, node: &str) -> Vec<String> {
        let mut taken = Vec::new();
        self.steer.retain(|(to, text)| {
            if same_node(to, node) {
                taken.push(text.clone());
                false
            } else {
                true
            }
        });
        taken
    }

    /// What to show about this run.
    pub fn view(&self) -> ControlView {
        let mut steering: Vec<String> = self.steer.iter().map(|(node, _)| node.clone()).collect();
        steering.dedup();
        ControlView {
            paused: self.paused,
            stopped: self.stopped.clone(),
            steering,
        }
    }

    /// Whether this holds nothing worth keeping, so a finished run can be forgotten.
    pub fn is_empty(&self) -> bool {
        !self.paused && self.stopped.is_empty() && self.steer.is_empty()
    }
}

/// Whether two node names are the same node.
fn same_node(a: &str, b: &str) -> bool {
    normalized_node_name(a) == normalized_node_name(b)
}

/// The stable identity shared by every control delivery path for a node.
///
/// Case, and nothing else. A node's name reaches this from two directions — the dashboard sends
/// what the graph drew, a node polls with what it runs as — and both are the same lowercase machine
/// name, so this is tolerance for a control target typed by hand against the API rather than a rule
/// anything depends on. `validate::machine_name` admits only lowercase ASCII, digits and `_`, so
/// lowercasing cannot collide two legal identities.
///
/// Nothing further is folded. A workflow declares stage ids of its own and those are only checked
/// for being distinct as written, so any folding wider than case merges two legal identities into
/// one control address: folding underscores out of every name made `implement_er` and `implementer`
/// one control target, and a stop aimed at one ended the other while steer text went to whichever
/// polled first.
///
/// The durable provider-pause ledger keys its rows on this, so a change here changes durable keys —
/// see `NODE_KEY_SPELLING` in ratatoskr-store.
pub fn normalized_node_name(node: &str) -> String {
    node.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steer(node: &str, text: &str) -> Command {
        Command::Steer {
            node: node.to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn a_paused_run_holds_every_node() {
        let mut control = RunControl::default();
        control.apply(Command::Pause);
        assert_eq!(control.poll("analyst").directive, Directive::Hold);
        assert_eq!(control.poll("implementer").directive, Directive::Hold);

        control.apply(Command::Resume);
        assert_eq!(control.poll("analyst").directive, Directive::Continue);
    }

    #[test]
    fn a_stop_reaches_only_the_node_it_names() {
        // The fork runs two nodes at once. Stopping one must not end the other.
        let mut control = RunControl::default();
        control.apply(Command::Stop {
            node: "implementer".to_string(),
        });
        assert_eq!(control.poll("implementer").directive, Directive::Stop);
        assert_eq!(control.poll("redteam").directive, Directive::Continue);
    }

    #[test]
    fn a_stopped_node_stays_stopped_until_it_is_started() {
        // The directive is what parks the run, so it has to survive being read: a node that polls
        // twice while parking must be told to stop both times.
        let mut control = RunControl::default();
        control.apply(Command::Stop {
            node: "analyst".to_string(),
        });
        assert_eq!(control.poll("analyst").directive, Directive::Stop);
        assert_eq!(control.poll("analyst").directive, Directive::Stop);
        assert!(control.is_stopped("analyst"));

        control.apply(Command::Start {
            node: "analyst".to_string(),
        });
        assert_eq!(control.poll("analyst").directive, Directive::Continue);
        assert!(!control.is_stopped("analyst"));
    }

    #[test]
    fn stopping_a_node_twice_leaves_one_start_to_undo_it() {
        // `stopped` is a set, not a count. Two clicks must not need two starts — the second start
        // would then be a no-op the operator reads as the button not working.
        let mut control = RunControl::default();
        for _ in 0..2 {
            control.apply(Command::Stop {
                node: "analyst".to_string(),
            });
        }
        control.apply(Command::Start {
            node: "analyst".to_string(),
        });
        assert_eq!(control.poll("analyst").directive, Directive::Continue);
    }

    #[test]
    fn text_reaches_its_node_once_and_in_order() {
        let mut control = RunControl::default();
        control.apply(steer("implementer", "first"));
        control.apply(steer("implementer", "second"));
        control.apply(steer("redteam", "not yours"));

        let taken = control.poll("implementer").steer;
        assert_eq!(taken, ["first", "second"]);
        // Delivered once: a message the model sees on every turn reads as the operator repeating
        // themselves, and would keep steering long after it was meant to.
        assert!(control.poll("implementer").steer.is_empty());
        assert_eq!(control.poll("redteam").steer, ["not yours"]);
    }

    #[test]
    fn a_stopped_node_is_still_handed_what_was_said_to_it() {
        let mut control = RunControl::default();
        control.apply(steer("analyst", "look at the ruleset"));
        control.apply(Command::Stop {
            node: "analyst".to_string(),
        });
        let control = control.poll("analyst");
        assert_eq!(control.directive, Directive::Stop);
        assert_eq!(control.steer, ["look at the ruleset"]);
    }

    #[test]
    fn a_control_target_reaches_its_node_whatever_case_it_was_typed_in() {
        // Stage ids are lowercase by construction, so a capital can only come from a target typed
        // by hand against the API — where a command silently addressing nothing is the worst
        // possible answer.
        let mut control = RunControl::default();
        control.apply(Command::Stop {
            node: "RedTeam".to_string(),
        });
        assert_eq!(control.poll("redteam").directive, Directive::Stop);

        control.apply(Command::Start {
            node: "REDTEAM".to_string(),
        });
        assert_eq!(control.poll("redteam").directive, Directive::Continue);

        control.apply(steer("RedTeam", "check the baseline"));
        assert_eq!(control.poll("redteam").steer, ["check the baseline"]);
    }

    #[test]
    fn a_workflow_stage_is_not_the_standard_node_it_is_spelled_like() {
        // A workflow may declare a stage id of its own, and `implement_er` is a legal one. Every
        // name is its own address, underscores included: folding underscores out of every name made
        // this stage the standard implementer's address too, so a stop aimed at one ended the other
        // and steer text went to whichever polled first.
        let mut control = RunControl::default();
        control.apply(Command::Stop {
            node: "implementer".to_string(),
        });
        assert_eq!(control.poll("implement_er").directive, Directive::Continue);

        control.apply(steer("implementer", "not for the workflow's stage"));
        assert!(control.poll("implement_er").steer.is_empty());
        assert_eq!(
            control.poll("implementer").steer,
            ["not for the workflow's stage"]
        );
    }

    #[test]
    fn stopping_beats_pausing() {
        // Both are set. Holding would park a node the operator asked to end.
        let mut control = RunControl::default();
        control.apply(Command::Pause);
        control.apply(Command::Stop {
            node: "analyst".to_string(),
        });
        assert_eq!(control.poll("analyst").directive, Directive::Stop);
    }

    #[test]
    fn the_view_reports_what_the_buttons_should_show() {
        let mut control = RunControl::default();
        assert!(control.is_empty());
        control.apply(Command::Pause);
        control.apply(Command::Stop {
            node: "implementer".to_string(),
        });
        control.apply(steer("redteam", "hi"));

        let view = control.view();
        assert!(view.paused);
        assert_eq!(view.stopped, ["implementer"]);
        assert_eq!(view.steering, ["redteam"]);
        assert!(!control.is_empty());
    }
}
