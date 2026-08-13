//! The shape of the graph a run executed: which nodes existed, and where they sat.
//!
//! This lives here, rather than in the dashboard that draws it, because a run has to be able to
//! *record* its own shape. A viewer that reads the shape from its own build can only show runs
//! from that build: change the pipeline, or drop the built-in one, and every run recorded before
//! the change is drawn against a graph it did not execute. An imported run makes that immediate —
//! it may come from a machine whose pipeline this one has never had.
//!
//! So the run writes its shape down when it starts, taken from the layout its workflow declared,
//! and a viewer reads that and nothing else. There is deliberately no compiled-in default to fall
//! back to: a second copy of the pipeline in Rust could only drift from the declaration that
//! actually runs, and a run drawn against it would be drawn against a graph nothing executed.

use serde::{Deserialize, Serialize};

/// One node's place in the graph: `stage` is the column, `lane` the row within it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeNode {
    pub name: String,
    pub stage: usize,
    pub lane: usize,
    /// Whether the node runs at all is a property of configuration, not of the run — the overseer
    /// only runs where a workflow has to be chosen, the verifier and publisher only where the repo
    /// gave them a route. An optional node with no checkpoint has not stalled; it was never asked.
    pub optional: bool,
}

/// Read the shape a run recorded.
///
/// Empty when there is nothing readable to read: a workflow that declared no layout, or a run from
/// before shapes were stored. Nothing is substituted for it — a viewer places such a run's nodes
/// from the records it actually has, which is the most that can be said about where they sat.
pub fn recorded(shape_json: Option<&str>) -> Vec<ShapeNode> {
    shape_json
        .and_then(|raw| serde_json::from_str::<Vec<ShapeNode>>(raw).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_is_drawn_against_the_shape_it_recorded_and_nothing_else() {
        // The case this exists for: a run from a graph this build does not have. Reading it back
        // must give that graph — otherwise its nodes vanish from the view and the run appears to
        // have done nothing.
        let foreign = r#"[
            {"name":"scout","stage":0,"lane":0,"optional":false},
            {"name":"custom","stage":1,"lane":0,"optional":false}
        ]"#;
        let shape = recorded(Some(foreign));
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[1].name, "custom");

        // Nothing to read is an empty shape, not a guess at one. A run whose workflow declared no
        // layout is placed from its own records rather than from a pipeline it may never have run.
        assert!(recorded(None).is_empty());
        assert!(recorded(Some("not json")).is_empty());
        assert!(recorded(Some("[]")).is_empty());
    }
}
