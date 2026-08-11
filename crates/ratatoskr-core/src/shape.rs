//! The shape of the graph a run executed: which nodes existed, and where they sat.
//!
//! This lives here, rather than in the dashboard that draws it, because a run has to be able to
//! *record* its own shape. A viewer that reads the shape from its own build can only show runs
//! from that build: change the pipeline, or drop the built-in one, and every run recorded before
//! the change is drawn against a graph it did not execute. An imported run makes that immediate —
//! it may come from a machine whose pipeline this one has never had.
//!
//! So the run writes its shape down when it starts, and a viewer prefers what the run recorded to
//! anything it knows itself. [`BUILT_IN`] is only the default a run adopts when it has no other,
//! and the fallback for runs recorded before shapes were stored.

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

/// One column of the pipeline. The fork is a single stage with two nodes.
pub struct Stage {
    pub nodes: &'static [&'static str],
    pub optional: bool,
}

const fn required(nodes: &'static [&'static str]) -> Stage {
    Stage {
        nodes,
        optional: false,
    }
}

const fn optional(nodes: &'static [&'static str]) -> Stage {
    Stage {
        nodes,
        optional: true,
    }
}

/// The pipeline this build runs, in execution order.
///
/// Not the definition of "the pipeline" — the definition of *this build's default*. A run records
/// what it actually ran, and that recording is what anything reading it afterwards should use.
pub const BUILT_IN: &[Stage] = &[
    optional(&["overseer"]),
    required(&["context"]),
    required(&["analyst"]),
    required(&["red_team", "implementer"]),
    optional(&["verifier"]),
    // The run's two deliveries: one writes to the memory graph, the other to the tracker. Neither
    // needs the other's result, so `run_full` reaches them together. The publisher is opt-in; the
    // bookkeeper always runs. Their in-flight activity comes from the live event stream rather
    // than checkpoint-derived pipeline state, which only proves that an attempt finished.
    required(&["bookkeeper", "publisher"]),
];

/// The built-in pipeline as a recordable shape.
pub fn built_in() -> Vec<ShapeNode> {
    BUILT_IN
        .iter()
        .enumerate()
        .flat_map(|(stage, s)| {
            s.nodes
                .iter()
                .enumerate()
                .map(move |(lane, name)| ShapeNode {
                    name: (*name).to_string(),
                    stage,
                    lane,
                    optional: s.optional,
                })
        })
        .collect()
}

/// Read a recorded shape, falling back to this build's own when there is none to read.
///
/// A run recorded before shapes were stored has no shape of its own; the built-in is the best
/// guess available and was, for those runs, the truth.
pub fn recorded_or_built_in(shape_json: Option<&str>) -> Vec<ShapeNode> {
    shape_json
        .and_then(|raw| serde_json::from_str::<Vec<ShapeNode>>(raw).ok())
        .filter(|nodes| !nodes.is_empty())
        .unwrap_or_else(built_in)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_shape_places_the_fork_side_by_side() {
        let shape = built_in();
        let red = shape.iter().find(|n| n.name == "red_team").unwrap();
        let imp = shape.iter().find(|n| n.name == "implementer").unwrap();
        assert_eq!(red.stage, imp.stage, "the fork is one stage");
        assert_ne!(red.lane, imp.lane, "in two lanes");
        assert!(
            shape
                .iter()
                .find(|n| n.name == "overseer")
                .unwrap()
                .optional
        );
        assert!(!shape.iter().find(|n| n.name == "context").unwrap().optional);
    }

    #[test]
    fn a_run_is_drawn_against_the_shape_it_recorded_not_this_builds() {
        // The case this exists for: a run from a graph this build does not have. Reading it back
        // must give that graph, not the local one — otherwise its nodes vanish from the view and
        // the run appears to have done nothing.
        let foreign = r#"[
            {"name":"scout","stage":0,"lane":0,"optional":false},
            {"name":"custom","stage":1,"lane":0,"optional":false}
        ]"#;
        let shape = recorded_or_built_in(Some(foreign));
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[1].name, "custom");

        // No recording, or an unreadable one, falls back rather than showing an empty graph.
        assert_eq!(recorded_or_built_in(None), built_in());
        assert_eq!(recorded_or_built_in(Some("not json")), built_in());
        assert_eq!(recorded_or_built_in(Some("[]")), built_in());
    }
}
