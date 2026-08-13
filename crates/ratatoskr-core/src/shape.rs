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
    /// The stages whose work this node is, in declaration order.
    ///
    /// One box, and the turns inside it. A node that is a single stage names that stage and nothing
    /// else; one composed of several — the red team's classifier and its test author — names each,
    /// because they run on different profiles with different tool sets and each records its own
    /// turn. Reading a box's cost means totalling these, and drawing it means folding their events
    /// into one box rather than tacking each on as a node of its own.
    ///
    /// Recorded with the shape, for the same reason the shape is recorded at all: which stages
    /// compose a node is a property of the workflow that ran, and a viewer reading it from its own
    /// build would read an imported run against a composition nobody executed. Empty for a run
    /// recorded before this was carried, which reads as "this box is just its own name".
    #[serde(default)]
    pub stages: Vec<String>,
    /// Whether the node runs at all is a property of configuration, not of the run — the overseer
    /// only runs where a workflow has to be chosen, the verifier and publisher only where the repo
    /// gave them a route. An optional node with no checkpoint has not stalled; it was never asked.
    pub optional: bool,
}

/// Read the shape a run recorded.
///
/// Empty when there is nothing readable to read: a workflow that declared no layout, a run from
/// before shapes were stored, or a recording whose positions are not positions. Nothing is
/// substituted for it — a viewer places such a run's nodes from the records it actually has, which
/// is the most that can be said about where they sat.
///
/// The bound is here rather than at the writer because a shape does not only arrive from a workflow
/// this machine validated: an imported bundle carries one another machine recorded, and it is
/// written to the store as it came. A reader sizing anything from `stage` — grouping into columns
/// is the obvious one — would be sizing it from a number the run's author chose. Positions index a
/// shape's own nodes, so one at or past their count is not a position, and the whole recording is
/// unreadable rather than partly trusted.
pub fn recorded(shape_json: Option<&str>) -> Vec<ShapeNode> {
    let nodes: Vec<ShapeNode> = shape_json
        .and_then(|raw| serde_json::from_str::<Vec<ShapeNode>>(raw).ok())
        .unwrap_or_default();
    if nodes
        .iter()
        .any(|node| node.stage >= nodes.len() || node.lane >= nodes.len())
    {
        return Vec::new();
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_past_the_shape_it_indexes_makes_the_whole_recording_unreadable() {
        // An imported bundle's shape was validated on the machine that wrote it, and is stored as
        // it arrived. A reader that groups by `stage` sizes that grouping from the number in the
        // record, so a shape claiming a position no node could occupy is refused outright rather
        // than placed — the run is then drawn from its own records, as any unshaped run is.
        let far = r#"[{"name":"x","stage":1000000000,"lane":0,"optional":false}]"#;
        assert!(
            recorded(Some(far)).is_empty(),
            "a position must index this shape's own nodes"
        );

        let saturated = format!(
            r#"[{{"name":"x","stage":{},"lane":0,"optional":false}}]"#,
            usize::MAX
        );
        assert!(
            recorded(Some(&saturated)).is_empty(),
            "and must not be one that overflows"
        );

        let lane = r#"[{"name":"x","stage":0,"lane":9000,"optional":false}]"#;
        assert!(recorded(Some(lane)).is_empty(), "a lane is a position too");

        // The fork: two nodes, one column, two lanes. Every index is inside the node count, which
        // is what a shape this build wrote always looks like.
        let real = r#"[
            {"name":"redteam","stage":0,"lane":0,"optional":false},
            {"name":"implementer","stage":0,"lane":1,"optional":false}
        ]"#;
        assert_eq!(recorded(Some(real)).len(), 2);
    }

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
