//! The graph a run executed: which stages it had, which node's work each is, and where those
//! nodes sat.
//!
//! This lives here, rather than in the dashboard that draws it, because a run has to be able to
//! *record* its own graph. A viewer that reads it from its own build can only show runs from that
//! build: change the pipeline, or drop the built-in one, and every run recorded before the change
//! is drawn against a graph it did not execute. An imported run makes that immediate — it may come
//! from a machine whose pipeline this one has never had.
//!
//! So the run writes it down when it starts and a viewer reads that and nothing else. There is
//! deliberately no compiled-in default to fall back to: a second copy of the pipeline in Rust could
//! only drift from the declaration that actually runs, and a run drawn against it would be drawn
//! against a graph nothing executed.
//!
//! Two facts, from two sources, and keeping them apart is the point. Where a node sits comes from
//! the workflow's `layout`, which is optional — a workflow that declares none records no positions,
//! and a viewer places such a run's nodes from the records it has. Which stages compose a node
//! comes from the *registry*, which every run has. Hanging membership off the positions is how a
//! layout-less run came to record none at all, and then drew each member as a box of its own with
//! controls addressed to a name the runtime never polls.

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

/// One stage of the registry a run executed, and the node whose work it is.
///
/// A stage records under its own identity, so the members of a composed node — the red team's
/// classifier and its test author — write turns and emit events under names no column carries.
/// This is what says they belong in one box: without it a reader draws each beside the node it is
/// part of, and aims that box's controls at a name nothing answers to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunStage {
    pub id: String,
    /// The node this stage's work is drawn in. Its own id unless it declared otherwise.
    pub node: String,
    /// The identity this stage's `[models.*]` route, ruleset and plugin bindings resolve under.
    /// `None` for a stage that governs as itself, which is nearly all of them.
    ///
    /// Recorded because a box's route is its stages', and a box need not be one of them: the
    /// implementer's box runs `models.implementer` through `implementer_attempt`, and a stage drawn
    /// under its own id may still govern as something else. Reading a route under the box's own
    /// name reports the wrong one or none at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub governed_by: Option<String>,
}

/// What a run recorded about the graph it executed.
///
/// Serialized into the run's `shape_json`. A recording that is a bare array of [`ShapeNode`] is one
/// written before the registry travelled with it, and reads as positions with no membership — every
/// node exactly its own stage, which is what such a run's nodes were.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recorded {
    /// Where each node sat. Empty for a workflow that declared no layout.
    #[serde(default)]
    pub nodes: Vec<ShapeNode>,
    /// Every stage the run could execute, and the node each belongs to.
    #[serde(default)]
    pub stages: Vec<RunStage>,
}

impl Recorded {
    /// The stages whose work `node` is, in registry order.
    ///
    /// A node no stage claims is exactly itself: a name from a recording that carries no registry,
    /// and the ordinary answer for every node that is one stage.
    pub fn members(&self, node: &str) -> Vec<String> {
        let members: Vec<String> = self
            .stages
            .iter()
            .filter(|stage| stage.node == node)
            .map(|stage| stage.id.clone())
            .collect();
        match members.is_empty() {
            true => vec![node.to_string()],
            false => members,
        }
    }

    /// The node a record written under `stage` is drawn in — the stage itself unless it said
    /// otherwise.
    pub fn node_of<'a>(&'a self, stage: &'a str) -> &'a str {
        self.stages
            .iter()
            .find(|known| known.id == stage)
            .map_or(stage, |known| known.node.as_str())
    }

    /// The identity `stage`'s route is configured under — the stage itself unless it said
    /// otherwise, and the stage itself for a name no recorded registry knows.
    pub fn governance_of<'a>(&'a self, stage: &'a str) -> &'a str {
        self.stages
            .iter()
            .find(|known| known.id == stage)
            .and_then(|known| known.governed_by.as_deref())
            .unwrap_or(stage)
    }
}

/// Read the graph a run recorded.
///
/// Empty when there is nothing readable to read: a run from before this was stored, or a recording
/// whose positions are not positions. Nothing is substituted for it — a viewer places such a run's
/// nodes from the records it actually has, which is the most that can be said about where they sat.
///
/// The bound is here rather than at the writer because a recording does not only arrive from a
/// workflow this machine validated: an imported bundle carries one another machine wrote, and it is
/// stored as it came. A reader sizing anything from `stage` — grouping into columns is the obvious
/// one — would be sizing it from a number the run's author chose. Positions index a recording's own
/// nodes, so one at or past their count is not a position, and the placement is dropped whole
/// rather than partly trusted. Membership survives it: it indexes nothing, and a run whose columns
/// are unreadable still draws its boxes out of the right stages.
pub fn recorded(shape_json: Option<&str>) -> Recorded {
    let Some(raw) = shape_json else {
        return Recorded::default();
    };
    let mut record = serde_json::from_str::<Recorded>(raw)
        .ok()
        .or_else(|| {
            serde_json::from_str::<Vec<ShapeNode>>(raw)
                .ok()
                .map(|nodes| Recorded {
                    nodes,
                    stages: Vec::new(),
                })
        })
        .unwrap_or_default();
    if record
        .nodes
        .iter()
        .any(|node| node.stage >= record.nodes.len() || node.lane >= record.nodes.len())
    {
        record.nodes = Vec::new();
    }
    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_past_the_shape_it_indexes_makes_the_whole_placement_unreadable() {
        // An imported bundle's recording was validated on the machine that wrote it, and is stored
        // as it arrived. A reader that groups by `stage` sizes that grouping from the number in the
        // record, so a recording claiming a position no node could occupy is refused outright
        // rather than placed — the run is then drawn from its own records, as any unplaced run is.
        let far = r#"{"nodes":[{"name":"x","stage":1000000000,"lane":0,"optional":false}]}"#;
        assert!(
            recorded(Some(far)).nodes.is_empty(),
            "a position must index this recording's own nodes"
        );

        let saturated = format!(
            r#"{{"nodes":[{{"name":"x","stage":{},"lane":0,"optional":false}}]}}"#,
            usize::MAX
        );
        assert!(
            recorded(Some(&saturated)).nodes.is_empty(),
            "and must not be one that overflows"
        );

        let lane = r#"{"nodes":[{"name":"x","stage":0,"lane":9000,"optional":false}]}"#;
        assert!(
            recorded(Some(lane)).nodes.is_empty(),
            "a lane is a position too"
        );

        // Membership indexes nothing, so an unreadable placement does not cost it: the run is drawn
        // from its records, and those still fold into the boxes the registry says they belong to.
        let with_stages = r#"{
            "nodes":[{"name":"x","stage":40,"lane":0,"optional":false}],
            "stages":[{"id":"x_turn","node":"x"}]
        }"#;
        let record = recorded(Some(with_stages));
        assert!(record.nodes.is_empty());
        assert_eq!(record.members("x"), ["x_turn"]);

        // The fork: two nodes, one column, two lanes. Every index is inside the node count, which
        // is what a recording this build writes always looks like.
        let real = r#"{"nodes":[
            {"name":"redteam","stage":0,"lane":0,"optional":false},
            {"name":"implementer","stage":0,"lane":1,"optional":false}
        ]}"#;
        assert_eq!(recorded(Some(real)).nodes.len(), 2);
    }

    #[test]
    fn a_run_is_drawn_against_the_graph_it_recorded_and_nothing_else() {
        // The case this exists for: a run from a graph this build does not have. Reading it back
        // must give that graph — otherwise its nodes vanish from the view and the run appears to
        // have done nothing.
        let foreign = r#"{"nodes":[
            {"name":"scout","stage":0,"lane":0,"optional":false},
            {"name":"custom","stage":1,"lane":0,"optional":false}
        ]}"#;
        let record = recorded(Some(foreign));
        assert_eq!(record.nodes.len(), 2);
        assert_eq!(record.nodes[1].name, "custom");

        // Nothing to read is an empty recording, not a guess at one. A run whose workflow declared
        // no layout is placed from its own records rather than from a pipeline it may never have
        // run.
        assert_eq!(recorded(None), Recorded::default());
        assert_eq!(recorded(Some("not json")), Recorded::default());
        assert_eq!(recorded(Some("[]")), Recorded::default());
        assert_eq!(recorded(Some("{}")), Recorded::default());
    }

    #[test]
    fn membership_is_read_from_the_registry_and_not_from_where_a_node_was_placed() {
        // The layout-less case: no positions at all, and the boxes are still known. This is the
        // whole reason the two are recorded apart — a run of a workflow that declares no layout has
        // a registry like any other, and its composed nodes have to draw as one box each.
        let record = recorded(Some(
            r#"{"stages":[
                {"id":"redteam_classifier","node":"redteam"},
                {"id":"redteam_author","node":"redteam"},
                {"id":"analyst","node":"analyst"}
            ]}"#,
        ));
        assert!(record.nodes.is_empty());
        assert_eq!(
            record.members("redteam"),
            ["redteam_classifier", "redteam_author"]
        );
        assert_eq!(record.members("analyst"), ["analyst"]);
        assert_eq!(record.node_of("redteam_author"), "redteam");
        assert_eq!(record.node_of("analyst"), "analyst");
        // A name the registry never heard of is its own box, which is what an unplaced record of
        // one is.
        assert_eq!(record.members("clarification"), ["clarification"]);
        assert_eq!(record.node_of("clarification"), "clarification");
    }

    #[test]
    fn a_recording_written_before_the_registry_travelled_with_it_still_places_its_nodes() {
        // A bare array is what runs recorded when the shape was positions alone. Its nodes place as
        // they always did, and every one of them is exactly its own stage — which is what they were.
        let legacy = r#"[
            {"name":"analyst","stage":0,"lane":0,"optional":false},
            {"name":"redteam","stage":1,"lane":0,"optional":false,"stages":["redteam_classifier"]}
        ]"#;
        let record = recorded(Some(legacy));
        assert_eq!(record.nodes.len(), 2);
        assert!(record.stages.is_empty());
        assert_eq!(record.members("redteam"), ["redteam"]);
    }
}
