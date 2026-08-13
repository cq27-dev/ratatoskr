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
    /// The attempt-continuation scope this stage DECLARED, if it declared one. `None` means it
    /// takes whatever its route says, which is nearly every stage.
    ///
    /// The declaration rather than the resolved scope, because that is the half the recorder knows:
    /// the other half is the route, which a reader has from config and which may have been
    /// reconfigured since. Applying one to the other is `Stage::session_scope`, and a reader must
    /// do the same — reading the route alone reports a box on a scope its stages will not run.
    ///
    /// Absent from a recording written before this travelled, which reads as "declared nothing" —
    /// the same thing it means on a stage that declares nothing today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<crate::SessionScope>,
}

/// A recording written before the registry travelled beside the positions: a bare array of nodes,
/// each naming the stages composing it.
///
/// Kept as a decode path, not as a shape anything writes. `shape_json` carries no version to refuse
/// a recording on, so the alternative to converting is not rejecting the run — it is drawing it
/// against no membership at all and saying nothing, which puts every member back as a box of its
/// own with a control address the run never polled. (Where a version *does* exist, refuse instead:
/// `Bundle::version` is why a bundle's fields carry no serde defaults.)
#[derive(Deserialize)]
struct PositionedNode {
    #[serde(flatten)]
    node: ShapeNode,
    #[serde(default)]
    stages: Vec<String>,
}

/// What a run recorded about the graph it executed.
///
/// Serialized into the run's `shape_json`.
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

    /// Which node each stage's records are drawn in, in registry order.
    ///
    /// The accessor for a reader that needs the whole mapping, rather than walking [`Self::stages`]
    /// and reading `.node` off each row. That walk assumes one stage has one node, which is true
    /// today and stops being true when membership is resolved per invocation from span parentage
    /// (#244) — `characterizer` already cannot answer it, which is why it declares none. Asking
    /// through here, and through [`Self::node_of`], is what survives that.
    pub fn membership(&self) -> Vec<(&str, &str)> {
        self.stages
            .iter()
            .map(|stage| (stage.id.as_str(), stage.node.as_str()))
            .collect()
    }

    /// The node a record written under `stage` is drawn in — the stage itself unless it said
    /// otherwise.
    pub fn node_of<'a>(&'a self, stage: &'a str) -> &'a str {
        self.stages
            .iter()
            .find(|known| known.id == stage)
            .map_or(stage, |known| known.node.as_str())
    }

    /// The attempt-continuation scope `stage` declared, if it declared one. `None` means it takes
    /// its route's, which is what [`crate::SessionScope`]'s absence has always meant.
    pub fn session_of(&self, stage: &str) -> Option<crate::SessionScope> {
        self.stages
            .iter()
            .find(|known| known.id == stage)
            .and_then(|known| known.session)
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
            serde_json::from_str::<Vec<PositionedNode>>(raw)
                .ok()
                .map(convert)
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

/// Turn a recording that hung membership off each node into one that records the registry.
///
/// A node's members become its stages. `governed_by` is set to the box for a member that is not the
/// box itself, because that format carried no governance and its reader took a box's route from the
/// box's own name — so this is what the recording meant, stated in the vocabulary that replaced it.
/// Guessing anything else would leave every composed box of every stored run reporting no route.
///
/// A node naming no stages contributes none, which is the recording from before membership existed:
/// [`Recorded::members`] then answers with the node's own name, as it did.
fn convert(placed: Vec<PositionedNode>) -> Recorded {
    Recorded {
        stages: placed
            .iter()
            .flat_map(|placed| {
                placed.stages.iter().map(|id| RunStage {
                    id: id.clone(),
                    node: placed.node.name.clone(),
                    governed_by: (*id != placed.node.name).then(|| placed.node.name.clone()),
                    // That format recorded no declaration, and its reader took the box's scope
                    // straight from the route. `None` is exactly that.
                    session: None,
                })
            })
            .collect(),
        nodes: placed.into_iter().map(|placed| placed.node).collect(),
    }
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
    fn a_recording_that_hung_membership_off_each_node_keeps_it() {
        // The format immediately before this one: a bare array whose nodes each name the stages
        // composing them. Runs recorded by it are in stores now and arrive in bundles, and there is
        // no version on `shape_json` to refuse one on — so dropping the field would not refuse the
        // recording, it would draw the run wrong and say nothing. Its members would come back as
        // boxes of their own, with controls addressed to names the run never polled.
        let placed = r#"[
            {"name":"analyst","stage":0,"lane":0,"optional":false,"stages":["analyst"]},
            {"name":"redteam","stage":1,"lane":0,"optional":false,
             "stages":["redteam_classifier","redteam_author"]}
        ]"#;
        let record = recorded(Some(placed));
        assert_eq!(record.nodes.len(), 2);
        assert_eq!(
            record.members("redteam"),
            ["redteam_classifier", "redteam_author"]
        );
        assert_eq!(record.node_of("redteam_author"), "redteam");
        assert_eq!(record.members("analyst"), ["analyst"]);

        // That format carried no governance, and its reader took a box's route from the box's own
        // name. Converting says so, rather than leaving every composed box of every stored run
        // reporting no route at all: `[models.redteam]` is what such a run's red team ran on.
        assert_eq!(record.governance_of("redteam_author"), "redteam");
        assert_eq!(record.governance_of("analyst"), "analyst");
    }

    #[test]
    fn a_recording_from_before_membership_makes_every_node_its_own_stage() {
        // Older still: positions alone. Its nodes place as they always did, and every one of them is
        // exactly its own stage — which is what they were.
        let bare = r#"[
            {"name":"analyst","stage":0,"lane":0,"optional":false},
            {"name":"redteam","stage":1,"lane":0,"optional":false}
        ]"#;
        let record = recorded(Some(bare));
        assert_eq!(record.nodes.len(), 2);
        assert!(record.stages.is_empty());
        assert_eq!(record.members("redteam"), ["redteam"]);
        assert_eq!(record.governance_of("redteam"), "redteam");
    }
}
