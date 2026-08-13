//! What a workflow may do with a standard identifier.
//!
//! Stage overlay treats every identifier as a registry key, so a declaration named after a standard
//! stage silently becomes that stage. Some of those identifiers are the run's, not a workflow's:
//! Rust invokes them, reads their checkpoints back by name, or deserializes their output into a
//! concrete type. This module is the one table that says which is which, so classifying a new
//! standard stage is a single edit rather than an audit of five scattered lists.
//!
//! Three classes, and every standard identifier is in exactly one:
//!
//! 1. [`Class::Overridable`] — a workflow may declare it. `contract` is the output contract a Rust
//!    adapter deserializes the stage's output as; an override may not change it.
//! 2. [`Class::Reserved`] — a workflow may not declare it at all, for the reason [`Reserved`] names.
//! 3. Absent from [`STANDARD_IDENTIFIERS`] — a repository's own stage id, unconstrained.

/// Why an identifier belongs to the run rather than to a workflow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Reserved {
    /// A Rust-owned workflow operation, installed as a JavaScript global under this name. A stage
    /// declared under it would be overwritten by the operation and never run.
    Operation,
    /// Bookkeeping and delivery. They run from Rust terminal adapters after the run outcome is
    /// accepted, with resources — a push grant, the committed worktree — no workflow operation
    /// holds, so a workflow can neither call one nor change how it is called.
    Terminal,
    /// Selection. It runs before a workflow is chosen, so there is no answer to which workflow's
    /// overseer picks among workflows.
    Selection,
    /// A checkpoint identity the run's lifecycle reads back by name: the iteration ordinal and the
    /// ceiling gate count `implementer` records, `latest_checkpoint` deserializes `implementer`,
    /// `redteam` and `memory` into concrete types, and `finish_full` reads the same rows. A stage
    /// checkpointing arbitrary output under one of these names inflates an iteration count, spends
    /// the ceiling recovery early, or fails deserialization mid-run.
    Lifecycle,
    /// A record the run writes itself, identified by name alone by every reader — `issue_text` in
    /// ratatoskr-serve, the clarification-history check, and the shape API's caller resolution.
    Record,
    /// An internal gate with a fixed capability boundary. Never a configurable stage or agent.
    InternalGate,
    /// The governance identity of a standard stage, with no stage of its own under that name.
    ///
    /// Resolution matches a stage id before falling back to `governedBy`, so a workflow stage
    /// declared under this name wins the lookups made *for* the standard stages that name it —
    /// their enablement, their route, their profile. It could then enable a red team whose own
    /// route resolution has nowhere to go, and the run would die on the name it just introduced.
    ///
    /// `redteam` is a lifecycle checkpoint identity as well — the run records the red team's output
    /// under it and reads it back by name — and is classed here because shadowing the governance
    /// lookups is the sharper of the two hazards and the one worth naming in the refusal.
    GovernanceIdentity,
}

impl Reserved {
    /// The clause that goes in the refusal, after "which is".
    pub(crate) fn because(self) -> &'static str {
        match self {
            Self::Operation => {
                "the name of a Rust-owned workflow operation; choose a different stage identifier"
            }
            Self::Terminal => {
                "a terminal adapter owned by the run rather than a workflow operation"
            }
            Self::Selection => "the selection between workflows, and so cannot be declared by one",
            Self::Lifecycle => {
                "a lifecycle checkpoint identity the run reads back by name; choose a different \
                 stage identifier"
            }
            Self::Record => {
                "a checkpoint the run writes itself; choose a different stage identifier"
            }
            Self::InternalGate => {
                "an internal gate with a fixed capability, not a configurable stage"
            }
            Self::GovernanceIdentity => {
                "the governance identity of the standard red-team stages, which a stage of that \
                 name would answer for; choose a different stage identifier"
            }
        }
    }

    /// Whether the identifier is still a governance identity a declared stage may run under.
    ///
    /// `governedBy` picks a turn's ruleset, its `[models.*]` route, its plugin bindings, its
    /// telemetry attribution and its conversation key. A reserved identifier whose model turn the
    /// workflow itself drives is a real identity to run under, and the bundled definitions use it:
    /// `implementer_attempt` is governed by `implementer`, `context_distillation` by `context`.
    /// One whose turn runs outside any workflow is not — selection happens before a workflow is
    /// chosen and delivery after its outcome is accepted — so a stage governed by `overseer` or
    /// `publisher` would spend that route, and be recorded under that identity, for a turn the
    /// workflow never made.
    fn governable(self) -> bool {
        matches!(
            self,
            Self::Operation | Self::Lifecycle | Self::GovernanceIdentity
        )
    }
}

/// Who invokes a standard stage, and what becomes of its output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Invocation {
    /// Installed as a workflow global under its own id. The executor checkpoints its output there.
    Workflow,
    /// Invoked only from a Rust lifecycle adapter, which supplies a worktree, a shell grant or a
    /// review gate a generic JavaScript host has no way to hold. Never installed as a workflow
    /// global — a workflow that could call one directly could hand itself the record the gate
    /// reads. The adapter has the executor checkpoint the output under this name.
    Adapter,
    /// As [`Self::Adapter`], but the adapter folds the output into another stage's record rather
    /// than checkpointing it: nothing is ever written under this name, and the executor's
    /// checkpoint-scoped work — delegation above all — does not run for it.
    AdapterEvidence,
}

/// What a workflow may do with one standard identifier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Class {
    Overridable {
        /// The output contract a Rust adapter deserializes this stage's output as. An override that
        /// changes it type-errors mid-run, so it is refused at load.
        contract: Option<&'static str>,
        invocation: Invocation,
    },
    Reserved(Reserved),
}

const fn overridable(contract: &'static str) -> Class {
    Class::Overridable {
        contract: Some(contract),
        invocation: Invocation::Workflow,
    }
}

const fn adapter(contract: &'static str) -> Class {
    Class::Overridable {
        contract: Some(contract),
        invocation: Invocation::Adapter,
    }
}

const fn adapter_evidence(contract: &'static str) -> Class {
    Class::Overridable {
        contract: Some(contract),
        invocation: Invocation::AdapterEvidence,
    }
}

/// Every identifier the standard registry, the run's lifecycle or a Rust adapter owns a claim on.
///
/// An identifier absent from this table is class 3: a repository's own stage id, unconstrained.
/// A new standard stage is classified here, once, and every gate follows.
pub(crate) const STANDARD_IDENTIFIERS: &[(&str, Class)] = &[
    // --- class 1: overridable, with the contract the Rust side deserializes ------------------
    ("scout", overridable("ScoutOutput")),
    ("analyst", overridable("AnalystOutput")),
    ("characterizer", overridable("CharacterizerOutput")),
    ("redteam_classifier", overridable("Classification")),
    ("context_distillation", overridable("Distillation")),
    // Rust-invoked: the review gate reads the *last* `verifier` checkpoint, so a workflow able to
    // call `verifier(..)` after `verify()` could answer the gate that judges it.
    ("verifier", adapter("VerifierOutput")),
    // Rust-invoked: write authority inside the prepared worktree, which only the lifecycle adapter
    // owns. A generic host has no worktree from which to derive a safe resource root. Both are
    // evidence for the record their adapter writes — the red team's output and the implementer's
    // iteration — so neither is checkpointed under its own name.
    ("redteam_author", adapter_evidence("AuthoredTests")),
    ("implementer_attempt", adapter_evidence("Report")),
    // --- class 2: not declarable ---------------------------------------------------------------
    ("overseer", Class::Reserved(Reserved::Selection)),
    ("bookkeeper", Class::Reserved(Reserved::Terminal)),
    ("publisher", Class::Reserved(Reserved::Terminal)),
    ("context", Class::Reserved(Reserved::Operation)),
    ("redTeam", Class::Reserved(Reserved::Operation)),
    ("implement", Class::Reserved(Reserved::Operation)),
    ("iterate", Class::Reserved(Reserved::Operation)),
    ("replanAtCeiling", Class::Reserved(Reserved::Operation)),
    ("verify", Class::Reserved(Reserved::Operation)),
    ("isConverged", Class::Reserved(Reserved::Operation)),
    ("testCommandRan", Class::Reserved(Reserved::Operation)),
    // `implementer` and `context`, the other two governance identities the standard stages name,
    // are already reserved above for reasons of their own; `redteam` has no such cover.
    ("redteam", Class::Reserved(Reserved::GovernanceIdentity)),
    ("implementer", Class::Reserved(Reserved::Lifecycle)),
    ("memory", Class::Reserved(Reserved::Lifecycle)),
    ("issue", Class::Reserved(Reserved::Record)),
    ("clarification", Class::Reserved(Reserved::Record)),
    ("referee", Class::Reserved(Reserved::InternalGate)),
];

/// Selection's stage id. Named here because it is where its class is recorded.
pub(crate) const SELECTION_STAGE_ID: &str = "overseer";

/// "Does this output actually deserialize as the Rust type the contract names?"
///
/// [`STANDARD_IDENTIFIERS`] says which contract a stage's output is read back as, and
/// [`validate_declarations`](crate::validate::validate_declarations) refuses an override that
/// changes the contract *name*. Neither says anything about `outputSchema`, which is the thing that
/// actually gates what the model may return — so an override could keep `outputContract:
/// "AnalystOutput"`, declare `outputSchema: { "type": "string" }`, pass every gate, checkpoint a
/// bare string, and fail much later inside `latest_checkpoint` with a serde error that no longer
/// says which stage produced it.
///
/// Deliberately not a schema-compatibility check against the canonical schema: that would forbid
/// legitimate tightening (a `minLength`, an extra `required`) and still not prove the read
/// succeeds. Deserializing is the exact boundary.
///
/// Second table rather than a field on [`Class`] only because a `const` cannot hold a monomorphised
/// function pointer per contract; `every_deserialized_contract_has_a_check` bolts the two together,
/// so adding a standard stage is still a single edit here.
type ContractCheck = fn(&serde_json::Value) -> Result<(), serde_json::Error>;

fn reads_as<T: serde::de::DeserializeOwned>(
    value: &serde_json::Value,
) -> Result<(), serde_json::Error> {
    serde_json::from_value::<T>(value.clone()).map(|_| ())
}

const CONTRACT_CHECKS: &[(&str, ContractCheck)] = &[
    ("ScoutOutput", reads_as::<crate::scout::ScoutOutput>),
    ("AnalystOutput", reads_as::<crate::analyst::AnalystOutput>),
    (
        "CharacterizerOutput",
        reads_as::<crate::testrun::CharacterizerOutput>,
    ),
    ("Classification", reads_as::<crate::redteam::Classification>),
    ("Distillation", reads_as::<crate::context::Distillation>),
    (
        "VerifierOutput",
        reads_as::<crate::verifier::VerifierOutput>,
    ),
    ("AuthoredTests", reads_as::<crate::redteam::AuthoredTests>),
    ("Report", reads_as::<crate::implementer::Report>),
];

/// Refuse a stage's output before it is checkpointed, when Rust reads that stage back as a type.
///
/// `Ok(())` for a workflow's own stage id: those are absent from [`STANDARD_IDENTIFIERS`], nothing
/// deserializes them, and their output shape is theirs to choose.
pub(crate) fn check_typed_output(id: &str, output: &serde_json::Value) -> Result<(), String> {
    let Some(contract) = required_contract(id) else {
        return Ok(());
    };
    let Some((_, check)) = CONTRACT_CHECKS.iter().find(|(name, _)| *name == contract) else {
        return Ok(());
    };
    check(output).map_err(|error| {
        format!("stage `{id}` output does not deserialize as `{contract}`: {error}")
    })
}

fn class(id: &str) -> Option<Class> {
    STANDARD_IDENTIFIERS
        .iter()
        .find(|(name, _)| *name == id)
        .map(|(_, class)| *class)
}

/// Why `id` may not be declared by a workflow, or `None` if it may.
pub(crate) fn reserved(id: &str) -> Option<Reserved> {
    match class(id)? {
        Class::Reserved(reason) => Some(reason),
        Class::Overridable { .. } => None,
    }
}

/// Why `id` may not be a `governedBy`, or `None` if it may.
pub(crate) fn reserved_for_governance(id: &str) -> Option<Reserved> {
    reserved(id).filter(|reason| !reason.governable())
}

/// The identities a run writes a checkpoint under today, with no stage of that name to declare
/// them.
///
/// `latest_checkpoint` reads `implementer` and `redteam` back by name, so a run records under
/// those whatever its registry calls the stages behind them, and a workflow's layout may place them
/// for that reason. A list rather than a predicate because a refusal has to name what *is* allowed
/// to be actionable, and a set the caller can print is the only way acceptance and the message
/// cannot drift apart.
///
/// These are also the names under which a run's *model events* for that work arrive, which is what
/// makes them drawable rather than merely written: an `implementer_attempt` turn is recorded under
/// `implementer`, and `redteam_classifier` and `redteam_author` turns under `redteam`.
///
/// Written out rather than filtered over a [`Class`], because "the run owns this name" and "a run
/// records under this name" are different questions and no one class answers both. `context` is
/// [`Reserved::Operation`] — what its entry records is why a workflow may not *declare* a stage
/// under the name — and the run still checkpoints the merged gather step under it (`context_host`),
/// with `context_distillation`, the model turn inside that step, governing as `context`: both
/// halves of the record arrive under this one name. `memory` is the mirror image, and stays out.
/// It is [`Reserved::Lifecycle`] because `reconstruct_plan` still reads a `memory` checkpoint back
/// for a run recorded before the merged `context` one existed, so no workflow may declare a stage
/// under it — but nothing a run does now writes one, and a column naming it would draw a box that
/// stays empty for the whole run, which is what refusing an undrawable name exists to prevent.
///
/// `bundled_standard_full_sequences_revision_review_and_rust_terminal_actions` bolts this list to
/// what a run of the bundled workflow records, so an entry cannot go stale here unnoticed.
pub(crate) const RUN_CHECKPOINT_IDENTITIES: &[&str] = &["context", "implementer", "redteam"];

/// The output contract an override of `id` must keep, because Rust deserializes it.
pub(crate) fn required_contract(id: &str) -> Option<&'static str> {
    match class(id)? {
        Class::Overridable { contract, .. } => contract,
        Class::Reserved(_) => None,
    }
}

/// Whether a registry stage becomes a JavaScript host under its own id.
///
/// A repository's own stage always does. A standard one does unless the run owns the name outright
/// or invokes it only from a Rust adapter.
pub(crate) fn is_js_host(id: &str) -> bool {
    match class(id) {
        None => true,
        Some(Class::Overridable { invocation, .. }) => invocation == Invocation::Workflow,
        Some(Class::Reserved(_)) => false,
    }
}

/// Whether the run folds this stage's output into another record instead of checkpointing it.
///
/// The executor's checkpoint-scoped behaviour is skipped for such a stage, so anything a declaration
/// asks for there — a `delegation` above all — would be accepted and then never run. Refusing it is
/// [`crate::validate`]'s job; saying which stages those are is this table's.
pub(crate) fn folded_as_evidence(id: &str) -> bool {
    matches!(
        class(id),
        Some(Class::Overridable {
            invocation: Invocation::AdapterEvidence,
            ..
        })
    )
}

/// The clause that goes in a refusal, after "which is" — the [`Reserved::because`] of evidence
/// disposition.
pub(crate) const FOLDED_AS_EVIDENCE_BECAUSE: &str = "run by a Rust adapter as evidence for another stage's checkpoint rather than checkpointed \
     itself, so nothing would consume the child's evidence; delegate from a stage the run \
     checkpoints";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_operation_host_is_reserved_as_one() {
        // The host table and the policy table are separate because one carries behaviour and the
        // other carries classification. This is the bolt between them: a host added there with no
        // entry here would be installable as a global *and* declarable as a stage.
        for (name, _) in crate::workflow::OPERATION_HOSTS {
            assert_eq!(
                reserved(name),
                Some(Reserved::Operation),
                "operation host `{name}` is not classified as one"
            );
        }
    }

    #[test]
    fn every_drawable_run_identity_is_one_no_workflow_can_declare() {
        // A layout may name these though no stage does. Each must therefore be a name a workflow
        // cannot declare a stage under, or one box would mean the run's own record here and some
        // repository's stage there. The converse is deliberately not asserted: `memory` is
        // reserved and not drawable, because nothing a run does records under it.
        for name in RUN_CHECKPOINT_IDENTITIES {
            assert!(
                reserved(name).is_some(),
                "`{name}` is drawable as a run's own record, so a stage must not be able to claim it"
            );
        }
    }

    #[test]
    fn every_standard_identifier_is_classified_once() {
        let mut seen = std::collections::BTreeSet::new();
        for (name, _) in STANDARD_IDENTIFIERS {
            assert!(seen.insert(*name), "`{name}` is classified twice");
        }
    }

    #[test]
    fn every_deserialized_contract_has_a_check() {
        // The bolt between the two tables. A standard stage classified as `Overridable` with a
        // contract nothing here can read would fall through the gate silently, which is precisely
        // the failure the gate exists for.
        for (name, class) in STANDARD_IDENTIFIERS {
            let Class::Overridable {
                contract: Some(contract),
                ..
            } = class
            else {
                continue;
            };
            assert!(
                CONTRACT_CHECKS.iter().any(|(known, _)| known == contract),
                "`{name}` is deserialized as `{contract}`, which nothing here can check"
            );
        }
    }

    #[test]
    fn a_typed_stage_refuses_output_its_type_cannot_read() {
        // The schema is what gates a stage's output, and an override may declare its own while
        // keeping the contract name — so the name alone never proved the checkpoint is readable.
        let error = check_typed_output("analyst", &serde_json::json!("a bare string"))
            .expect_err("a string is not an AnalystOutput");
        assert!(error.contains("analyst"), "{error}");
        assert!(error.contains("AnalystOutput"), "{error}");
        assert!(
            check_typed_output(
                "analyst",
                &serde_json::json!({ "impact_summary": "it fits" })
            )
            .is_ok()
        );
        // A workflow's own stage id shapes its output however it likes.
        assert!(check_typed_output("security_review", &serde_json::json!("anything")).is_ok());
    }

    #[test]
    fn a_repositorys_own_identifier_is_unconstrained() {
        assert!(reserved("security_review").is_none());
        assert!(required_contract("security_review").is_none());
        assert!(is_js_host("security_review"));
    }

    #[test]
    fn selection_and_the_review_gate_are_not_reachable_from_javascript() {
        assert!(!is_js_host("overseer"));
        assert!(!is_js_host("verifier"));
        assert!(!is_js_host("bookkeeper"));
        assert!(!is_js_host("publisher"));
        assert!(!is_js_host("redteam_author"));
        assert!(!is_js_host("implementer_attempt"));
    }

    #[test]
    fn an_operation_identity_may_still_govern_a_stage_but_a_terminal_one_may_not() {
        // `implementer_attempt` is governed by `implementer`, `context_distillation` by `context`
        // and both red-team stages by `redteam` in the bundled definitions, so reserving those ids
        // must not bar them here.
        assert!(reserved_for_governance("implementer").is_none());
        assert!(reserved_for_governance("context").is_none());
        assert!(reserved_for_governance("redteam").is_none());
        assert_eq!(
            reserved_for_governance("publisher"),
            Some(Reserved::Terminal)
        );
        assert_eq!(
            reserved_for_governance("overseer"),
            Some(Reserved::Selection)
        );
    }
}
