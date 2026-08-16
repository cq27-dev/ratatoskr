//! A run, packaged so somebody else can analyse it.
//!
//! A run is only comparable with what it carries: the row says how it ended, the checkpoints say
//! what each node produced and cost, and the events say when everything happened. A bundle without
//! the events would import as a run you can total up but not look through, which is most of the
//! reason to send one.
//!
//! BSON rather than JSON: this is machine-to-machine, read back whole, and typed — an `i64` count
//! survives the round trip as an `i64` rather than as whatever a JSON reader decides a number is.

use serde::{Deserialize, Serialize};

use crate::{Checkpoint, EventRow, Run, Store, StoreError};

/// What one exported run carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBundle {
    pub run: Run,
    pub checkpoints: Vec<Checkpoint>,
    pub events: Vec<EventRow>,
}

/// One or more runs, with a note of where they came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bundle {
    /// Format version. Present so a future reader can refuse a bundle it does not understand
    /// rather than mis-reading one — the first thing a format regrets not having.
    pub version: u32,
    /// Who exported it, for the `origin` an imported run is recorded under.
    pub exported_by: String,
    pub exported_at: String,
    pub runs: Vec<RunBundle>,
}

/// Refuse a run whose executions do not form a graph, before any of it is written.
///
/// One span id names one span: it is what a consumer resolves a parent against and what tells two
/// invocations of a stage apart, so two rows claiming one id collapse two executions and leave the
/// reader no way to notice. And parentage must terminate — a consumer walking from a record to the
/// execution that drove it has to arrive somewhere. A cycle of any length is what stops that, not
/// only the one-edge case where an execution is its own parent.
///
/// A parent this run has no record for is NOT a cycle and not an error: most parents are host calls
/// and nested turns, which announce themselves in the event stream and write no checkpoint of their
/// own. Walking simply ends there.
///
/// Checked at the boundary rather than trusted, because a bundle is the one checkpoint path this
/// process did not produce.
fn check_execution_graph(
    run_id: &str,
    checkpoints: &[Checkpoint],
    events: &[EventRow],
) -> Result<(), StoreError> {
    use ratatoskr_core::span::SpanId;
    let bad = |problem: String| StoreError::BadExecutionGraph {
        run_id: run_id.to_string(),
        problem,
    };

    let mut parents = std::collections::HashMap::new();
    // What one record says about an execution's parentage, folded with what the others said.
    //
    // A record that omits the field says NOTHING about parentage — not that there is none. Many
    // records describe one execution and they need not each repeat it, so only two records naming
    // DIFFERENT parents are a disagreement. The absence a reader reads as "the run drove this" is a
    // checkpoint row's, which is one record and the whole of what that row states.
    let mut claim = |span_id: SpanId, parent: Option<SpanId>| {
        let known = parents.entry(span_id).or_insert(parent);
        match (*known, parent) {
            (Some(before), Some(now)) if before != now => Err(bad(format!(
                "execution {span_id} is described twice with different parentage"
            ))),
            (None, Some(now)) => {
                *known = Some(now);
                Ok(())
            }
            _ => Ok(()),
        }
    };

    // A ROW per execution, though. Two rows claiming one execution collapse two of them into one
    // with nothing to notice it by, which is what an identity is for.
    let mut rows = std::collections::HashSet::new();
    for checkpoint in checkpoints {
        if let Some(invocation) = checkpoint.invocation {
            if !rows.insert(invocation.span_id) {
                return Err(bad(format!(
                    "execution {} names more than one record",
                    invocation.span_id
                )));
            }
            claim(invocation.span_id, invocation.parent_span_id)?;
        }
    }
    // And the executions that exist only as events. Most of them do: a host call, a clarification,
    // a turn whose failure a workflow recovered from and an answerer all announce themselves and
    // write no checkpoint, so validating rows alone leaves the larger half of a run's execution
    // graph unchecked — including the parents the rows point at.
    for event in events {
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.payload_json) else {
            continue;
        };
        let written = |key: &str| payload.get(key).and_then(serde_json::Value::as_str);
        // Present and unreadable is not absent. Absent says nothing; a value nobody can parse says
        // something that cannot be checked, and recording it as a root asserts a shape the bundle
        // never claimed — the same rule the checkpoint reader follows.
        let id = |key: &str| match written(key) {
            None => Ok(None),
            Some(hex) => SpanId::parse(hex).map(Some).ok_or_else(|| {
                bad(format!(
                    "an execution names `{hex}`, which is not an execution"
                ))
            }),
        };
        if let Some(span_id) = id("span_id")? {
            claim(span_id, id("parent_span_id")?)?;
        }
    }

    // Walk each execution's ancestry. Everything already walked is known to terminate, so each
    // execution is visited once and the whole check is linear in the run.
    //
    // Two records of one walk, because they answer different questions: the set says whether this
    // walk has been here, in one comparison rather than one per step already taken, and the list
    // says in what order — which is the only thing that makes a reported cycle readable. Scanning
    // the list for membership made the walk quadratic in a chain's length, which is exactly the
    // shape an unbounded bundle would have to exploit.
    let mut terminates = std::collections::HashSet::new();
    for start in parents.keys() {
        let mut walked = Vec::new();
        let mut on_this_walk = std::collections::HashSet::new();
        let mut at = *start;
        loop {
            if terminates.contains(&at) {
                break;
            }
            if !on_this_walk.insert(at) {
                return Err(bad(format!(
                    "executions {} form a cycle, so nothing they contain has a root",
                    walked
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" → ")
                )));
            }
            walked.push(at);
            // A parent with no record of its own — a host call, a nested turn — ends the walk.
            match parents.get(&at).copied().flatten() {
                Some(parent) => at = parent,
                None => break,
            }
        }
        terminates.extend(walked);
    }
    Ok(())
}

/// The version this build writes and is willing to read.
///
/// One version is one shape: a bundle claiming this version carries every field of it, so adding a
/// field is a version bump rather than a defaulted key the reader has to guess at.
pub const FORMAT_VERSION: u32 = 3;

/// What an import did, per run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imported {
    pub run_id: String,
    /// False when a run with this id was already present and was left alone.
    pub inserted: bool,
    pub checkpoints: usize,
    pub events: usize,
}

impl Store {
    /// Package `run_ids` for export. Unknown ids are skipped rather than failing the export.
    pub async fn export(
        &self,
        run_ids: &[String],
        by: &str,
        at: &str,
    ) -> Result<Bundle, StoreError> {
        let mut runs = Vec::new();
        for id in run_ids {
            let Some(mut run) = self.run(id).await? else {
                continue;
            };
            let mut one = [run.clone()];
            self.attach_tags(&mut one).await?;
            run.tags = std::mem::take(&mut one[0].tags);
            runs.push(RunBundle {
                checkpoints: self.checkpoints_for_run(id).await?,
                events: self.events_for_run(id).await?,
                run,
            });
        }
        Ok(Bundle {
            version: FORMAT_VERSION,
            exported_by: by.to_string(),
            exported_at: at.to_string(),
            runs,
        })
    }

    /// Take in an exported bundle.
    ///
    /// A run id already present is left untouched and reported rather than merged: two runs with
    /// the same id are either the same run — in which case there is nothing to do — or a collision,
    /// where overwriting would destroy local work to make room for a copy.
    pub async fn import(&self, bundle: &Bundle) -> Result<Vec<Imported>, StoreError> {
        // Its own version and no other. One version is one shape, and the shape includes what a
        // record MEANS: a `span_end` that carries no parent now says the run drove that execution,
        // where the version-2 exporter simply left the field off. Reading one as the other would
        // report a nested execution as top-level, or refuse a bundle that is fine — and a reader
        // that guesses which is worse than one that declines.
        if bundle.version != FORMAT_VERSION {
            return Err(StoreError::Unsupported {
                found: bundle.version,
            });
        }
        let mut report = Vec::new();
        for one in &bundle.runs {
            let id = one.run.run_id.as_str();
            if self.run(id).await?.is_some() {
                report.push(Imported {
                    run_id: id.to_string(),
                    inserted: false,
                    checkpoints: 0,
                    events: 0,
                });
                continue;
            }
            check_execution_graph(id, &one.checkpoints, &one.events)?;
            self.insert_imported_run(&one.run, &bundle.exported_by)
                .await?;
            for c in &one.checkpoints {
                self.insert_checkpoint(crate::CheckpointWrite {
                    run_id: id,
                    node_name: &c.node_name,
                    output_json: &c.output_json,
                    input_json: c.input_json.as_deref(),
                    iteration: c.iteration,
                    invocation: c.invocation,
                    telemetry: c.telemetry.clone(),
                })
                .await?;
            }
            let events = self.ingest_events(id, one.events.clone()).await?;
            if !one.run.tags.is_empty() {
                self.tag_run(id, one.run.tags.clone()).await?;
            }
            report.push(Imported {
                run_id: id.to_string(),
                inserted: true,
                checkpoints: one.checkpoints.len(),
                events,
            });
        }
        Ok(report)
    }
}

/// Serialize a bundle to BSON bytes.
pub fn to_bytes(bundle: &Bundle) -> Result<Vec<u8>, StoreError> {
    bson::serialize_to_vec(bundle).map_err(|e| StoreError::Bundle(e.to_string()))
}

/// Read a bundle back from BSON bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<Bundle, StoreError> {
    bson::deserialize_from_slice(bytes).map_err(|e| StoreError::Bundle(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CheckpointWrite;

    async fn seeded() -> Store {
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("r1", Some("91"), "converged")
            .await
            .unwrap();
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "r1",
                node_name: "analyst",
                output_json: r#"{"impact_summary":"x"}"#,
                ..Default::default()
            })
            .await
            .unwrap();
        store
            .ingest_events(
                "r1",
                vec![EventRow {
                    seq: 0,
                    at: "2026-08-07T10:00:00Z".into(),
                    kind: "node_start".into(),
                    node: Some("analyst".into()),
                    payload_json: r#"{"kind":"node_start","node":"analyst"}"#.into(),
                }],
            )
            .await
            .unwrap();
        store.tag_run("r1", vec!["arm-a".into()]).await.unwrap();
        store
    }

    /// A bundle carrying one run with these checkpoints.
    fn bundle_of(checkpoints: Vec<Checkpoint>) -> Bundle {
        Bundle {
            version: FORMAT_VERSION,
            exported_at: "2026-08-15T00:00:00Z".into(),
            exported_by: "test".into(),
            runs: vec![RunBundle {
                run: crate::Run {
                    run_id: "imported".into(),
                    issue_id: None,
                    status: "converged".into(),
                    updated_at: "2026-08-15T00:00:00Z".into(),
                    config_json: None,
                    graph_hash: None,
                    repo_sha: None,
                    image_digest: None,
                    origin: None,
                    shape_json: None,
                    tags: Vec::new(),
                },
                checkpoints,
                events: Vec::new(),
            }],
        }
    }

    fn imported_checkpoint(node: &str, invocation: ratatoskr_core::span::Invocation) -> Checkpoint {
        Checkpoint {
            node_name: node.into(),
            output_json: "{}".into(),
            created_at: "2026-08-15T00:00:00Z".into(),
            input_json: None,
            iteration: None,
            invocation: Some(invocation),
            telemetry: Default::default(),
        }
    }

    #[tokio::test]
    async fn a_bundle_whose_executions_are_not_a_graph_is_refused_before_anything_is_written() {
        // One span id names one span: it is what a parent reference resolves against and what tells
        // two invocations of a stage apart. Two rows claiming one id collapse two executions with
        // nothing to notice it by, and an execution invoked by itself is a cycle a consumer walking
        // to a root would not escape. A bundle is the one checkpoint path this process did not
        // produce, so it is checked rather than trusted.
        use ratatoskr_core::span::{Invocation, SpanId};
        let shared = SpanId::parse("00000000000000a1").unwrap();

        let store = Store::open_in_memory().unwrap();
        let twice = bundle_of(vec![
            imported_checkpoint("analyst", Invocation::root(shared)),
            imported_checkpoint("implementer", Invocation::root(shared)),
        ]);
        assert!(matches!(
            store.import(&twice).await,
            Err(StoreError::BadExecutionGraph { .. })
        ));

        let cycle = bundle_of(vec![imported_checkpoint(
            "analyst",
            Invocation {
                span_id: shared,
                parent_span_id: Some(shared),
            },
        )]);
        assert!(matches!(
            store.import(&cycle).await,
            Err(StoreError::BadExecutionGraph { .. })
        ));

        // And a cycle that is not a self-edge. A→B→A has two distinct ids and neither execution is
        // its own parent, yet a consumer walking to a root goes round it forever.
        let a = SpanId::parse("00000000000000aa").unwrap();
        let b = SpanId::parse("00000000000000bb").unwrap();
        let ring = bundle_of(vec![
            imported_checkpoint(
                "analyst",
                Invocation {
                    span_id: a,
                    parent_span_id: Some(b),
                },
            ),
            imported_checkpoint(
                "implementer",
                Invocation {
                    span_id: b,
                    parent_span_id: Some(a),
                },
            ),
        ]);
        assert!(matches!(
            store.import(&ring).await,
            Err(StoreError::BadExecutionGraph { .. })
        ));

        // Refused before anything is written: a run half-imported is worse than one refused.
        assert!(store.run("imported").await.unwrap().is_none());

        // An execution that exists only as EVENTS is checked too. Most of a run's executions do: a
        // host call, a clarification, an answerer and a recovered failure all announce themselves
        // and write no checkpoint, so validating rows alone leaves the larger half unchecked.
        let lifecycle = |span: &str, parent: &str| EventRow {
            seq: 0,
            at: "2026-08-15T00:00:00Z".into(),
            kind: "span_start".into(),
            node: None,
            payload_json: format!(
                r#"{{"kind":"span_start","span_id":"{span}","parent_span_id":"{parent}"}}"#
            ),
        };
        let mut ring = bundle_of(vec![]);
        ring.runs[0].run.run_id = "ring".into();
        ring.runs[0].events = vec![
            lifecycle("00000000000000aa", "00000000000000bb"),
            lifecycle("00000000000000bb", "00000000000000aa"),
        ];
        assert!(matches!(
            store.import(&ring).await,
            Err(StoreError::BadExecutionGraph { .. })
        ));

        // One execution described twice, disagreeing about what invoked it.
        let mut split = bundle_of(vec![]);
        split.runs[0].run.run_id = "split".into();
        split.runs[0].events = vec![
            lifecycle("00000000000000cc", "00000000000000aa"),
            lifecycle("00000000000000cc", "00000000000000bb"),
        ];
        assert!(matches!(
            store.import(&split).await,
            Err(StoreError::BadExecutionGraph { .. })
        ));

        // A record that omits the parent says nothing about it, rather than claiming there is none.
        // A run's log holds many records of one execution and they need not each repeat it — which
        // is also what lets a run recorded before ends carried parentage still transfer.
        let bare = |span: &str| EventRow {
            seq: 1,
            at: "2026-08-15T00:00:00Z".into(),
            kind: "span_end".into(),
            node: None,
            payload_json: format!(r#"{{"kind":"span_end","span_id":"{span}"}}"#),
        };
        let mut quiet = bundle_of(vec![]);
        quiet.runs[0].run.run_id = "quiet".into();
        quiet.runs[0].events = vec![
            lifecycle("00000000000000dd", "00000000000000aa"),
            bare("00000000000000dd"),
        ];
        assert!(store.import(&quiet).await.is_ok());

        // Present and unreadable is not absent, though: it says something nobody can check, and
        // recording it as a root asserts a shape the bundle never claimed.
        let mut malformed = bundle_of(vec![]);
        malformed.runs[0].run.run_id = "malformed".into();
        malformed.runs[0].events = vec![EventRow {
            seq: 0,
            at: "2026-08-15T00:00:00Z".into(),
            kind: "span_start".into(),
            node: None,
            payload_json:
                r#"{"kind":"span_start","span_id":"00000000000000ee","parent_span_id":"nope"}"#
                    .into(),
        }];
        assert!(matches!(
            store.import(&malformed).await,
            Err(StoreError::BadExecutionGraph { .. })
        ));

        // Repeating what it already said is not disagreement: a start and its end both name the
        // execution and its parent, and a turn's records are many.
        let mut repeated = bundle_of(vec![]);
        repeated.runs[0].run.run_id = "twice".into();
        repeated.runs[0].events = vec![
            lifecycle("00000000000000cc", "00000000000000aa"),
            lifecycle("00000000000000cc", "00000000000000aa"),
        ];
        assert!(store.import(&repeated).await.is_ok());

        // And the shape a run actually produces goes in — including the ordinary case of a parent
        // this run has no checkpoint for, since host calls and nested turns announce themselves in
        // the event stream and write no row. A walk that ends there has ended, not failed.
        let host = SpanId::parse("00000000000000d4").unwrap();
        let parent = Invocation::root(SpanId::parse("00000000000000b2").unwrap());
        let ok = bundle_of(vec![
            imported_checkpoint("analyst", parent),
            imported_checkpoint(
                "implementer",
                parent.child(SpanId::parse("00000000000000c3").unwrap()),
            ),
            imported_checkpoint(
                "referee",
                Invocation {
                    span_id: SpanId::parse("00000000000000e5").unwrap(),
                    parent_span_id: Some(host),
                },
            ),
        ]);
        assert_eq!(store.import(&ok).await.unwrap()[0].checkpoints, 3);
    }

    #[tokio::test]
    async fn a_long_ancestry_costs_what_its_length_costs() {
        // The walk asks "have I been here on this walk" once per hop. Asked by scanning the steps
        // already taken, that is quadratic in the chain's length — and a bundle is the one
        // checkpoint path this process did not produce, so its shape is whatever an author chose.
        use ratatoskr_core::span::{Invocation, SpanId};
        let deep = 20_000u64;
        let id = |n: u64| SpanId::new(n.to_be_bytes()).expect("nonzero");
        let chain: Vec<Checkpoint> = (1..=deep)
            .map(|n| {
                imported_checkpoint(
                    "node",
                    Invocation {
                        span_id: id(n),
                        // Each names the one before it, so the first walk is the whole chain.
                        parent_span_id: (n > 1).then(|| id(n - 1)),
                    },
                )
            })
            .collect();

        let started = std::time::Instant::now();
        let store = Store::open_in_memory().unwrap();
        super::check_execution_graph("deep", &chain, &[]).expect("a chain is not a cycle");
        assert!(
            started.elapsed() < std::time::Duration::from_millis(250),
            "checking a chain of {deep} took {:?}",
            started.elapsed()
        );
        drop(store);
    }

    #[tokio::test]
    async fn a_bundle_round_trips_through_bson_into_another_store() {
        let bundle = seeded()
            .await
            .export(&["r1".to_string()], "kk@host", "2026-08-07T12:00:00Z")
            .await
            .unwrap();
        let bytes = to_bytes(&bundle).unwrap();
        let read = from_bytes(&bytes).unwrap();

        // Somewhere else entirely: a fresh store, as a different developer's would be.
        let theirs = Store::open_in_memory().unwrap();
        let report = theirs.import(&read).await.unwrap();
        assert_eq!(report.len(), 1);
        assert!(report[0].inserted);
        assert_eq!(report[0].checkpoints, 1);
        assert_eq!(report[0].events, 1);

        let run = theirs.run("r1").await.unwrap().expect("the run arrived");
        assert_eq!(run.status, "converged");
        assert_eq!(run.issue_id.as_deref(), Some("91"));
        // The whole point of importing: it is marked as someone else's.
        assert_eq!(run.origin.as_deref(), Some("kk@host"));

        // And it is analysable, not just countable.
        let events = theirs.events_for_run("r1").await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "node_start");
        let mut runs = theirs.list_runs().await.unwrap();
        theirs.attach_tags(&mut runs).await.unwrap();
        assert_eq!(runs[0].tags, ["arm-a"]);
    }

    #[tokio::test]
    async fn importing_a_run_that_is_already_here_changes_nothing() {
        let store = seeded().await;
        let bundle = store
            .export(&["r1".to_string()], "kk@host", "t")
            .await
            .unwrap();

        // Re-importing into the store it came from must not double its history or relabel a local
        // run as somebody else's.
        let report = store.import(&bundle).await.unwrap();
        assert!(!report[0].inserted);
        assert_eq!(store.events_for_run("r1").await.unwrap().len(), 1);
        assert_eq!(store.checkpoints_for_run("r1").await.unwrap().len(), 1);
        assert_eq!(store.run("r1").await.unwrap().unwrap().origin, None);
    }

    #[tokio::test]
    async fn a_newer_format_is_refused_rather_than_misread() {
        let mut bundle = seeded()
            .await
            .export(&["r1".to_string()], "kk@host", "t")
            .await
            .unwrap();
        bundle.version = FORMAT_VERSION + 1;
        let store = Store::open_in_memory().unwrap();
        assert!(matches!(
            store.import(&bundle).await,
            Err(StoreError::Unsupported { .. })
        ));

        // And an older one, for the same reason rather than a different one. A version is a shape,
        // and the shape includes what a record MEANS: a `span_end` carrying no parent now says the
        // run drove that execution, where an earlier exporter simply left the field off. A reader
        // that took one for the other would report a nested execution as top-level, or refuse a
        // bundle that was never wrong.
        bundle.version = FORMAT_VERSION - 1;
        assert!(matches!(
            store.import(&bundle).await,
            Err(StoreError::Unsupported { .. })
        ));
    }

    #[tokio::test]
    async fn an_exported_run_carries_its_image_digest() {
        // The pin travels with the bundle: a run analysed somewhere else keeps the digest of
        // the image it executed in, alongside the config and the graph hash.
        let store = seeded().await;
        store
            .record_run_provenance(
                "r1",
                Some("{}"),
                Some("deadbeef"),
                Some("abc123"),
                None,
                Some("sha256:abc"),
            )
            .await
            .unwrap();
        let bundle = store
            .export(&["r1".to_string()], "kk@host", "t")
            .await
            .unwrap();
        let bytes = to_bytes(&bundle).unwrap();
        let read = from_bytes(&bytes).unwrap();

        let theirs = Store::open_in_memory().unwrap();
        let report = theirs.import(&read).await.unwrap();
        assert!(report[0].inserted);

        let run = theirs.run("r1").await.unwrap().expect("the run arrived");
        assert_eq!(run.image_digest.as_deref(), Some("sha256:abc"));
        assert_eq!(run.graph_hash.as_deref(), Some("deadbeef"));
    }
}
