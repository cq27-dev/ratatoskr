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
/// invocations of a stage apart, so two rows claiming one id collapse two executions into one and
/// leave the reader no way to notice. An execution invoked by itself is a cycle, and a consumer
/// walking to a root would not terminate.
///
/// Checked at the boundary rather than trusted, because a bundle is the one checkpoint path whose
/// contents this process did not produce.
fn check_execution_graph(run_id: &str, checkpoints: &[Checkpoint]) -> Result<(), StoreError> {
    let mut seen = std::collections::HashSet::new();
    for checkpoint in checkpoints {
        let Some(invocation) = checkpoint.invocation else {
            continue;
        };
        if invocation.parent_span_id == Some(invocation.span_id) {
            return Err(StoreError::BadExecutionGraph {
                run_id: run_id.to_string(),
                problem: format!("execution {} is its own parent", invocation.span_id),
            });
        }
        if !seen.insert(invocation.span_id) {
            return Err(StoreError::BadExecutionGraph {
                run_id: run_id.to_string(),
                problem: format!(
                    "execution {} names more than one record",
                    invocation.span_id
                ),
            });
        }
    }
    Ok(())
}

/// The version this build writes and is willing to read.
///
/// One version is one shape: a bundle claiming this version carries every field of it, so adding a
/// field is a version bump rather than a defaulted key the reader has to guess at.
pub const FORMAT_VERSION: u32 = 2;

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
        if bundle.version > FORMAT_VERSION {
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
            check_execution_graph(id, &one.checkpoints)?;
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

        // Refused before anything is written: a run half-imported is worse than one refused.
        assert!(store.run("imported").await.unwrap().is_none());

        // And the shape a run actually produces goes in.
        let parent = Invocation::root(SpanId::parse("00000000000000b2").unwrap());
        let ok = bundle_of(vec![
            imported_checkpoint("analyst", parent),
            imported_checkpoint(
                "implementer",
                parent.child(SpanId::parse("00000000000000c3").unwrap()),
            ),
        ]);
        assert_eq!(store.import(&ok).await.unwrap()[0].checkpoints, 2);
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
