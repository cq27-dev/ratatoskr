//! Exporting a run's spans to an OpenTelemetry collector.
//!
//! The span vocabulary is already the convention's: `otel.name`, `otel.kind`, `otel.status_code`,
//! `error.type` and the `gen_ai.*` attributes are recorded where the spans are opened, and this
//! instance's own counts sit under `ratatoskr.usage.*` precisely so nothing here has to rename a
//! field a reader depends on. Export is therefore a layer added, not a translation.
//!
//! Two things it does have to do.
//!
//! **One run is one trace.** A trace id is derived from `run_id` rather than minted, so every span
//! a run opens lands in the same trace — see [`ratatoskr_core::span::TraceId::of_run`]. The SDK
//! would otherwise mint a fresh trace id per root span and scatter a run across as many traces as
//! it has top-level executions.
//!
//! **Nothing happens unless an endpoint is configured.** The endpoint comes from the environment
//! — `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, falling back to `OTEL_EXPORTER_OTLP_ENDPOINT`, the
//! names every OTel SDK already reads — and never from `ratatoskr.toml`. Where a deployment sends
//! its traces is not a property of the repository being worked on: a checked-in endpoint would
//! make every clone of that repo export to whoever wrote it down, the same hazard that keeps
//! `--public` and the webhook secret out of the project's config. With neither variable set the
//! provider is never built, no socket is opened, and no name is resolved.
//!
//! Span ids stay the SDK's. The run's own `span_id`/`parent_span_id` ride as ordinary span fields,
//! so they arrive as attributes and a checkpoint row joins to its exported span by attribute
//! rather than by identity. Minting the SDK's span ids from ours would mean an `IdGenerator`
//! reading a task-local that only *some* spans set — every structural span (`iterate`,
//! `replan_at_ceiling`, `finish_run`) deliberately has no `span_id` — so the generator would be
//! guessing for exactly the spans that carry no identity to guess with.

use std::sync::OnceLock;

use opentelemetry_sdk::trace::IdGenerator;
use ratatoskr_core::span::TraceId;

/// The trace every span in this process belongs to.
///
/// A process runs one run: the CLI runs the one it was given, and `serve` spawns a fresh process
/// per run rather than driving several in-process. So the trace is process-wide, and a `OnceLock`
/// says that — a second run in one process would be a different design, and would find this taken.
static TRACE: OnceLock<TraceId> = OnceLock::new();

/// Bind this process's trace to a run, once its id is known.
///
/// Called where the run span is opened rather than at subscriber assembly, because the subscriber
/// is built before the command has parsed — `ratatoskr runs list` has no run id at all, and a run
/// resuming or being given `--run-id` learns it later still. Spans opened before this reach the
/// random fallback, which is correct: they belong to no run.
pub fn bind_run(run_id: &str) {
    let _ = TRACE.set(TraceId::of_run(run_id));
}

/// Trace ids from the run; span ids from the SDK.
#[derive(Debug, Default)]
struct RunTrace(opentelemetry_sdk::trace::RandomIdGenerator);

impl IdGenerator for RunTrace {
    fn new_trace_id(&self) -> opentelemetry::TraceId {
        match TRACE.get() {
            Some(trace) => opentelemetry::TraceId::from_bytes(trace.to_bytes()),
            // Before a run is bound, or in a process that drives none. A random trace is honest
            // here: these spans genuinely belong to no run, and folding them into a fixed id would
            // invent a trace that nothing else joins.
            None => self.0.new_trace_id(),
        }
    }

    fn new_span_id(&self) -> opentelemetry::SpanId {
        self.0.new_span_id()
    }
}

/// The conventions' event name for one of this instance's record kinds, where one exists.
///
/// `None` means the kind keeps its own name. That is not a gap: most of what a run records has no
/// counterpart in the GenAI conventions, and inventing one would claim a conformance the payload
/// does not have.
///
/// The internal `kind` values are a UI feed — read across the server and the dashboard, persisted
/// on every event row — and deliberately finer-grained than OTel's event model, which is about one
/// record per inference. Renaming them at the source would coarsen the live feed and rewrite
/// recorded history for a consumer that does not exist yet, so the translation happens at the
/// exporter, on the way out, and nothing inside the process sees it.
fn convention_event_name(kind: &str) -> Option<&'static str> {
    match kind {
        "model_text" => Some("gen_ai.choice"),
        "tool_call" | "tool_result" => Some("gen_ai.tool.message"),
        _ => None,
    }
}

/// An exporter that renames events to the conventions as they leave, and changes nothing else.
///
/// A wrapper rather than a layer because the name a `tracing` event exports under is its
/// *message*, fixed at the macro call site — there is no hook between the subscriber and the SDK
/// that could rewrite it. `SpanData` is the last place the record is still ours.
#[derive(Debug)]
struct Vocabulary<E>(E);

impl<E: opentelemetry_sdk::trace::SpanExporter> opentelemetry_sdk::trace::SpanExporter
    for Vocabulary<E>
{
    async fn export(
        &self,
        mut batch: Vec<opentelemetry_sdk::trace::SpanData>,
    ) -> opentelemetry_sdk::error::OTelSdkResult {
        for span in &mut batch {
            for event in &mut span.events.events {
                let kind = event
                    .attributes
                    .iter()
                    .find(|kv| kv.key.as_str() == "kind")
                    .map(|kv| kv.value.as_str().into_owned());
                if let Some(name) = kind.as_deref().and_then(convention_event_name) {
                    event.name = name.into();
                }
            }
        }
        self.0.export(batch).await
    }

    fn shutdown(&self) -> opentelemetry_sdk::error::OTelSdkResult {
        self.0.shutdown()
    }

    fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
        self.0.force_flush()
    }

    fn set_resource(&mut self, resource: &opentelemetry_sdk::Resource) {
        self.0.set_resource(resource);
    }
}

/// Where this deployment sends traces, if anywhere.
///
/// The two standard OTel variables, in the order the specification gives them: the signal-specific
/// one wins over the general one. An empty value is absence — a variable set to nothing is how a
/// shell says "unset" by accident, and opening a socket to "" helps nobody.
pub fn endpoint() -> Option<String> {
    let read = |key| std::env::var(key).ok();
    chosen(
        read("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").as_deref(),
        read("OTEL_EXPORTER_OTLP_ENDPOINT").as_deref(),
    )
}

/// Which of the two variables wins, as a decision over values rather than over the environment.
///
/// Separate from [`endpoint`] so it can be tested without `set_var`, which is unsound beside
/// concurrently-running tests reading the environment — the defect this repo already carries as
/// #314. Nothing here needs a real process variable to be worth checking.
fn chosen(traces: Option<&str>, general: Option<&str>) -> Option<String> {
    [traces, general]
        .into_iter()
        .flatten()
        .find(|value| !value.trim().is_empty())
        .map(str::to_string)
}

/// Build the tracer for `endpoint`.
///
/// Returns the provider alongside the tracer because a batch exporter has to be shut down for the
/// last spans to leave the process — dropping it is not enough.
pub fn tracer(
    endpoint: &str,
) -> anyhow::Result<(
    opentelemetry_sdk::trace::SdkTracerProvider,
    opentelemetry_sdk::trace::SdkTracer,
)> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig as _;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .build()?;
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(Vocabulary(exporter))
        .with_id_generator(RunTrace::default())
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name("ratatoskr")
                .build(),
        )
        .build();
    let tracer = provider.tracer("ratatoskr");
    Ok((provider, tracer))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The internal vocabulary is unchanged; only what leaves the process is renamed.
    #[test]
    fn only_the_kinds_with_a_convention_name_are_translated() {
        assert_eq!(convention_event_name("model_text"), Some("gen_ai.choice"));
        assert_eq!(
            convention_event_name("tool_call"),
            Some("gen_ai.tool.message")
        );
        assert_eq!(
            convention_event_name("tool_result"),
            Some("gen_ai.tool.message")
        );
        // This instance's own vocabulary keeps its own name rather than being forced into one.
        for own in [
            "node_start",
            "checkpoint",
            "usage",
            "acceptance_step",
            "span_start",
            "span_end",
            "turn_usage",
            "response_usage",
            "control",
            "question",
            "question_answered",
            "skill",
            "committed",
        ] {
            assert_eq!(
                convention_event_name(own),
                None,
                "`{own}` has no convention name"
            );
        }
    }

    /// The off switch, and the only one: no endpoint, no exporter, no socket.
    #[test]
    fn a_deployment_that_names_no_collector_exports_nothing() {
        assert_eq!(chosen(None, None), None);
        // A variable set to nothing is how a shell says "unset" by accident, not an endpoint.
        assert_eq!(chosen(Some(""), None), None);
        assert_eq!(chosen(Some("   "), Some("")), None);
    }

    /// The specification's precedence: the signal-specific variable wins.
    #[test]
    fn the_traces_endpoint_wins_over_the_general_one() {
        assert_eq!(
            chosen(
                Some("http://traces:4318/v1/traces"),
                Some("http://all:4318")
            ),
            Some("http://traces:4318/v1/traces".to_string())
        );
        assert_eq!(
            chosen(None, Some("http://all:4318")),
            Some("http://all:4318".to_string())
        );
        assert_eq!(
            chosen(Some(" "), Some("http://all:4318")),
            Some("http://all:4318".to_string())
        );
    }

    /// The generator is what makes a run one trace rather than one trace per root span.
    #[test]
    fn every_span_of_a_bound_run_shares_the_runs_trace() {
        bind_run("6402ccea-650f-4472-bff5-24e34466fe6d");
        let ids = RunTrace::default();
        let first = ids.new_trace_id();
        let second = ids.new_trace_id();
        assert_eq!(first, second, "a run's spans must share one trace");
        assert_eq!(
            first,
            opentelemetry::TraceId::from_bytes(
                TraceId::of_run("6402ccea-650f-4472-bff5-24e34466fe6d").to_bytes()
            ),
            "and it must be the trace the run id derives, so a reader can find it by run"
        );
        // Span ids stay the SDK's, and stay distinct.
        assert_ne!(ids.new_span_id(), ids.new_span_id());
    }
}
