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

/// The process's trace cell, handed to the generator so it can read it when it mints.
///
/// The subscriber is assembled before the command has parsed — `ratatoskr runs list` has no run id
/// at all, and a run given `--run-id` learns it later still — so the value cannot be read at build
/// time. Spans opened before a run is bound reach the random fallback, which is correct: they
/// belong to no run.
pub fn run_trace() -> &'static OnceLock<TraceId> {
    &TRACE
}

/// Bind this process's trace to a run, once its id is known.
pub fn bind_run(run_id: &str) {
    let _ = TRACE.set(TraceId::of_run(run_id));
}

/// Trace ids from the run; span ids from the SDK.
///
/// Holds the *cell*, not the value, and reads it when it mints. The provider is built inside
/// `init_logging`, which runs before `Cli::parse` — so at build time no command has been parsed,
/// no run id exists, and a trace read then is always absent. A root span opens later, after
/// `run_span` has bound the run, which is the only moment the answer exists.
///
/// The cell is a parameter rather than the process global so a test can hand over its own and get
/// a deterministic answer. Reading the global directly here is what made this generator's test
/// depend on whichever test in the binary reached `bind_run` first.
#[derive(Debug)]
struct RunTrace {
    trace: &'static OnceLock<TraceId>,
    ids: opentelemetry_sdk::trace::RandomIdGenerator,
}

impl RunTrace {
    fn new(trace: &'static OnceLock<TraceId>) -> Self {
        Self {
            trace,
            ids: opentelemetry_sdk::trace::RandomIdGenerator::default(),
        }
    }
}

impl IdGenerator for RunTrace {
    fn new_trace_id(&self) -> opentelemetry::TraceId {
        match self.trace.get() {
            Some(trace) => opentelemetry::TraceId::from_bytes(trace.to_bytes()),
            // A process driving no run, or a span opened before one was bound. A random trace is
            // honest: those spans belong to no run, and folding them into a fixed id would invent
            // a trace that nothing else joins.
            None => self.ids.new_trace_id(),
        }
    }

    fn new_span_id(&self) -> opentelemetry::SpanId {
        self.ids.new_span_id()
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
        // Both, because the conventions have one name for a tool exchange. `tool_call` is not
        // strictly a `gen_ai.tool.message` — that name is the result fed back to the model — but
        // the `kind` attribute is preserved on the event, so the two stay distinguishable to
        // anything that cares, and inventing a name outside the conventions would help nobody.
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
                let renamed = event
                    .attributes
                    .iter()
                    .find(|kv| kv.key.as_str() == "kind")
                    .and_then(|kv| convention_event_name(kv.value.as_str().as_ref()));
                if let Some(name) = renamed {
                    event.name = name.into();
                }
            }
        }
        self.0.export(batch).await
    }

    fn shutdown_with_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> opentelemetry_sdk::error::OTelSdkResult {
        // This half, not `shutdown` — the trait's `shutdown` delegates here, so overriding only
        // `shutdown` leaves a timed shutdown returning `Ok(())` without touching the inner
        // exporter, and the last batch would be dropped rather than sent.
        self.0.shutdown_with_timeout(timeout)
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
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(str::to_string)
}

/// Whether a configured endpoint is one the SDK will actually use.
///
/// The SDK parses the variable itself and, on anything that is not a `Uri`, falls through to the
/// built-in `http://localhost:4318` — so a typo like `htp://collector:4318` does not disable
/// export, it silently redirects a run's traces to localhost. Checked here so that is a refusal
/// with a message instead.
pub fn usable(endpoint: &str) -> bool {
    endpoint
        .parse::<http::Uri>()
        .is_ok_and(|uri| matches!(uri.scheme_str(), Some("http" | "https")))
}

/// Build the tracer.
///
/// The endpoint is NOT passed in. The SDK reads the same two variables itself, and treats them
/// differently on purpose: `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is the full path, while
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is a base URL it appends `/v1/traces` to. Handing it a
/// pre-resolved string defeats that — the standard base URL every collector documents would then
/// POST to `/` and 404. [`endpoint`] therefore decides only *whether* to export, never *where*.
///
/// Returns the provider alongside the tracer because a batch exporter has to be shut down for the
/// last spans to leave the process — dropping it is not enough.
pub fn tracer(
    trace: &'static OnceLock<TraceId>,
    endpoint: Option<&str>,
) -> anyhow::Result<(
    opentelemetry_sdk::trace::SdkTracerProvider,
    opentelemetry_sdk::trace::SdkTracer,
)> {
    use opentelemetry::trace::TracerProvider as _;

    // `endpoint` is `None` in production, and must stay so: the SDK reads the two variables
    // itself and appends `/v1/traces` only to the general one. It exists for the round-trip test,
    // which must point at its own listener without mutating the environment (see #314).
    let http = opentelemetry_otlp::SpanExporter::builder().with_http();
    let exporter = match endpoint {
        Some(url) => {
            use opentelemetry_otlp::WithExportConfig as _;
            http.with_endpoint(url).build()?
        }
        None => http.build()?,
    };
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(Vocabulary(exporter))
        .with_id_generator(RunTrace::new(trace))
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
    ///
    /// Built with its trace rather than reading the process-global, so this asserts the decision
    /// and not whichever test in this binary happened to call `bind_run` first.
    #[test]
    fn every_span_of_a_bound_run_shares_the_runs_trace() {
        let run = "6402ccea-650f-4472-bff5-24e34466fe6d";
        // Its OWN cell, bound as `run_span` binds the process one — so this drives the production
        // wiring rather than a copy of it. Leaked because the generator holds it for 'static; one
        // cell per test run is nothing.
        let cell: &'static OnceLock<TraceId> = Box::leak(Box::new(OnceLock::new()));
        cell.set(TraceId::of_run(run)).expect("a fresh cell");
        let ids = RunTrace::new(cell);
        let first = ids.new_trace_id();
        assert_eq!(first, ids.new_trace_id(), "a run's spans share one trace");
        assert_eq!(
            first,
            opentelemetry::TraceId::from_bytes(TraceId::of_run(run).to_bytes()),
            "and it is the trace the run id derives, so a reader can find it by run"
        );
        // Span ids stay the SDK's, and stay distinct.
        assert_ne!(ids.new_span_id(), ids.new_span_id());
    }

    /// The whole path, end to end: a span opened through the layer arrives at a listener as OTLP.
    ///
    /// This is the test that was missing, and both defects it now pins were shipped without it.
    /// The batch processor exports from its own thread with no tokio context, so pairing it with
    /// an async HTTP client panics there and nothing ever leaves — invisible to every unit test.
    /// And a pre-resolved endpoint handed to `with_endpoint` bypasses the SDK's `/v1/traces`
    /// resolution, so the standard base URL POSTs to `/`.
    ///
    /// A bare `TcpListener` rather than a real collector: the assertion is what went over the
    /// wire, which needs no backend to observe.
    #[test]
    fn a_span_reaches_a_collector_as_otlp_over_http() {
        use prost::Message as _;
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a local port");
        let port = listener.local_addr().expect("its address").port();
        let (tx, rx) = std::sync::mpsc::channel();
        let collector = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("one connection");
            let mut raw = Vec::new();
            let mut buf = [0u8; 8192];
            // Read until the body is complete, which the head's Content-Length tells us.
            loop {
                let read = socket.read(&mut buf).expect("reading the request");
                if read == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..read]);
                let Some(head) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let text = String::from_utf8_lossy(&raw[..head]).to_ascii_lowercase();
                let len: usize = text
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                if raw.len() >= head + 4 + len {
                    let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                    let _ = socket.flush();
                    tx.send((text, raw[head + 4..].to_vec()))
                        .expect("handing it back");
                    break;
                }
            }
        });

        let run = "6402ccea-650f-4472-bff5-24e34466fe6d";
        // Its OWN cell, bound as `run_span` binds the process one — so this drives the production
        // wiring rather than a copy of it. Leaked because the generator holds it for 'static; one
        // cell per test run is nothing.
        let cell: &'static OnceLock<TraceId> = Box::leak(Box::new(OnceLock::new()));
        cell.set(TraceId::of_run(run)).expect("a fresh cell");
        let (provider, tracer) = tracer(cell, Some(&format!("http://127.0.0.1:{port}/v1/traces")))
            .expect("the tracer builds");

        {
            use tracing_subscriber::layer::SubscriberExt as _;
            let layer = tracing_opentelemetry::layer().with_tracer(tracer);
            let subscriber = tracing_subscriber::registry().with(layer);
            tracing::subscriber::with_default(subscriber, || {
                let parent = tracing::info_span!(
                    "invoke_agent analyst",
                    "gen_ai.provider.name" = "anthropic"
                );
                let _entered = parent.enter();
                tracing::info!(kind = "model_text", "model text");
                let child = tracing::info_span!("invoke_agent implementer");
                let _nested = child.enter();
            });
        }
        // Shutdown runs on its own thread and is never joined on the failure path. It is what
        // flushes the batch, but with nothing to flush it BLOCKS — measured, not assumed — so
        // calling it inline turns "exported nothing", the exact regression this test exists for,
        // into a hung CI job with no failing assertion. The receive below is the deadline for the
        // whole exchange; when it expires the test fails red and the parked threads die with the
        // process. `collector` is joined only on the path where a request actually arrived.
        let flushed = std::thread::spawn(move || provider.shutdown());
        let (head, body) = rx
            .recv_timeout(std::time::Duration::from_secs(15))
            .expect("a request reached the collector within 15s");
        flushed
            .join()
            .expect("the shutdown thread")
            .expect("the batch exporter flushes on shutdown");
        collector.join().expect("the listener thread");
        assert!(head.starts_with("post /v1/traces "), "got: {head}");
        assert!(
            head.contains("content-type: application/x-protobuf"),
            "got: {head}"
        );

        let request =
            opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest::decode(
                &body[..],
            )
            .expect("the body is an OTLP trace export");
        let spans: Vec<_> = request
            .resource_spans
            .iter()
            .flat_map(|r| &r.scope_spans)
            .flat_map(|s| &s.spans)
            .collect();
        assert!(!spans.is_empty(), "at least one span was exported");

        // One run is one trace.
        let expected = TraceId::of_run(run).to_bytes();
        for span in &spans {
            assert_eq!(
                span.trace_id, expected,
                "every span carries the run's trace"
            );
        }
        // Nesting survives: the child names the parent's span id.
        let parent = spans
            .iter()
            .find(|s| s.name == "invoke_agent analyst")
            .expect("the parent span");
        let child = spans
            .iter()
            .find(|s| s.name == "invoke_agent implementer")
            .expect("the child span");
        assert_eq!(
            child.parent_span_id, parent.span_id,
            "the child must name its parent, not a fresh root"
        );
        // A `gen_ai.*` attribute survives the trip — the conventions are the point of exporting.
        assert!(
            parent
                .attributes
                .iter()
                .any(|kv| kv.key == "gen_ai.provider.name"),
            "gen_ai attributes must reach the wire, got {:?}",
            parent
                .attributes
                .iter()
                .map(|kv| &kv.key)
                .collect::<Vec<_>>()
        );
        // And the vocabulary was translated on the way out, not at the emit site.
        assert!(
            parent.events.iter().any(|e| e.name == "gen_ai.choice"),
            "the `model_text` event exports under its convention name, got {:?}",
            parent.events.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
    }

    /// A process driving no run invents no trace to put its spans in.
    #[test]
    fn an_unbound_process_gets_a_random_trace_per_root() {
        let unbound: &'static OnceLock<TraceId> = Box::leak(Box::new(OnceLock::new()));
        let ids = RunTrace::new(unbound);
        assert_ne!(ids.new_trace_id(), ids.new_trace_id());
    }
}
