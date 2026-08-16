//! Which execution a record came from, and which execution invoked it.
//!
//! Every record a run produces identifies a node by *name*. A name is not an execution: a stage is
//! invoked repeatedly — once per converge pass — and a workflow may invoke one concurrently, so a
//! name cannot say which of two live invocations a record belongs to, or tell attempt N from
//! attempt N+1. Nor can it say what invoked it: a node's clarification answerer and, later, the
//! subagents a node spawns are executions with no place of their own in the pipeline shape, and a
//! reader with only names can do nothing with their records but invent a column.
//!
//! The identity is shaped as an OpenTelemetry span id — eight bytes, sixteen lowercase hex
//! characters — because the run's records already name their fields for the GenAI semantic
//! conventions, and an exporter then needs a re-encoding rather than a new mechanism. `run_id` is
//! the trace: one run, one trace.

use serde::{Deserialize, Serialize};

/// One execution's identity: eight bytes, never all-zero.
///
/// All-zero is OpenTelemetry's *invalid* span id, so it is refused rather than stored — a reader
/// must not have to tell an id that means "nothing here" from one that means "here". Absence is
/// `None`, and it is the only way to say it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpanId([u8; 8]);

impl SpanId {
    /// The identity of these bytes, or `None` for the invalid all-zero id.
    pub fn new(bytes: [u8; 8]) -> Option<Self> {
        (bytes != [0; 8]).then_some(Self(bytes))
    }

    /// Read one back from the sixteen hex characters it is written as.
    ///
    /// Strict: exactly sixteen lowercase-or-uppercase hex digits, and not the invalid id. Anything
    /// else is `None` — a value that arrived malformed is not an identity, and guessing at one
    /// would put a record under an execution that never happened.
    pub fn parse(hex: &str) -> Option<Self> {
        if hex.len() != 16 {
            return None;
        }
        let mut bytes = [0u8; 8];
        for (byte, pair) in bytes.iter_mut().zip(hex.as_bytes().chunks_exact(2)) {
            let digits = std::str::from_utf8(pair).ok()?;
            *byte = u8::from_str_radix(digits, 16).ok()?;
        }
        Self::new(bytes)
    }

    pub fn to_bytes(self) -> [u8; 8] {
        self.0
    }
}

impl std::fmt::Display for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for SpanId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SpanId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(deserializer)?;
        Self::parse(&hex).ok_or_else(|| serde::de::Error::custom(format!("not a span id: {hex}")))
    }
}

/// One execution: who it is, and what invoked it.
///
/// Minted by whatever *does* the invoking, and passed down — the workspace's rule for ids, and the
/// only way a nested execution can be given a parent it did not have to look up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invocation {
    pub span_id: SpanId,
    /// The execution that invoked this one; `None` for a stage the run itself drove.
    ///
    /// Absent is the ordinary case, not a gap in the record: a pipeline stage is invoked by the
    /// run, which is the trace rather than a span. Nothing may read absence as an error.
    pub parent_span_id: Option<SpanId>,
}

impl Invocation {
    /// An execution the run itself drove.
    pub fn root(span_id: SpanId) -> Self {
        Self {
            span_id,
            parent_span_id: None,
        }
    }

    /// An execution invoked from inside `self`, given its own identity.
    ///
    /// Takes the child's id rather than generating one, so a caller that must be able to say what
    /// it minted — a test, a replay — can, and nothing reaches for ambient randomness.
    pub fn child(&self, span_id: SpanId) -> Self {
        Self {
            span_id,
            parent_span_id: Some(self.span_id),
        }
    }
}

/// What a log record says it is: the node it belongs to, the execution that produced it, what
/// invoked that one, and where a control aimed at it goes.
///
/// One representation and one parser, read by everything that consumes a record — the event
/// normaliser the dashboard folds, and the import that validates a bundle. They had drifted:
/// resolved separately, a record's name came from one place and its identity from another, and a
/// bundle was validated against fields the reader does not even use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Attribution {
    /// What ran. `None` for a record that belongs to no node — a lifecycle record names an
    /// execution, and a host call is an execution the shape cannot place.
    pub node: Option<String>,
    pub invocation: Option<Invocation>,
    /// Where a Stop or a Steer for this record's turn is addressed, when that is not [`Self::node`].
    pub controlled_as: Option<String>,
    /// Whether the record stated its own identity, rather than inheriting it from the span it was
    /// emitted inside.
    ///
    /// The difference matters to a validator: a record's own absent parent may be an assertion that
    /// the run drove it, while an inherited one is only what the enclosing span happened to say.
    pub stated: bool,
}

/// Why a record's execution could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Malformed {
    /// A field is present and is not sixteen hex characters.
    NotAnId { key: &'static str, found: String },
}

impl std::fmt::Display for Malformed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAnId { key, found } => write!(f, "`{key}` is `{found}`, which is not an id"),
        }
    }
}

impl Attribution {
    /// Read one record.
    ///
    /// A record's own fields are one source and are taken together — stating an identity and no
    /// parent states that it has none, and inheriting a parent from the span around it would
    /// describe a parentage that never existed. Otherwise the innermost span naming an execution
    /// answers, and answers all of it: a turn runs inside the host call that drove it, and a record
    /// belongs to the nearest execution it was emitted inside.
    ///
    /// A lifecycle record is attributed to no node whatever it carries. It names an execution, and
    /// anything folding a record with a node into node state would draw a box for a host call.
    ///
    /// A present-but-unreadable id is an error rather than an absence. Absent says nothing; a value
    /// nobody can parse says something that cannot be checked, and reading it as absent asserts a
    /// shape the record never claimed.
    pub fn of(record: &serde_json::Value) -> Result<Self, Malformed> {
        let text = |key: &str| record.get(key).and_then(serde_json::Value::as_str);
        let kind = text("kind").unwrap_or("event");
        let lifecycle = matches!(kind, "span_start" | "span_end");
        let named = |node: Option<&str>| {
            (!lifecycle)
                .then_some(node)
                .flatten()
                .map(ToString::to_string)
        };
        // Missing and unreadable are different findings, so `as_str` cannot make the call: it turns
        // a number, an object or a boolean sitting where an id belongs into the same `None` a
        // missing key produces, and a malformed parentage was thereby demoted to a root. JSON's
        // `null` is the one non-string that reads as absent — it is how absence is spelled by a
        // producer that writes the key at all.
        let id = |source: &serde_json::Value, key: &'static str| match source.get(key) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(hex)) => {
                SpanId::parse(hex).map(Some).ok_or(Malformed::NotAnId {
                    key,
                    found: hex.to_string(),
                })
            }
            Some(other) => Err(Malformed::NotAnId {
                key,
                found: other.to_string(),
            }),
        };

        if let Some(span_id) = id(record, "span_id")? {
            return Ok(Self {
                node: named(text("node")),
                invocation: Some(Invocation {
                    span_id,
                    parent_span_id: id(record, "parent_span_id")?,
                }),
                controlled_as: text("controlled_as").map(ToString::to_string),
                stated: true,
            });
        }

        let spans = || {
            record
                .get("spans")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
        };
        let enclosing = spans().rev().find(|span| span.get("span_id").is_some());
        let of_span = |key: &str| {
            enclosing
                .and_then(|span| span.get(key))
                .and_then(serde_json::Value::as_str)
        };
        Ok(Self {
            // The record's own name still wins where it has one — a checkpoint names the node it
            // covers, which is not always the node whose span it was written inside.
            node: named(text("node").or_else(|| of_span("node")).or_else(|| {
                spans()
                    .rev()
                    .find_map(|s| s.get("node").and_then(serde_json::Value::as_str))
            })),
            invocation: match enclosing {
                None => None,
                // Both halves of the span, and both refused if either is unreadable. Swallowing the
                // parent's error promoted the execution to a root — the shape a reader would then
                // build, out of a value nobody could parse, and the opposite of what the same
                // malformed field means when the record carries it itself.
                Some(span) => match id(span, "span_id")? {
                    None => None,
                    Some(span_id) => Some(Invocation {
                        span_id,
                        parent_span_id: id(span, "parent_span_id")?,
                    }),
                },
            },
            // From the same span as the identity, so a turn's records keep the address its start
            // announced even where the start itself has been trimmed out of view.
            controlled_as: text("controlled_as")
                .or_else(|| of_span("controlled_as"))
                .map(ToString::to_string),
            stated: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_attributed_from_one_source_and_refuses_what_it_cannot_read() {
        let of = |value: serde_json::Value| Attribution::of(&value);

        // Its own fields, taken together: an identity and no parent states that it has none.
        let stated = of(serde_json::json!({
            "kind": "checkpoint",
            "node": "implementer",
            "span_id": "00000000000000a1",
            "spans": [{ "node": "analyst", "span_id": "00000000000000b2",
                        "parent_span_id": "00000000000000c3" }],
        }))
        .unwrap();
        assert_eq!(stated.node.as_deref(), Some("implementer"));
        assert_eq!(
            stated.invocation.and_then(|i| i.parent_span_id),
            None,
            "a record that names its own execution names its own parent, or has none"
        );
        assert!(stated.stated);

        // Otherwise the innermost span naming an execution answers all of it, control address
        // included — which is how a turn's records keep their address when its start is out of view.
        let inherited = of(serde_json::json!({
            "kind": "tool_call",
            "tool": "Read",
            "spans": [
                { "node": "implementer", "span_id": "00000000000000a1" },
                { "node": "analyst", "span_id": "00000000000000b2",
                  "parent_span_id": "00000000000000a1", "controlled_as": "implementer" },
            ],
        }))
        .unwrap();
        assert_eq!(inherited.node.as_deref(), Some("analyst"));
        assert_eq!(
            inherited.invocation.map(|i| i.span_id),
            SpanId::parse("00000000000000b2")
        );
        assert_eq!(inherited.controlled_as.as_deref(), Some("implementer"));
        assert!(!inherited.stated);

        // A lifecycle record is attributed to no node, whatever it carries.
        let lifecycle = of(serde_json::json!({
            "kind": "span_end",
            "node": "implementer",
            "span_id": "00000000000000c3",
        }))
        .unwrap();
        assert_eq!(lifecycle.node, None);

        // `null` is absence spelled out: a producer that writes the key at all writes it for every
        // record, and refusing it would refuse whole streams for saying "none" explicitly.
        let spelled = of(serde_json::json!({
            "kind": "usage", "span_id": "00000000000000a1", "parent_span_id": null,
        }))
        .unwrap();
        assert_eq!(spelled.invocation.and_then(|i| i.parent_span_id), None);

        // Present and unreadable is refused wherever it sits — on the record, or on the span it was
        // emitted inside. Reading it as absent would report a nested execution as one the run drove.
        for unreadable in [
            serde_json::json!({ "kind": "usage", "span_id": "nope" }),
            // A present value of the wrong TYPE is as unreadable as a wrong string: `as_str` made
            // a number sitting where an id belongs indistinguishable from a missing key.
            serde_json::json!({ "kind": "usage", "span_id": 41 }),
            serde_json::json!({ "kind": "usage", "span_id": "00000000000000a1",
                                "parent_span_id": true }),
            serde_json::json!({ "kind": "tool_call",
                                "spans": [{ "span_id": "00000000000000a1",
                                            "parent_span_id": {} }] }),
            serde_json::json!({ "kind": "usage", "span_id": "00000000000000a1",
                                "parent_span_id": "nope" }),
            serde_json::json!({ "kind": "tool_call", "spans": [{ "span_id": "nope" }] }),
            serde_json::json!({ "kind": "tool_call",
                                "spans": [{ "span_id": "00000000000000a1",
                                            "parent_span_id": "nope" }] }),
        ] {
            assert!(
                of(unreadable.clone()).is_err(),
                "{unreadable} names something that is not an execution"
            );
        }
    }

    #[test]
    fn an_id_is_sixteen_hex_characters_and_reads_back_as_itself() {
        let id = SpanId::new([0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]).unwrap();
        assert_eq!(id.to_string(), "0123456789abcdef");
        assert_eq!(SpanId::parse("0123456789abcdef"), Some(id));
        // Written lower, read either way: a hex digit's case is not part of the identity.
        assert_eq!(SpanId::parse("0123456789ABCDEF"), Some(id));
    }

    #[test]
    fn the_invalid_id_is_refused_wherever_it_arrives_from() {
        // All-zero is OpenTelemetry's "no span". Storing it would give a reader a value that means
        // nothing to distinguish from one that means something.
        assert_eq!(SpanId::new([0; 8]), None);
        assert_eq!(SpanId::parse("0000000000000000"), None);
    }

    #[test]
    fn a_malformed_id_is_not_guessed_at() {
        // Every one of these has arrived somewhere as "close enough" to an id: a truncated write, a
        // trace id in the wrong field, a hyphenated uuid, an empty column read as a string.
        for wrong in [
            "",
            "0123456789abcde",                  // one short
            "0123456789abcdef0",                // one long
            "0123456789abcdef0123456789abcdef", // a trace id
            "0123-456789abcdef",
            "0123456789abcdeg",
            " 123456789abcdef",
        ] {
            assert_eq!(SpanId::parse(wrong), None, "{wrong} is not a span id");
        }
    }

    #[test]
    fn a_child_carries_its_parent_and_a_root_carries_none() {
        let parent = Invocation::root(SpanId::parse("00000000000000a1").unwrap());
        assert_eq!(parent.parent_span_id, None);

        let child = parent.child(SpanId::parse("00000000000000b2").unwrap());
        assert_eq!(child.parent_span_id, Some(parent.span_id));
        assert_ne!(child.span_id, parent.span_id);

        // Two levels: a subagent of a nested execution names the one that spawned it, not the run.
        let grandchild = child.child(SpanId::parse("00000000000000c3").unwrap());
        assert_eq!(grandchild.parent_span_id, Some(child.span_id));
    }

    #[test]
    fn an_id_survives_the_json_a_record_travels_as() {
        let invocation = Invocation::root(SpanId::parse("00000000000000a1").unwrap())
            .child(SpanId::parse("fedcba9876543210").unwrap());
        let json = serde_json::to_string(&invocation).unwrap();
        assert!(json.contains("fedcba9876543210"), "{json}");
        assert_eq!(
            serde_json::from_str::<Invocation>(&json).unwrap(),
            invocation
        );

        // And a malformed one does not deserialize into something that reads as an execution.
        assert!(serde_json::from_str::<SpanId>("\"nope\"").is_err());
    }
}
