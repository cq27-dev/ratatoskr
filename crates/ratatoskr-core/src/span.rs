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

#[cfg(test)]
mod tests {
    use super::*;

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
