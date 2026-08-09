//! Authority ceilings shared by configuration and node execution.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A persisted authority level. Higher levels include every lower level.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    JsonSchema,
    strum::IntoStaticStr,
    strum::Display,
    strum::EnumString,
    strum::EnumIter,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Capability {
    Read,
    Write,
    Publish,
}

impl Capability {
    /// The greatest authority granted by a ceiling. An empty ceiling grants nothing.
    pub fn ceiling(capabilities: &[Self]) -> Option<Self> {
        capabilities.iter().copied().max()
    }

    /// Whether this ceiling includes `required` authority.
    pub fn permits(self, required: Self) -> bool {
        self >= required
    }
}
