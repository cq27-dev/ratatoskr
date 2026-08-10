//! Who may do what, on an instance somebody else can reach.
//!
//! Loopback needs none of this: whoever can reach the port is already the person who owns the
//! repository. A hosted instance is the other case — the dashboard can start runs, and a run
//! drives a tool-using model against a checkout and spends API credits, so "can reach the port"
//! stops being an acceptable answer to "may start a run".

use serde::{Deserialize, Serialize};

/// What a principal is allowed to do.
///
/// Three, because there are exactly three distinctions worth drawing: reading a run, causing one,
/// and deciding who may. A fourth would be a policy language, and the whole point of a closed enum
/// here is that every route names one of these and the compiler checks it did.
///
/// Ordered weakest-first, and compared by that order, so a guard is written `role >= Operator`
/// rather than by listing the variants that pass — a list that a new variant would silently fall
/// out of.
///
/// Persists (strum) and serialises (serde) as the same snake_case token, kept beside the variants,
/// exactly as [`crate::state::RunStatus`] does: this string is written to a database that outlives
/// any given build, so renaming a variant is a migration, not a refactor.
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
    strum::IntoStaticStr,
    strum::Display,
    strum::EnumString,
    strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Role {
    /// May read runs. The default for anyone who logs in and the ceiling for anonymous callers on
    /// a public project.
    Viewer,
    /// May also start runs and answer a node's clarification.
    ///
    /// This is the expensive one: starting a run drives a tool-using model against the repository,
    /// and answering a clarification puts text into the prompt of an agent that holds tools. Both
    /// are "may spend money and change a checkout", which is why they are one role and not folded
    /// into `Viewer`.
    Operator,
    /// May also manage principals: who exists, what role they hold, and who is disabled.
    Admin,
}

impl Role {
    /// The persisted string form (delegates to `strum::IntoStaticStr`).
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }
}

/// What a request is permitted to do with a project, as opposed to who is asking.
///
/// Separate from [`Role`] because the two answer different questions and are checked together: a
/// viewer may read a private project, an anonymous caller may read a public one, and neither may
/// act. Handlers name the access they need and let the check combine it with the role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Reading a run: its rows, its checkpoints, its event history.
    Read,
    /// Causing something: starting a run, answering a question.
    Act,
}

/// Whether a project may be read without a session.
///
/// Deliberately per project and deliberately *not* read from that project's own `ratatoskr.toml`.
/// A repository must not be able to declare itself public — that is the host operator's decision,
/// so it is configured where the instance's projects are listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Readable by anyone, including anonymous callers. Everything a run recorded is then public:
    /// the issue text, the model's output, and the contents of every file a tool read.
    Public,
    /// Readable only with a session. The default, because the safe direction for a mistake in a
    /// config file is "nobody can see it" rather than "everybody can".
    #[default]
    Private,
}

impl Visibility {
    /// Whether an anonymous caller may read this project at all.
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn roles_order_weakest_first_so_guards_can_compare() {
        // What every guard in the server relies on. If this ordering is ever rearranged, `>=`
        // checks silently start admitting the wrong people rather than failing to compile.
        assert!(Role::Viewer < Role::Operator);
        assert!(Role::Operator < Role::Admin);
        assert!(Role::Admin >= Role::Operator);
    }

    #[test]
    fn role_tokens_round_trip_through_their_persisted_form() {
        // These strings are in a database that outlives the build that wrote them.
        for role in [Role::Viewer, Role::Operator, Role::Admin] {
            assert_eq!(Role::from_str(role.as_str()).unwrap(), role);
        }
        assert_eq!(Role::Operator.as_str(), "operator");
    }

    #[test]
    fn an_unknown_role_is_an_error_rather_than_a_default() {
        // A row carrying a role this build does not know is a database from the future or a
        // corrupted one. Reading it as `Viewer` would be a quiet downgrade; reading it as anything
        // else would be a quiet escalation. It has to fail.
        assert!(Role::from_str("superuser").is_err());
        assert!(Role::from_str("").is_err());
    }

    #[test]
    fn visibility_defaults_to_private() {
        // The direction a mistake should fail in.
        assert_eq!(Visibility::default(), Visibility::Private);
        assert!(!Visibility::default().is_public());
    }
}
