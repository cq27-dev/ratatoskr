//! The instance's own database: who may use it, and who is currently logged in.
//!
//! Separate from [`crate::Store`] on purpose — see `auth_schema.sql` for why. The two share this
//! crate because they share SQLite and its connection handling, not because they are the same
//! store; nothing here touches a project's checkpoints and nothing there touches a principal.
//!
//! Everything is `spawn_blocking`, like the checkpoint store, and for a sharper reason here:
//! argon2 is *designed* to take a measurable slice of CPU, so a verify on the async runtime would
//! stall every other request on that worker for the duration of a login.

use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use argon2::Argon2;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use rand_core::{OsRng, RngCore};
use ratatoskr_core::auth::Role;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::StoreError;

/// How long a session lasts before it has to be made again.
///
/// Long, because the cost of a short one is not security but a login prompt in the middle of
/// watching a run; the token is revocable and stored hashed, which is what actually bounds the
/// damage of one leaking.
const SESSION_DAYS: u32 = 30;

/// Bytes of entropy in a session token. 256 bits, so the token is not the weak part of anything.
const TOKEN_BYTES: usize = 32;

/// The `provider` value for a username and password held by this instance.
pub const LOCAL: &str = "local";

/// Someone who may act, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub principal_id: String,
    pub display_name: String,
    pub role: Role,
}

/// Why a login or a lookup did not produce a principal.
///
/// Deliberately coarse at the boundary: [`AuthStore::authenticate`] returns `None` for a missing
/// user, a wrong password and a disabled account alike, because telling them apart tells an
/// attacker which usernames exist.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("password hashing failed: {0}")]
    Hash(String),
    #[error("a principal already holds the identity {provider}:{subject}")]
    IdentityTaken { provider: String, subject: String },
    #[error("row carries an unknown role `{0}` — this database was written by a newer build")]
    UnknownRole(String),
}

/// So `?` works on rusqlite calls inside the blocking closures, without a second variant that
/// would mean the same thing as `Store` and have to be matched separately by every caller.
impl From<rusqlite::Error> for AuthError {
    fn from(e: rusqlite::Error) -> Self {
        AuthError::Store(StoreError::Sqlite(e))
    }
}

/// A handle to the instance's identity database. Cheap to clone.
#[derive(Clone)]
pub struct AuthStore {
    conn: Arc<Mutex<Connection>>,
}

impl AuthStore {
    /// Open (creating if needed) the identity database at `path`, in WAL mode with the schema
    /// applied.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::from_connection(crate::open_sqlite(path.as_ref())?)
    }

    /// An in-memory identity database, for tests.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self, StoreError> {
        // ON DELETE CASCADE on `identities` and `sessions` only does anything with this on. The
        // bundled SQLite this crate links already defaults it on; set anyway, because the failure
        // mode if that ever changes is silently orphaned rows rather than an error.
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(include_str!("auth_schema.sql"))?;
        Ok(AuthStore {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Whether anyone at all exists yet.
    ///
    /// What tells a first run from a locked-out one: an instance with no principals needs someone
    /// bootstrapped, and an instance with principals must never hand out another admin for free.
    pub async fn is_empty(&self) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            let count: i64 = conn.query_row("SELECT count(*) FROM principals", [], |r| r.get(0))?;
            Ok::<_, StoreError>(count == 0)
        })
        .await?
    }

    /// Create a principal with a local username and password.
    ///
    /// One call rather than create-then-attach, because a principal with no way to log in is not a
    /// state worth being able to reach halfway through.
    pub async fn create_local(
        &self,
        username: &str,
        password: &str,
        display_name: &str,
        role: Role,
    ) -> Result<Principal, AuthError> {
        let principal = Principal {
            principal_id: uuid::Uuid::new_v4().to_string(),
            display_name: display_name.to_string(),
            role,
        };
        let secret = hash_password(password)?;
        let conn = Arc::clone(&self.conn);
        let (username, stored) = (username.to_string(), principal.clone());
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("auth store mutex poisoned");
            let tx = conn.transaction()?;
            tx.execute(
                "INSERT INTO principals (principal_id, display_name, role) VALUES (?1, ?2, ?3)",
                params![
                    stored.principal_id,
                    stored.display_name,
                    stored.role.as_str()
                ],
            )?;
            let taken = tx
                .execute(
                    "INSERT OR IGNORE INTO identities (provider, subject, principal_id, secret)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![LOCAL, username, stored.principal_id, secret],
                )
                .map(|rows| rows == 0)?;
            if taken {
                // Rolls back the principal too: `INSERT OR IGNORE` would otherwise leave one with
                // no way to log in, and the caller would have been told the username was taken
                // while a half-made account accumulated.
                return Err(AuthError::IdentityTaken {
                    provider: LOCAL.to_string(),
                    subject: username,
                });
            }
            tx.commit()?;
            Ok(stored)
        })
        .await
        .map_err(|e| AuthError::Store(StoreError::Join(e)))?
    }

    /// The principal behind a username and password, or `None`.
    ///
    /// `None` covers "no such user", "wrong password" and "disabled" without distinction. The
    /// verify runs even when the user does not exist, against a throwaway hash, so the response
    /// time does not say which usernames are real.
    pub async fn authenticate(
        &self,
        username: &str,
        password: &str,
    ) -> Result<Option<Principal>, AuthError> {
        let conn = Arc::clone(&self.conn);
        let (username, password) = (username.to_string(), password.to_string());
        tokio::task::spawn_blocking(move || {
            let found = {
                let conn = conn.lock().expect("auth store mutex poisoned");
                conn.query_row(
                    "SELECT p.principal_id, p.display_name, p.role, i.secret
                     FROM identities i JOIN principals p USING (principal_id)
                     WHERE i.provider = ?1 AND i.subject = ?2 AND p.disabled_at IS NULL",
                    params![LOCAL, username],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, Option<String>>(3)?,
                        ))
                    },
                )
                .optional()
                .map_err(StoreError::from)?
            };

            let Some((principal_id, display_name, role, Some(secret))) = found else {
                // Spend the same work on a username that does not exist. Without this, "no such
                // user" returns in microseconds and a real one takes as long as argon2 does, which
                // enumerates the account list from timing alone.
                let _ = verify_password(password.as_bytes(), &decoy_hash());
                return Ok(None);
            };
            if !verify_password(password.as_bytes(), &secret) {
                return Ok(None);
            }
            let role = Role::from_str(&role).map_err(|_| AuthError::UnknownRole(role))?;
            Ok(Some(Principal {
                principal_id,
                display_name,
                role,
            }))
        })
        .await
        .map_err(|e| AuthError::Store(StoreError::Join(e)))?
    }

    /// Start a session, returning the token to hand the browser. Only its hash is stored.
    pub async fn create_session(
        &self,
        principal_id: &str,
        user_agent: Option<&str>,
    ) -> Result<String, StoreError> {
        let token = random_token();
        let digest = digest(&token);
        let conn = Arc::clone(&self.conn);
        let (principal_id, user_agent) = (
            principal_id.to_string(),
            user_agent.map(|s| s.chars().take(200).collect::<String>()),
        );
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            conn.execute(
                "INSERT INTO sessions (token_hash, principal_id, expires_at, user_agent)
                 VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?3), ?4)",
                params![
                    digest,
                    principal_id,
                    format!("+{SESSION_DAYS} days"),
                    user_agent
                ],
            )?;
            Ok::<_, StoreError>(())
        })
        .await??;
        Ok(token)
    }

    /// The principal a session token belongs to, if it is live.
    ///
    /// Expiry and the principal's disabled flag are both checked in SQL, so a revoked account
    /// stops working on its next request rather than when its session happens to lapse.
    pub async fn principal_for_session(&self, token: &str) -> Result<Option<Principal>, AuthError> {
        let digest = digest(token);
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            let found = conn
                .query_row(
                    "SELECT p.principal_id, p.display_name, p.role
                     FROM sessions s JOIN principals p USING (principal_id)
                     WHERE s.token_hash = ?1
                       AND s.expires_at > strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       AND p.disabled_at IS NULL",
                    params![digest],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()
                .map_err(StoreError::from)?;
            let Some((principal_id, display_name, role)) = found else {
                return Ok(None);
            };
            let role = Role::from_str(&role).map_err(|_| AuthError::UnknownRole(role))?;
            Ok(Some(Principal {
                principal_id,
                display_name,
                role,
            }))
        })
        .await
        .map_err(|e| AuthError::Store(StoreError::Join(e)))?
    }

    /// End one session. Idempotent: logging out twice is not an error.
    pub async fn revoke_session(&self, token: &str) -> Result<(), StoreError> {
        let digest = digest(token);
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?1",
                params![digest],
            )?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Drop every session that has lapsed. Nothing depends on this for correctness — expiry is
    /// enforced on read — so it is housekeeping, not a security boundary.
    pub async fn purge_expired_sessions(&self) -> Result<usize, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            let gone = conn.execute(
                "DELETE FROM sessions WHERE expires_at <= strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                [],
            )?;
            Ok::<_, StoreError>(gone)
        })
        .await?
    }

    /// The principal behind a local username, for commands that address an account by name.
    ///
    /// Unlike [`AuthStore::authenticate`] this proves nothing and is not a login — it is the
    /// lookup an administrator needs to name someone, and it deliberately still reports a disabled
    /// account, because re-enabling one is a thing an administrator does.
    pub async fn principal_for_local(
        &self,
        username: &str,
    ) -> Result<Option<Principal>, AuthError> {
        let conn = Arc::clone(&self.conn);
        let username = username.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            let found = conn
                .query_row(
                    "SELECT p.principal_id, p.display_name, p.role
                     FROM identities i JOIN principals p USING (principal_id)
                     WHERE i.provider = ?1 AND i.subject = ?2",
                    params![LOCAL, username],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()?;
            let Some((principal_id, display_name, role)) = found else {
                return Ok(None);
            };
            let role = Role::from_str(&role).map_err(|_| AuthError::UnknownRole(role))?;
            Ok(Some(Principal {
                principal_id,
                display_name,
                role,
            }))
        })
        .await
        .map_err(|e| AuthError::Store(StoreError::Join(e)))?
    }

    /// Every principal, for `ratatoskr users list`.
    pub async fn list_principals(&self) -> Result<Vec<(Principal, bool)>, AuthError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            let mut stmt = conn
                .prepare(
                    "SELECT principal_id, display_name, role, disabled_at IS NOT NULL
                     FROM principals ORDER BY display_name",
                )
                .map_err(StoreError::from)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, bool>(3)?,
                    ))
                })
                .map_err(StoreError::from)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(StoreError::from)?;
            rows.into_iter()
                .map(|(principal_id, display_name, role, disabled)| {
                    let role = Role::from_str(&role).map_err(|_| AuthError::UnknownRole(role))?;
                    Ok((
                        Principal {
                            principal_id,
                            display_name,
                            role,
                        },
                        disabled,
                    ))
                })
                .collect()
        })
        .await
        .map_err(|e| AuthError::Store(StoreError::Join(e)))?
    }

    /// Change what a principal may do. Every live session picks it up on its next request, because
    /// the role is read from the row rather than baked into the token.
    pub async fn set_role(&self, principal_id: &str, role: Role) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        let principal_id = principal_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("auth store mutex poisoned");
            let changed = conn.execute(
                "UPDATE principals SET role = ?2 WHERE principal_id = ?1",
                params![principal_id, role.as_str()],
            )?;
            Ok::<_, StoreError>(changed > 0)
        })
        .await?
    }

    /// Disable or re-enable a principal, dropping their sessions when disabling.
    ///
    /// Both halves matter: the flag stops new logins and the delete stops the browser that is
    /// already logged in, which is the one you are usually trying to stop.
    pub async fn set_disabled(
        &self,
        principal_id: &str,
        disabled: bool,
    ) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        let principal_id = principal_id.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("auth store mutex poisoned");
            let tx = conn.transaction()?;
            let changed = tx.execute(
                "UPDATE principals
                 SET disabled_at = CASE WHEN ?2 THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now') END
                 WHERE principal_id = ?1",
                params![principal_id, disabled],
            )?;
            if disabled {
                tx.execute(
                    "DELETE FROM sessions WHERE principal_id = ?1",
                    params![principal_id],
                )?;
            }
            tx.commit()?;
            Ok::<_, StoreError>(changed > 0)
        })
        .await?
    }

    /// Replace a principal's local password, ending their other sessions.
    pub async fn set_password(&self, username: &str, password: &str) -> Result<bool, AuthError> {
        let secret = hash_password(password)?;
        let conn = Arc::clone(&self.conn);
        let username = username.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("auth store mutex poisoned");
            let tx = conn.transaction()?;
            let changed = tx.execute(
                "UPDATE identities SET secret = ?3 WHERE provider = ?1 AND subject = ?2",
                params![LOCAL, username, secret],
            )?;
            // A password change is usually a response to it having leaked, so the sessions it
            // opened have to go with it.
            tx.execute(
                "DELETE FROM sessions WHERE principal_id IN
                   (SELECT principal_id FROM identities WHERE provider = ?1 AND subject = ?2)",
                params![LOCAL, username],
            )?;
            tx.commit()?;
            Ok::<_, AuthError>(changed > 0)
        })
        .await
        .map_err(|e| AuthError::Store(StoreError::Join(e)))?
    }
}

/// A 256-bit token, hex encoded.
fn random_token() -> String {
    let mut bytes = [0u8; TOKEN_BYTES];
    OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// What is stored for a token. Not a password hash: see `auth_schema.sql`.
fn digest(token: &str) -> String {
    let out = Sha256::digest(token.as_bytes());
    out.iter().map(|b| format!("{b:02x}")).collect()
}

fn hash_password(password: &str) -> Result<String, AuthError> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| AuthError::Hash(e.to_string()))
}

fn verify_password(password: &[u8], phc: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(phc) else {
        return false;
    };
    Argon2::default().verify_password(password, &parsed).is_ok()
}

/// A real hash to verify against when the username does not exist, so the work — and therefore the
/// response time — matches a real login. Computed per call rather than held in a `static` so the
/// cost tracks whatever parameters `Argon2::default()` currently uses.
fn decoy_hash() -> String {
    hash_password("a password nobody has").unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store() -> AuthStore {
        AuthStore::open_in_memory().unwrap()
    }

    #[tokio::test]
    async fn a_new_instance_is_empty_and_stops_being_so() {
        let auth = store().await;
        assert!(auth.is_empty().await.unwrap());
        auth.create_local("kk", "hunter2", "KK", Role::Admin)
            .await
            .unwrap();
        assert!(!auth.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn the_right_password_authenticates_and_a_wrong_one_does_not() {
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Operator)
            .await
            .unwrap();
        let got = auth.authenticate("kk", "hunter2").await.unwrap().unwrap();
        assert_eq!(got, made);
        assert_eq!(got.role, Role::Operator);
        assert!(auth.authenticate("kk", "hunter3").await.unwrap().is_none());
        assert!(
            auth.authenticate("nobody", "hunter2")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn the_password_is_not_stored() {
        // The whole point of argon2 here. A store that kept the password — or a reversible form of
        // it — would turn one leaked file into everyone's password everywhere else.
        let auth = store().await;
        auth.create_local("kk", "hunter2", "KK", Role::Admin)
            .await
            .unwrap();
        let secret: String = {
            let conn = auth.conn.lock().unwrap();
            conn.query_row("SELECT secret FROM identities", [], |r| r.get(0))
                .unwrap()
        };
        assert!(!secret.contains("hunter2"));
        assert!(secret.starts_with("$argon2"));
    }

    #[tokio::test]
    async fn a_session_round_trips_and_its_token_is_not_stored() {
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Viewer)
            .await
            .unwrap();
        let token = auth
            .create_session(&made.principal_id, Some("a browser"))
            .await
            .unwrap();
        assert_eq!(
            auth.principal_for_session(&token).await.unwrap().unwrap(),
            made
        );

        let stored: String = {
            let conn = auth.conn.lock().unwrap();
            conn.query_row("SELECT token_hash FROM sessions", [], |r| r.get(0))
                .unwrap()
        };
        // A leaked database must not yield a usable cookie.
        assert_ne!(stored, token);
        assert!(auth.principal_for_session(&stored).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_unknown_token_is_nobody() {
        let auth = store().await;
        assert!(
            auth.principal_for_session("not a token")
                .await
                .unwrap()
                .is_none()
        );
        assert!(auth.principal_for_session("").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn revoking_ends_the_session_and_repeats_are_fine() {
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Viewer)
            .await
            .unwrap();
        let token = auth.create_session(&made.principal_id, None).await.unwrap();
        auth.revoke_session(&token).await.unwrap();
        assert!(auth.principal_for_session(&token).await.unwrap().is_none());
        auth.revoke_session(&token).await.unwrap();
    }

    #[tokio::test]
    async fn an_expired_session_is_refused_and_purged() {
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Viewer)
            .await
            .unwrap();
        let token = auth.create_session(&made.principal_id, None).await.unwrap();
        {
            let conn = auth.conn.lock().unwrap();
            conn.execute(
                "UPDATE sessions SET expires_at = '2000-01-01T00:00:00.000Z'",
                [],
            )
            .unwrap();
        }
        // Refused on read, which is what makes expiry a rule rather than a cleanup schedule.
        assert!(auth.principal_for_session(&token).await.unwrap().is_none());
        assert_eq!(auth.purge_expired_sessions().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn disabling_stops_both_new_logins_and_the_browser_already_open() {
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Operator)
            .await
            .unwrap();
        let token = auth.create_session(&made.principal_id, None).await.unwrap();

        assert!(auth.set_disabled(&made.principal_id, true).await.unwrap());
        assert!(auth.authenticate("kk", "hunter2").await.unwrap().is_none());
        // The half that is easy to forget: the flag alone would leave an open tab working.
        assert!(auth.principal_for_session(&token).await.unwrap().is_none());

        assert!(auth.set_disabled(&made.principal_id, false).await.unwrap());
        assert!(auth.authenticate("kk", "hunter2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_role_change_reaches_a_session_that_already_exists() {
        // The role is read from the row on every request rather than baked into the token, so a
        // demotion takes effect without waiting for the session to lapse.
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Admin)
            .await
            .unwrap();
        let token = auth.create_session(&made.principal_id, None).await.unwrap();
        auth.set_role(&made.principal_id, Role::Viewer)
            .await
            .unwrap();
        assert_eq!(
            auth.principal_for_session(&token)
                .await
                .unwrap()
                .unwrap()
                .role,
            Role::Viewer
        );
    }

    #[tokio::test]
    async fn changing_a_password_ends_the_sessions_it_opened() {
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Viewer)
            .await
            .unwrap();
        let token = auth.create_session(&made.principal_id, None).await.unwrap();
        assert!(auth.set_password("kk", "hunter3").await.unwrap());

        assert!(auth.principal_for_session(&token).await.unwrap().is_none());
        assert!(auth.authenticate("kk", "hunter2").await.unwrap().is_none());
        assert!(auth.authenticate("kk", "hunter3").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_duplicate_username_is_refused_without_leaving_a_half_made_principal() {
        let auth = store().await;
        auth.create_local("kk", "hunter2", "KK", Role::Admin)
            .await
            .unwrap();
        let again = auth
            .create_local("kk", "other", "Someone", Role::Viewer)
            .await;
        assert!(matches!(again, Err(AuthError::IdentityTaken { .. })));

        // The rollback: a principal with no identity could never log in and could never be
        // cleaned up by username either.
        let principals = auth.list_principals().await.unwrap();
        assert_eq!(principals.len(), 1);
        assert!(auth.authenticate("kk", "hunter2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_username_resolves_to_its_principal_even_when_disabled() {
        // Administration has to be able to name a disabled account — re-enabling one is the whole
        // point — which is why this is not `authenticate` with the password left out.
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Operator)
            .await
            .unwrap();
        assert_eq!(auth.principal_for_local("kk").await.unwrap().unwrap(), made);
        auth.set_disabled(&made.principal_id, true).await.unwrap();
        assert!(auth.principal_for_local("kk").await.unwrap().is_some());
        assert!(auth.principal_for_local("nobody").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn listing_reports_roles_and_who_is_disabled() {
        let auth = store().await;
        let a = auth
            .create_local("ann", "pw", "Ann", Role::Admin)
            .await
            .unwrap();
        auth.create_local("bob", "pw", "Bob", Role::Viewer)
            .await
            .unwrap();
        auth.set_disabled(&a.principal_id, true).await.unwrap();

        let listed = auth.list_principals().await.unwrap();
        assert_eq!(listed.len(), 2);
        let ann = listed
            .iter()
            .find(|(p, _)| p.display_name == "Ann")
            .unwrap();
        assert_eq!(ann.0.role, Role::Admin);
        assert!(ann.1);
        let bob = listed
            .iter()
            .find(|(p, _)| p.display_name == "Bob")
            .unwrap();
        assert!(!bob.1);
    }

    #[tokio::test]
    async fn two_sessions_for_one_principal_are_independent() {
        // A phone and a laptop: logging out of one must not log out the other.
        let auth = store().await;
        let made = auth
            .create_local("kk", "hunter2", "KK", Role::Viewer)
            .await
            .unwrap();
        let phone = auth.create_session(&made.principal_id, None).await.unwrap();
        let laptop = auth.create_session(&made.principal_id, None).await.unwrap();
        assert_ne!(phone, laptop);

        auth.revoke_session(&phone).await.unwrap();
        assert!(auth.principal_for_session(&phone).await.unwrap().is_none());
        assert!(auth.principal_for_session(&laptop).await.unwrap().is_some());
    }

    #[test]
    fn tokens_are_unpredictable_and_full_width() {
        // 32 bytes as hex. A short or repeating token would be the weakest link in the whole
        // scheme, and nothing else here would show it.
        let a = random_token();
        assert_eq!(a.len(), TOKEN_BYTES * 2);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        let many: std::collections::HashSet<String> = (0..100).map(|_| random_token()).collect();
        assert_eq!(many.len(), 100);
    }
}
