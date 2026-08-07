//! Who is asking, and whether they may.
//!
//! Two decisions are made here and nowhere else: turning a request into a [`Caller`], and deciding
//! whether that caller may do a given thing to a given project. Handlers name what they need and
//! get a compile error if they forget, because the project lookup they already had to call now
//! demands the caller and the access alongside the slug.
//!
//! **The session rides in a cookie, not a header.** The dashboard's live feed is `EventSource`,
//! which cannot set `Authorization`, so a bearer token would authenticate every route except the
//! one that streams a run. A token in the query string would work and would also be written to
//! every access log and proxy trace on the way.
//!
//! That choice brings CSRF with it, answered by two things rather than a token dance: `SameSite`
//! withholds the cookie from cross-site POSTs entirely, and every mutating route parses a JSON
//! body, which a form-based forgery cannot send.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::extract::{FromRef, FromRequestParts};
use axum::http::HeaderValue;
use axum::http::header::{COOKIE, SET_COOKIE};
use axum::http::request::Parts;
use ratatoskr_core::auth::{Access, Role, Visibility};
use ratatoskr_store::auth::{AuthStore, Principal};

use crate::ApiError;

/// The cookie a session lives in.
///
/// The `__Host-` prefix is a browser-enforced promise, not a naming convention: it refuses the
/// cookie unless it is `Secure`, has no `Domain`, and is pathed at `/`. That closes the hole where
/// a sibling subdomain over plain HTTP can plant a session cookie the main site will honour.
pub const COOKIE_NAME: &str = "__Host-rat_session";

/// Who is making a request.
#[derive(Debug, Clone)]
pub enum Caller {
    /// Nobody logged in. May read a public project and nothing else.
    Anonymous,
    /// A live session.
    Session(Principal),
}

impl Caller {
    /// What this caller may do, or `None` for nobody.
    pub fn role(&self) -> Option<Role> {
        match self {
            Caller::Anonymous => None,
            Caller::Session(p) => Some(p.role),
        }
    }

    /// Who to record against an action. Anonymous callers never reach a mutating route, so this
    /// only ever labels a log line for a real principal.
    pub fn id(&self) -> &str {
        match self {
            Caller::Anonymous => "anonymous",
            Caller::Session(p) => &p.principal_id,
        }
    }

    /// Whether this caller may do `access` to a project of `visibility`.
    ///
    /// The one place the two axes meet. Visibility answers "may a stranger read this", the role
    /// answers "may this person act at all", and neither alone is the question a handler asks.
    pub fn may(&self, visibility: Visibility, access: Access) -> bool {
        match (access, self.role()) {
            // Acting always needs an operator, whatever the project's visibility. A public project
            // is public to *read*; nothing is public to run code against.
            (Access::Act, role) => role.is_some_and(|r| r >= Role::Operator),
            (Access::Read, Some(_)) => true,
            (Access::Read, None) => visibility.is_public(),
        }
    }

    /// The error for a caller who may not, distinguishing "log in" from "you cannot".
    ///
    /// A 401 to an anonymous caller tells the dashboard to show a login form; a 403 to someone
    /// already logged in tells it not to, because logging in again will not help.
    pub(crate) fn denied(&self, what: &str) -> ApiError {
        match self {
            Caller::Anonymous => ApiError::Unauthorized(format!("log in to {what}")),
            Caller::Session(p) => ApiError::Forbidden(format!(
                "{} is a {} — {what} needs operator",
                p.display_name,
                p.role.as_str()
            )),
        }
    }
}

/// Lets a handler write `caller: Caller` and get one.
///
/// Infallible on purpose: a bad or expired cookie produces [`Caller::Anonymous`], not an error.
/// Whether anonymous is good enough is the authorization question, and answering it here would
/// mean a stale cookie turned a public project's page into an error instead of just logging you
/// out.
impl<S> FromRequestParts<S> for Caller
where
    AuthStore: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let Some(token) = session_cookie(parts) else {
            return Ok(Caller::Anonymous);
        };
        let auth = AuthStore::from_ref(state);
        match auth.principal_for_session(&token).await {
            Ok(Some(principal)) => Ok(Caller::Session(principal)),
            Ok(None) => Ok(Caller::Anonymous),
            Err(error) => {
                // The database being unreachable must not promote anyone. Logged rather than
                // swallowed, because "everyone is suddenly anonymous" needs to be findable.
                tracing::warn!(%error, "could not resolve a session; treating as anonymous");
                Ok(Caller::Anonymous)
            }
        }
    }
}

/// The session token from a request's `Cookie` header, if there is one.
///
/// Hand-parsed rather than pulling in a cookie crate: this reads one name out of a header, and the
/// parts of cookie handling that are genuinely fiddly — attributes, expiry, domain matching — are
/// the browser's job, not ours.
fn session_cookie(parts: &Parts) -> Option<String> {
    cookie_value(parts.headers.get(COOKIE)?.to_str().ok()?)
}

/// The session token out of a `Cookie` header's value.
fn cookie_value(header: &str) -> Option<String> {
    header.split(';').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        // Both names, because which one was *set* depends on whether the instance has TLS — and
        // this code cannot tell. Matching only the prefixed one made a loopback login set a cookie
        // that could never be read back, which looks exactly like a wrong password.
        let name = name.trim();
        (name == COOKIE_NAME || name == COOKIE_NAME_INSECURE).then(|| value.trim().to_string())
    })
}

/// The session token out of a plain header map, for handlers that take one rather than `Parts`.
pub fn token_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    let header = headers.get(COOKIE)?.to_str().ok()?;
    cookie_value(header)
}

/// The `Set-Cookie` that starts a session.
///
/// `SameSite=Lax` is the CSRF answer: it withholds the cookie from cross-site POSTs, which is
/// every route that can cause anything, while still sending it when someone follows a link to a
/// run — the case a stricter setting would break for no gain, since reading is what links are for.
pub fn set_cookie(token: &str, secure: bool) -> HeaderValue {
    // A hosted instance is behind TLS and gets the `__Host-` prefix and `Secure`. A loopback
    // instance is not, and a browser drops a `Secure` cookie from `http://localhost` — so there
    // the prefix has to go too, since it is only honoured with `Secure` set.
    let (name, secure_attr) = if secure {
        (COOKIE_NAME, "; Secure")
    } else {
        (COOKIE_NAME_INSECURE, "")
    };
    HeaderValue::from_str(&format!(
        "{name}={token}; Path=/; HttpOnly; SameSite=Lax{secure_attr}"
    ))
    .unwrap_or_else(|_| HeaderValue::from_static(""))
}

/// The cookie name used when there is no TLS to satisfy the `__Host-` prefix.
pub const COOKIE_NAME_INSECURE: &str = "rat_session";

/// The `Set-Cookie` that ends one. Same attributes, expired: a browser matches on name, path and
/// domain, so a clear that differs in any of them leaves the original in place.
pub fn clear_cookie(secure: bool) -> HeaderValue {
    let (name, secure_attr) = if secure {
        (COOKIE_NAME, "; Secure")
    } else {
        (COOKIE_NAME_INSECURE, "")
    };
    HeaderValue::from_str(&format!(
        "{name}=; Path=/; HttpOnly; SameSite=Lax{secure_attr}; Max-Age=0"
    ))
    .unwrap_or_else(|_| HeaderValue::from_static(""))
}

/// The header name for a `Set-Cookie`, re-exported so the routes do not import `http` themselves.
pub const SET_COOKIE_HEADER: axum::http::HeaderName = SET_COOKIE;

/// How many failed logins one username may accumulate before it has to wait.
const MAX_FAILURES: u32 = 10;

/// How long that wait is, and the window failures are counted over.
const COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// Failed logins, per username.
///
/// Keyed by username rather than by client address, deliberately. Behind a reverse proxy — which
/// is how a hosted instance is reached — every request arrives from the proxy, so an address key
/// throttles either everyone or no one unless we trust a forwarded header, and that header is
/// written by the client. A username key works the same wherever the request came from and
/// protects the thing actually under attack.
///
/// The cost is that someone can hold a known username in cooldown by failing on purpose. That is a
/// nuisance, not a compromise, and the alternative — no throttle — is a password guessed at
/// whatever rate the network allows.
#[derive(Default)]
pub struct LoginThrottle {
    failures: Mutex<HashMap<String, (u32, Instant)>>,
}

impl LoginThrottle {
    /// Whether this username may attempt a login now.
    pub fn may_try(&self, username: &str) -> bool {
        let mut failures = self.failures.lock().expect("login throttle poisoned");
        match failures.get(username) {
            Some(&(count, last)) if count >= MAX_FAILURES => {
                if last.elapsed() < COOLDOWN {
                    return false;
                }
                // The window has passed: forget it entirely rather than decaying, so a legitimate
                // owner returning after the cooldown gets a full allowance.
                failures.remove(username);
                true
            }
            _ => true,
        }
    }

    /// Record a failure.
    pub fn failed(&self, username: &str) {
        let mut failures = self.failures.lock().expect("login throttle poisoned");
        let entry = failures
            .entry(username.to_string())
            .or_insert((0, Instant::now()));
        entry.0 += 1;
        entry.1 = Instant::now();
    }

    /// Forget a username's failures. A successful login proves the attempts were the owner's.
    pub fn succeeded(&self, username: &str) {
        self.failures
            .lock()
            .expect("login throttle poisoned")
            .remove(username);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use ratatoskr_core::auth::{Access, Visibility};

    fn principal(role: Role) -> Principal {
        Principal {
            principal_id: "p1".to_string(),
            display_name: "KK".to_string(),
            role,
        }
    }

    fn parts_with(header: &str, value: &str) -> Parts {
        let request = Request::builder()
            .header(header, value)
            .body(())
            .expect("a valid test request");
        request.into_parts().0
    }

    #[test]
    fn anonymous_reads_a_public_project_and_nothing_else() {
        let caller = Caller::Anonymous;
        assert!(caller.may(Visibility::Public, Access::Read));
        assert!(!caller.may(Visibility::Private, Access::Read));
        // The one that matters: a public project is public to read, never to run code against.
        assert!(!caller.may(Visibility::Public, Access::Act));
    }

    #[test]
    fn a_viewer_reads_everything_and_acts_on_nothing() {
        let caller = Caller::Session(principal(Role::Viewer));
        assert!(caller.may(Visibility::Private, Access::Read));
        assert!(!caller.may(Visibility::Public, Access::Act));
    }

    #[test]
    fn operator_and_admin_may_act() {
        for role in [Role::Operator, Role::Admin] {
            let caller = Caller::Session(principal(role));
            assert!(caller.may(Visibility::Private, Access::Act), "{role}");
            assert!(caller.may(Visibility::Private, Access::Read), "{role}");
        }
    }

    #[test]
    fn denial_tells_a_stranger_to_log_in_and_a_viewer_not_to_bother() {
        // The distinction the dashboard renders: a 401 opens the login form, a 403 must not,
        // because logging in again does not raise your role.
        assert!(matches!(
            Caller::Anonymous.denied("start a run"),
            ApiError::Unauthorized(_)
        ));
        assert!(matches!(
            Caller::Session(principal(Role::Viewer)).denied("start a run"),
            ApiError::Forbidden(_)
        ));
    }

    #[test]
    fn a_username_is_throttled_after_enough_failures_and_freed_by_success() {
        let throttle = LoginThrottle::default();
        assert!(throttle.may_try("kk"));
        for _ in 0..MAX_FAILURES {
            assert!(throttle.may_try("kk"));
            throttle.failed("kk");
        }
        assert!(!throttle.may_try("kk"), "guessing has to stop somewhere");
        // Another account is unaffected: the throttle protects a username, not the login route.
        assert!(throttle.may_try("someone-else"));

        throttle.succeeded("kk");
        assert!(throttle.may_try("kk"));
    }

    #[test]
    fn the_session_cookie_is_found_among_others() {
        let parts = parts_with(
            "cookie",
            &format!("theme=dark; {COOKIE_NAME}=abc123; other=1"),
        );
        assert_eq!(session_cookie(&parts).as_deref(), Some("abc123"));
    }

    #[test]
    fn either_cookie_name_is_read_back() {
        // Which name was set depends on whether the instance has TLS. Reading only one of them
        // means every login on the other transport silently fails to stick.
        for name in [COOKIE_NAME, COOKIE_NAME_INSECURE] {
            let parts = parts_with("cookie", &format!("{name}=abc123"));
            assert_eq!(session_cookie(&parts).as_deref(), Some("abc123"), "{name}");
        }
    }

    #[test]
    fn a_request_without_the_cookie_carries_no_token() {
        assert_eq!(session_cookie(&parts_with("cookie", "theme=dark")), None);
        // A cookie whose name merely contains ours must not match.
        assert_eq!(
            session_cookie(&parts_with("cookie", &format!("not_{COOKIE_NAME}=abc"))),
            None
        );
    }

    #[test]
    fn a_secure_cookie_carries_the_attributes_the_host_prefix_requires() {
        let header = set_cookie("token", true);
        let value = header.to_str().unwrap();
        // `__Host-` is refused by the browser without every one of these.
        assert!(value.starts_with(COOKIE_NAME));
        assert!(value.contains("; Secure"));
        assert!(value.contains("; Path=/"));
        assert!(!value.contains("Domain"));
        assert!(value.contains("HttpOnly"));
        assert!(value.contains("SameSite=Lax"));
    }

    #[test]
    fn an_insecure_cookie_drops_both_the_prefix_and_the_flag() {
        // Over plain http a `Secure` cookie is discarded, and `__Host-` is only honoured with
        // `Secure` — so keeping either would silently break loopback logins.
        let value = set_cookie("token", false);
        let value = value.to_str().unwrap();
        assert!(value.starts_with(COOKIE_NAME_INSECURE));
        assert!(!value.contains("Secure"));
    }

    #[test]
    fn clearing_matches_the_cookie_it_clears() {
        // A browser matches on name, path and domain; a clear that differs in any of them leaves
        // the session cookie sitting there.
        for secure in [true, false] {
            let set = set_cookie("token", secure);
            let clear = clear_cookie(secure);
            let (set, clear) = (set.to_str().unwrap(), clear.to_str().unwrap());
            let name = set.split('=').next().unwrap();
            assert!(clear.starts_with(name));
            assert!(clear.contains("Path=/"));
            assert!(clear.contains("Max-Age=0"));
        }
    }
}
