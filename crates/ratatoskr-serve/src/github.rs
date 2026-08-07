//! Starting a run by mentioning the bot in a GitHub issue.
//!
//! The shape is the same one the dashboard uses, with a different way of saying who is asking.
//! GitHub names the person who commented; that maps to a principal through
//! [`AuthStore::principal_for_identity`], and the principal needs `operator` exactly as it would to
//! press the button. Nothing about "it came from GitHub" grants anything: the repository being
//! public means anyone in the world can send this endpoint a comment.
//!
//! **The signature is the only thing that makes a request trustworthy.** The endpoint is public by
//! necessity — GitHub has to reach it — so every field in the body, including who GitHub says sent
//! it, is attacker-controlled until the HMAC checks out. Verify first, parse after.

use axum::body::Bytes;
use axum::http::HeaderMap;
use hmac::{Hmac, Mac};
use ratatoskr_core::auth::Role;
use ratatoskr_store::auth::AuthStore;
use serde::Deserialize;
use sha2::Sha256;

/// The provider token stored in `identities.provider`.
pub const GITHUB: &str = "github";

/// The header GitHub signs the body with.
const SIGNATURE: &str = "x-hub-signature-256";

/// The header naming what happened, since the payload alone does not say.
const EVENT: &str = "x-github-event";

/// What this integration needs to be usable at all.
#[derive(Clone)]
pub struct GitHubConfig {
    /// The account the bot posts as, without the `@`. A comment must mention it to be a request.
    pub bot: String,
    /// The webhook secret, shared with GitHub. Never logged, never echoed.
    pub secret: String,
}

/// Why a webhook did not start a run.
///
/// Deliberately blunt at the boundary. A caller who fails the signature learns only that they
/// failed it, because every other distinction — is that a repository you watch, is that a user you
/// know — is a fact about this instance that an unauthenticated caller may not have.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The signature was absent, malformed, or wrong.
    Unsigned,
    /// A well-formed, correctly signed delivery this integration has nothing to do with: a event
    /// kind we do not act on, a comment that does not mention the bot, an edit rather than a new
    /// comment. Answered 200, because GitHub retries anything else and there is nothing to retry.
    NotForUs,
    /// Signed and addressed to us, by someone this instance does not know or does not trust.
    NotAuthorized,
    /// Signed and addressed to us, about a repository this instance does not serve.
    UnknownRepository(String),
}

/// A comment that asks the bot to do something.
#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    /// The repository, as `owner/name`.
    pub repository: String,
    pub issue_number: u64,
    /// GitHub's immutable numeric id for the commenter, which is what an identity is keyed on.
    pub sender_id: String,
    /// The commenter's login, for the log line — never for authorization, since it can change.
    pub sender_login: String,
    /// What was asked, with the mention removed.
    pub instruction: String,
}

/// The subset of the payload this reads. Everything else GitHub sends is ignored.
#[derive(Debug, Deserialize)]
struct Payload {
    action: String,
    issue: Issue,
    comment: Comment,
    repository: Repository,
}

#[derive(Debug, Deserialize)]
struct Issue {
    number: u64,
}

#[derive(Debug, Deserialize)]
struct Comment {
    body: String,
    user: User,
}

#[derive(Debug, Deserialize)]
struct User {
    /// Numeric and immutable, unlike `login`.
    id: u64,
    login: String,
}

#[derive(Debug, Deserialize)]
struct Repository {
    full_name: String,
}

/// Whether the body carries GitHub's signature for `secret`.
///
/// Constant-time, through `Mac::verify_slice`: a byte-by-byte comparison that returns early leaks
/// how much of a guess was right, which is enough to recover a signature one byte at a time.
pub fn verify(headers: &HeaderMap, body: &[u8], secret: &str) -> bool {
    let Some(header) = headers.get(SIGNATURE).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let Some(hex) = header.strip_prefix("sha256=") else {
        return false;
    };
    let Ok(expected) = decode_hex(hex) else {
        return false;
    };
    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    mac.update(body);
    mac.verify_slice(&expected).is_ok()
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, ()> {
    if !hex.len().is_multiple_of(2) {
        return Err(());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| ()))
        .collect()
}

/// What a delivery is asking for, or why it is not asking for anything.
///
/// Signature first, before the body is even parsed — see the module docs.
pub fn read(headers: &HeaderMap, body: &Bytes, config: &GitHubConfig) -> Result<Request, Refusal> {
    if !verify(headers, body, &config.secret) {
        return Err(Refusal::Unsigned);
    }
    // `ping` is what GitHub sends when the webhook is saved, and answering it is how the setup page
    // goes green.
    let event = headers
        .get(EVENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if event != "issue_comment" {
        return Err(Refusal::NotForUs);
    }
    let payload: Payload = serde_json::from_slice(body).map_err(|_| Refusal::NotForUs)?;
    // Created only. Acting on `edited` would let someone edit an old comment of theirs into a new
    // instruction, and acting on `deleted` is meaningless.
    if payload.action != "created" {
        return Err(Refusal::NotForUs);
    }
    let instruction =
        instruction_for(&payload.comment.body, &config.bot).ok_or(Refusal::NotForUs)?;
    // The bot's own comments mention it constantly — the questions it asks are addressed to
    // someone. Without this it answers itself in a loop.
    if payload.comment.user.login.eq_ignore_ascii_case(&config.bot) {
        return Err(Refusal::NotForUs);
    }
    Ok(Request {
        repository: payload.repository.full_name,
        issue_number: payload.issue.number,
        sender_id: payload.comment.user.id.to_string(),
        sender_login: payload.comment.user.login,
        instruction,
    })
}

/// What a comment is asking, if it is addressed to the bot.
///
/// The mention has to be its own word: a comment discussing `@ratatoskr-docs`, or an email address
/// ending in the handle, is not addressed to us. Everything after the mention on the rest of the
/// comment is the instruction — a mention with nothing after it is a greeting, not a request.
fn instruction_for(body: &str, bot: &str) -> Option<String> {
    let mention = format!("@{bot}");
    let at = body
        .match_indices(&mention)
        .find(|(i, _)| {
            let before = body[..*i].chars().next_back();
            let after = body[i + mention.len()..].chars().next();
            // Preceded by nothing or whitespace, and followed by nothing or a non-name character.
            before.is_none_or(char::is_whitespace)
                && after.is_none_or(|c| !c.is_alphanumeric() && c != '-' && c != '_')
        })
        .map(|(i, _)| i)?;
    let rest = body[at + mention.len()..].trim();
    (!rest.is_empty()).then(|| rest.to_string())
}

/// Who is asking, if this instance knows them and trusts them to start a run.
pub async fn principal_for(
    auth: &AuthStore,
    request: &Request,
) -> Result<ratatoskr_store::auth::Principal, Refusal> {
    let found = auth
        .principal_for_identity(GITHUB, &request.sender_id)
        .await
        .map_err(|error| {
            tracing::warn!(%error, "could not resolve a GitHub identity");
            Refusal::NotAuthorized
        })?;
    match found {
        Some(principal) if principal.role >= Role::Operator => Ok(principal),
        // Both cases are "no": someone this instance has never heard of, and someone it knows but
        // has only given read access to. Neither gets to spend money.
        _ => Err(Refusal::NotAuthorized),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn config() -> GitHubConfig {
        GitHubConfig {
            bot: "ratatoskr".to_string(),
            secret: "it's a secret to everybody".to_string(),
        }
    }

    /// The signature GitHub would send for this body.
    fn sign(body: &[u8], secret: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("any key length");
        mac.update(body);
        let out = mac.finalize().into_bytes();
        format!(
            "sha256={}",
            out.iter().map(|b| format!("{b:02x}")).collect::<String>()
        )
    }

    fn headers(body: &[u8], secret: &str, event: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            SIGNATURE,
            HeaderValue::from_str(&sign(body, secret)).expect("ascii"),
        );
        headers.insert(EVENT, HeaderValue::from_str(event).expect("ascii"));
        headers
    }

    fn comment(body: &str, login: &str, id: u64) -> Bytes {
        Bytes::from(
            serde_json::json!({
                "action": "created",
                "issue": { "number": 164 },
                "comment": { "body": body, "user": { "id": id, "login": login } },
                "repository": { "full_name": "cq27-dev/ratatoskr" },
            })
            .to_string(),
        )
    }

    #[test]
    fn a_correctly_signed_mention_is_a_request() {
        let config = config();
        let body = comment("@ratatoskr please fix the flaky retry", "kk", 1234);
        let request = read(
            &headers(&body, &config.secret, "issue_comment"),
            &body,
            &config,
        )
        .expect("a request");
        assert_eq!(request.repository, "cq27-dev/ratatoskr");
        assert_eq!(request.issue_number, 164);
        // The numeric id, not the login: a login can be changed and reassigned.
        assert_eq!(request.sender_id, "1234");
        assert_eq!(request.instruction, "please fix the flaky retry");
    }

    #[test]
    fn an_unsigned_or_wrongly_signed_delivery_is_refused() {
        let config = config();
        let body = comment("@ratatoskr go", "kk", 1234);

        // No signature at all.
        let mut bare = HeaderMap::new();
        bare.insert(EVENT, HeaderValue::from_static("issue_comment"));
        assert_eq!(read(&bare, &body, &config), Err(Refusal::Unsigned));

        // Signed with the wrong secret — the case that matters, since the endpoint is public.
        let wrong = headers(&body, "not the secret", "issue_comment");
        assert_eq!(read(&wrong, &body, &config), Err(Refusal::Unsigned));

        // Correctly signed, but for a different body: this is the replay of one delivery's
        // signature onto another's payload.
        let other = comment("@ratatoskr delete everything", "kk", 1234);
        let stolen = headers(&body, &config.secret, "issue_comment");
        assert_eq!(read(&stolen, &other, &config), Err(Refusal::Unsigned));
    }

    #[test]
    fn a_malformed_signature_header_is_refused_rather_than_panicking() {
        let config = config();
        let body = comment("@ratatoskr go", "kk", 1234);
        for value in [
            "",
            "sha256=",
            "sha256=zz",
            "sha256=abc",
            "abc123",
            "sha1=abc",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(SIGNATURE, HeaderValue::from_str(value).expect("ascii"));
            headers.insert(EVENT, HeaderValue::from_static("issue_comment"));
            assert_eq!(
                read(&headers, &body, &config),
                Err(Refusal::Unsigned),
                "{value:?}"
            );
        }
    }

    #[test]
    fn only_a_new_comment_counts() {
        // Acting on `edited` would let someone edit a year-old comment of their own into a fresh
        // instruction, and nothing about the delivery would look unusual.
        let config = config();
        let mut value: serde_json::Value =
            serde_json::from_slice(&comment("@ratatoskr go", "kk", 1234)).unwrap();
        value["action"] = serde_json::json!("edited");
        let body = Bytes::from(value.to_string());
        assert_eq!(
            read(
                &headers(&body, &config.secret, "issue_comment"),
                &body,
                &config
            ),
            Err(Refusal::NotForUs)
        );
    }

    #[test]
    fn other_event_kinds_are_not_ours() {
        let config = config();
        let body = comment("@ratatoskr go", "kk", 1234);
        for event in ["ping", "push", "pull_request", ""] {
            assert_eq!(
                read(&headers(&body, &config.secret, event), &body, &config),
                Err(Refusal::NotForUs),
                "{event}"
            );
        }
    }

    #[test]
    fn the_bot_does_not_answer_itself() {
        // Its own comments mention it — the questions it asks are addressed to someone. Without
        // this the first question starts a run that asks another question.
        let config = config();
        let body = comment("@ratatoskr what should I assume here?", "ratatoskr", 99);
        assert_eq!(
            read(
                &headers(&body, &config.secret, "issue_comment"),
                &body,
                &config
            ),
            Err(Refusal::NotForUs)
        );
    }

    #[test]
    fn the_mention_has_to_be_the_whole_word() {
        assert_eq!(
            instruction_for("@ratatoskr do the thing", "ratatoskr").as_deref(),
            Some("do the thing")
        );
        // A different bot whose handle starts with ours.
        assert_eq!(
            instruction_for("@ratatoskr-docs do the thing", "ratatoskr"),
            None
        );
        // Part of a longer word, or of an address.
        assert_eq!(
            instruction_for("mail bot@ratatoskr.dev about it", "ratatoskr"),
            None
        );
        assert_eq!(instruction_for("see @@ratatoskr", "ratatoskr"), None);
        // A mention with nothing after it is a greeting, not a request.
        assert_eq!(instruction_for("@ratatoskr", "ratatoskr"), None);
        assert_eq!(instruction_for("@ratatoskr   ", "ratatoskr"), None);
    }

    #[test]
    fn the_instruction_is_what_follows_the_mention() {
        // Mid-sentence, and across lines: people write a paragraph and then ask.
        assert_eq!(
            instruction_for("I think @ratatoskr should look at this", "ratatoskr").as_deref(),
            Some("should look at this")
        );
        assert_eq!(
            instruction_for("@ratatoskr\nfix the retry\nit flakes", "ratatoskr").as_deref(),
            Some("fix the retry\nit flakes")
        );
    }
}
