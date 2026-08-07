-- Who may use a hosted instance.
--
-- Its own database, not a table in a project's checkpoint store, for two reasons. A `serve`
-- process watches several projects and each owns its store, so there is no project this could
-- belong to. And the checkpoint store is single-writer by construction, owned by the run process —
-- serve only reads it. Sessions are written on every login, so putting them there would break that.

-- Someone or something that acts: a person, or later a bot with its own credentials.
--
-- Identity is deliberately not a column here. A principal is *who*, and the ways to prove you are
-- them live in `identities` — which is what lets one person hold a password and a GitHub account
-- and be recognised as the same operator by both.
CREATE TABLE IF NOT EXISTS principals (
    principal_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    -- `viewer` | `operator` | `admin`. Parsed as a closed enum on read; an unknown token is an
    -- error rather than a default, because both directions of guessing are wrong.
    role TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- Set rather than deleting the row: a run this principal started still refers to them, and
    -- revoking access should not rewrite who did what.
    disabled_at TEXT
);

-- One way of proving you are a principal.
--
-- `provider` is `local` today and `github` when the bot lands. `subject` is what that provider
-- calls you: a username for `local`, an immutable numeric user id for `github` — an id and not a
-- login, because a login can be changed and then points at someone else.
--
-- `secret` holds an argon2 PHC string for `local` and is null for everything else, where the
-- provider does the proving. The PHC string carries its own parameters, so raising the cost later
-- applies on next login rather than needing a migration.
CREATE TABLE IF NOT EXISTS identities (
    provider TEXT NOT NULL,
    subject TEXT NOT NULL,
    principal_id TEXT NOT NULL REFERENCES principals(principal_id) ON DELETE CASCADE,
    secret TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (provider, subject)
);

CREATE INDEX IF NOT EXISTS idx_identities_principal ON identities(principal_id);

-- A logged-in browser.
--
-- The key is a SHA-256 of the token, never the token: a leaked copy of this file then cannot be
-- replayed against a live server. Hashing is enough here where passwords need argon2, because the
-- token is 256 bits of CSPRNG output — there is no dictionary to attack, only the whole space.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES principals(principal_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at TEXT NOT NULL,
    -- What was logged in from, so a principal can recognise a session they do not remember making.
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_principal ON sessions(principal_id);
