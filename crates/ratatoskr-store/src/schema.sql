CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    issue_id TEXT,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- Provenance, written once when the run starts. What it takes to say two runs were the same
    -- experiment: the resolved config, the graph that ran, and the tree it ran against.
    config_json TEXT,
    graph_hash TEXT,
    repo_sha TEXT
);

-- Per-node checkpoint snapshots: what a node was given, what it produced, what it cost, and which
-- model produced it. Everything past `created_at` is nullable — nodes that do no model work have
-- no usage to report, and a run recorded before these columns existed has none either.
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    node_name TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    input_json TEXT,
    -- The resolved route (`provider/model`), not the config alias that selected it: an alias
    -- repointed at a different model would otherwise rewrite the history of every past run.
    model TEXT,
    -- Which pass of the converge loop produced this row. Null for nodes that run once.
    iteration INTEGER,
    -- The turn's start is `created_at` minus this; it is not stored separately, because two
    -- columns that can disagree about one instant eventually do.
    duration_ms INTEGER,
    turns INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_input_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);

-- A run's event history, made durable.
--
-- The log files these come from rotate daily and are eventually removed, so a run older than the
-- retention window would otherwise lose its timeline entirely — and the timeline is what a
-- historical view is rebuilt from, and what an exported run has to carry to be worth analysing
-- somewhere else.
--
-- `payload_json` is the raw log record, unaltered: this table is a durable copy, not a second
-- interpretation of it. The three extracted columns are for ordering and filtering only.
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    -- Position within the run, as read. Part of the key so ingesting the same log twice is a
    -- no-op rather than a duplicate history.
    seq INTEGER NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    node TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

-- No index on (run_id, seq) here: the PRIMARY KEY above already is one, and on a rowid table
-- SQLite backs it with exactly that. A second identical index served no query and doubled the
-- index writes on the highest-volume table in the schema — every event of every run. Dropped
-- rather than merely not created, so an existing store loses it too.
DROP INDEX IF EXISTS idx_events_run;

-- What a run is FOR, which nothing else in the schema can answer.
--
-- Status, config and graph hash are all facts about how a run executed. Which arm of an experiment
-- it belongs to is a decision someone made, so it has to be recorded rather than derived.
CREATE TABLE IF NOT EXISTS run_tags (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    tag TEXT NOT NULL,
    PRIMARY KEY (run_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_run_tags_tag ON run_tags(tag);

-- The run list, which is the only query on a timer: every open dashboard re-reads it every ten
-- seconds. Without this it is a full scan of `runs` plus a temp B-tree for the sort — nothing at
-- seven runs, a scan per tab per ten seconds at ten thousand. The columns and their directions
-- match the ORDER BY exactly, because a mismatch leaves SQLite sorting anyway.
CREATE INDEX IF NOT EXISTS idx_runs_recent ON runs(updated_at DESC, run_id DESC);
