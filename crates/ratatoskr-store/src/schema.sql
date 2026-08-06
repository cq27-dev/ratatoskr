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
-- model produced it. Everything past `created_at` is nullable — a node that is not a model agent
-- (the implementer drives a coding CLI) has no usage to report, and a run recorded before these
-- columns existed has none either.
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
