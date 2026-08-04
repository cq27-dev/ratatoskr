CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    issue_id TEXT,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Per-node checkpoint snapshots. Populated starting Phase 2, once the executor exists to
-- write to it; the table is created now so Phase 2 doesn't need a schema migration on top
-- of Phase 0's.
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    node_name TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);
