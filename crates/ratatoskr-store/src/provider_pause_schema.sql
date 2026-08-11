-- Automatic provider pauses are dashboard-instance state, not project-checkpoint state: one
-- dashboard can watch several projects, and only the run process may write a project's store.
--
-- A run id is unique only inside its project. `exited` is a durable fence written after confirmed
-- child exit: it clears deliveries and rejects a late request that races process death.
-- `last_seen_ms` is refreshed by every paused poll and only drives suspect-liveness warnings.
CREATE TABLE IF NOT EXISTS provider_pause_runs (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    latest_generation INTEGER NOT NULL,
    exited INTEGER NOT NULL DEFAULT 0 CHECK (exited IN (0, 1)),
    last_seen_ms INTEGER NOT NULL,
    PRIMARY KEY (project, run_id)
);

-- A resumed row is a durable tombstone. It is retained until confirmed child exit so a lost
-- acknowledgement response can be retried after a dashboard restart and still observe a Stop
-- that landed after the first acknowledgement committed.
CREATE TABLE IF NOT EXISTS provider_pause_generations (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    resumed INTEGER NOT NULL DEFAULT 0 CHECK (resumed IN (0, 1)),
    PRIMARY KEY (project, run_id, generation),
    FOREIGN KEY (project, run_id)
        REFERENCES provider_pause_runs(project, run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS provider_pause_waiters (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    waiter TEXT NOT NULL,
    -- The node is part of delivery identity: Stop is node-specific, unlike the run-wide Resume.
    node TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
    PRIMARY KEY (project, run_id, generation, waiter),
    FOREIGN KEY (project, run_id, generation)
        REFERENCES provider_pause_generations(project, run_id, generation) ON DELETE CASCADE
);

-- A Stop remains durable until the operator starts that node again or the child exits. It
-- intentionally exists before any waiter when Stop races registration, and overrides a retained
-- Continue tombstone when Stop races acknowledgement delivery.
CREATE TABLE IF NOT EXISTS provider_pause_stops (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    node TEXT NOT NULL,
    PRIMARY KEY (project, run_id, node),
    FOREIGN KEY (project, run_id)
        REFERENCES provider_pause_runs(project, run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_provider_pause_waiters_outstanding
    ON provider_pause_waiters (project, run_id, generation, acknowledged);

CREATE INDEX IF NOT EXISTS idx_provider_pause_runs_liveness
    ON provider_pause_runs (exited, last_seen_ms);
