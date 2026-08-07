import { useRef, useState } from "react";
import { short } from "../ui/text";
import { startRun, type ProjectView, type RunSummary } from "../api";

function Projects({
  projects,
  selected,
  onSelect,
}: {
  projects: ProjectView[];
  selected: string | null;
  onSelect: (slug: string) => void;
}) {
  // With one project there is nothing to choose, so the switcher stays out of the way.
  if (projects.length < 2) return null;
  return (
    <div className="projects">
      <div className="sec">
        <span>[ PROJECTS ]</span>
        <output>{projects.length}</output>
      </div>
      {projects.map((p) => (
        <button
          key={p.slug}
          className="proj"
          aria-current={p.slug === selected}
          onClick={() => onSelect(p.slug)}
          data-tip={p.dir}
        >
          {p.slug}
        </button>
      ))}
    </div>
  );
}

function NewRun({
  project,
  onStarted,
}: {
  project: string;
  onStarted: (runId: string) => void;
}) {
  const [issue, setIssue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!issue.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const runId = await startRun(project, issue);
      setIssue("");
      onStarted(runId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  };

  return (
    <form className="newrun" onSubmit={(e) => void submit(e)}>
      <div className="sec">
        <span>[ NEW RUN ]</span>
      </div>
      <textarea
        ref={box}
        value={issue}
        onChange={(e) => setIssue(e.target.value)}
        placeholder="describe the task…"
        rows={3}
        spellCheck={false}
        // Enter submits; newlines still available for multi-line issues.
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) void submit(e);
        }}
      />
      <button type="submit" disabled={busy || !issue.trim()}>
        {busy ? "STARTING…" : ">>> START RUN"}
      </button>
      {error && <p className="newrun-error hazard">{error}</p>}
    </form>
  );
}

export function Rail({
  projects,
  project,
  onProject,
  runs,
  selected,
  onSelect,
  onStarted,
  mayAct,
}: {
  projects: ProjectView[];
  project: string;
  onProject: (slug: string) => void;
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
  onStarted: (runId: string) => void;
  /** Whether this viewer may start runs. The server enforces it; this only stops offering. */
  mayAct: boolean;
}) {
  return (
    <nav className="rail">
      <Projects projects={projects} selected={project} onSelect={onProject} />
      {/* Hidden rather than disabled: a greyed-out box invites a viewer to wonder what they did
          wrong, where its absence simply reflects what this account is for. The route refuses it
          either way — a control that is not drawn is not a permission. */}
      {mayAct && <NewRun project={project} onStarted={onStarted} />}
      <div className="sec">
        <span>[ RUNS ]</span>
        <output>{runs.length}</output>
      </div>
      {runs.length === 0 && <p className="empty">no runs recorded</p>}
      {runs.map((r) => (
        <button
          key={r.run_id}
          className="run"
          aria-current={r.run_id === selected}
          onClick={() => onSelect(r.run_id)}
        >
          <span className="run-id">
            <samp>{short(r.run_id)}</samp>
            <span className={`st st--${r.status}`}>{r.status}</span>
          </span>
          <span className="run-sub">
            {r.updated_at.replace("T", " ").slice(0, 19)}
          </span>
        </button>
      ))}
    </nav>
  );
}
