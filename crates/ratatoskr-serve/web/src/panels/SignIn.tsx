/**
 * Who you are, in the masthead: a name and a way out, or a way in.
 *
 * Deliberately not a page and not a modal. A hosted instance can be readable without signing in,
 * so an interstitial would demand a password to look at something already public; and the state
 * that matters most — *which* of several people is about to start a run — belongs where it stays
 * visible, not behind a dialog that has been dismissed.
 */
import { useEffect, useRef, useState, type JSX } from "react";
import { login, logout, type Me } from "../api";

export function SignIn({
  me,
  onChange,
}: {
  me: Me | null;
  /** Called with the new identity after signing in or out, so the page can re-read what it may see. */
  onChange: (me: Me) => void;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const first = useRef<HTMLInputElement>(null);

  // The point of opening the form is to type in it.
  useEffect(() => {
    if (open) first.current?.focus();
  }, [open]);

  if (me?.principal_id) {
    return (
      <span className="who">
        <span className="who-name" data-tip={`signed in as ${me.display_name}`}>
          {me.display_name}
        </span>
        <span className="who-role">{me.role}</span>
        <button
          type="button"
          className="who-out"
          onClick={() => {
            void logout().then(() => onChange({}));
          }}
        >
          SIGN OUT
        </button>
      </span>
    );
  }

  if (!open) {
    return (
      <button type="button" className="who-in" onClick={() => setOpen(true)}>
        SIGN IN
      </button>
    );
  }

  return (
    <form
      className="signin"
      onSubmit={(event) => {
        event.preventDefault();
        setBusy(true);
        setError(null);
        login(username, password)
          .then((me) => {
            setOpen(false);
            setPassword("");
            onChange(me);
          })
          .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
          .finally(() => setBusy(false));
      }}
    >
      {/* Named and typed so a password manager recognises them. `autoComplete` is what makes the
          browser offer to fill and to save — without it people pick weaker passwords. */}
      <input
        ref={first}
        name="username"
        autoComplete="username"
        placeholder="USERNAME"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        disabled={busy}
      />
      <input
        name="password"
        type="password"
        autoComplete="current-password"
        placeholder="PASSWORD"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        disabled={busy}
      />
      <button type="submit" disabled={busy || !username || !password}>
        {busy ? "…" : "GO"}
      </button>
      <button
        type="button"
        onClick={() => {
          setOpen(false);
          setError(null);
        }}
        disabled={busy}
      >
        ✕
      </button>
      {/* One message for every way this fails, matching the server: naming which half was wrong
          would tell someone guessing which usernames exist. */}
      {error && <span className="hazard">{error}</span>}
    </form>
  );
}
