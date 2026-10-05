import React from "react";
import { useEffect, useRef, useState, useCallback } from "react";

/**
 * Gates your app behind a "waking up server" screen for Render's free tier.
 *
 * Usage:
 *   <ServerWakeup apiUrl={import.meta.env.VITE_API_URL}>
 *     <App />
 *   </ServerWakeup>
 *
 * Requires a GET /health endpoint on your backend that returns 200 only
 * once the server (and any ML model) is actually ready.
 */

const STATUS = { WAKING: "waking", READY: "ready", FAILED: "failed" };

export default function ServerWakeup({
  apiUrl,
  children,
  healthPath = "/health",
  pollInterval = 3000,
  requestTimeout = 8000, // per attempt, so a hung cold-start request doesn't block polling
  giveUpAfter = 120000,
}) {
  const [status, setStatus] = useState(STATUS.WAKING);
  const [elapsed, setElapsed] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [attempt, setAttempt] = useState(0); // bump to restart
  const cancelled = useRef(false);

  const start = useCallback(() => {
    setStatus(STATUS.WAKING);
    setElapsed(0);
    setAttempts(0);
    setAttempt((n) => n + 1);
  }, []);

  // Polling loop
  useEffect(() => {
    cancelled.current = false;
    const startedAt = Date.now();

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    (async () => {
      while (!cancelled.current) {
        if (Date.now() - startedAt > giveUpAfter) {
          setStatus(STATUS.FAILED);
          return;
        }
        setAttempts((n) => n + 1);

        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), requestTimeout);
        try {
          const res = await fetch(`${apiUrl}${healthPath}`, {
            signal: ctrl.signal,
            cache: "no-store",
          });
          clearTimeout(timer);
          if (res.ok) {
            if (!cancelled.current) setStatus(STATUS.READY);
            return;
          }
        } catch {
          clearTimeout(timer);
        }
        await sleep(pollInterval);
      }
    })();

    return () => {
      cancelled.current = true;
    };
  }, [apiUrl, healthPath, pollInterval, requestTimeout, giveUpAfter, attempt]);

  // Elapsed-seconds ticker
  useEffect(() => {
    if (status !== STATUS.WAKING) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 250);
    return () => clearInterval(id);
  }, [status, attempt]);

  if (status === STATUS.READY) return children;

  const failed = status === STATUS.FAILED;
  const expected = 50; // Render's typical cold start, for the progress bar
  const pct = Math.min(95, (elapsed / expected) * 100);

  let message = "Starting the server";
  let detail = "Free hosting puts the server to sleep when it's idle. Waking it takes about a minute.";
  if (elapsed > 20) detail = "Still booting. The first request after a sleep is the slow one.";
  if (elapsed > expected) detail = "Taking longer than usual. It may be loading a model.";
  if (failed) {
    message = "The server didn't respond";
    detail = `No answer from ${healthPath} after ${Math.round(giveUpAfter / 1000)} seconds. Check the service logs on Render, then try again.`;
  }

  return (
    <>
      <style>{css}</style>
      <main className="wake" role="status" aria-live="polite">
        <div className="wake__card">
          <div className={`wake__dot ${failed ? "is-failed" : ""}`} aria-hidden="true" />
          <h1 className="wake__title">{message}</h1>
          <p className="wake__detail">{detail}</p>

          {!failed && (
            <>
              <div
                className="wake__bar"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(pct)}
                aria-label="Server startup progress"
              >
                <div className="wake__fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="wake__meta">
                {elapsed}s elapsed, {attempts} {attempts === 1 ? "check" : "checks"} sent
              </p>
            </>
          )}

          {failed && (
            <button className="wake__btn" onClick={start}>
              Try again
            </button>
          )}
        </div>
      </main>
    </>
  );
}

const css = `
.wake {
  --bg: #14161a;
  --card: #1c1f25;
  --line: #2b2f37;
  --text: #e8eaed;
  --muted: #9aa1ac;
  --accent: #7fd1b9;
  --fail: #ef8a7a;
  min-height: 100vh;
  min-height: 100dvh;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wake__card {
  width: 100%;
  max-width: 420px;
  padding: 32px 28px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
}
.wake__dot {
  width: 12px;
  height: 12px;
  margin-bottom: 20px;
  border-radius: 50%;
  background: var(--accent);
  animation: wake-pulse 1.6s ease-in-out infinite;
}
.wake__dot.is-failed {
  background: var(--fail);
  animation: none;
}
.wake__title {
  margin: 0 0 8px;
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.wake__detail {
  margin: 0 0 24px;
  max-width: 60ch;
  font-size: 0.95rem;
  line-height: 1.55;
  color: var(--muted);
}
.wake__bar {
  height: 4px;
  overflow: hidden;
  border-radius: 2px;
  background: var(--line);
}
.wake__fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.4s linear;
}
.wake__meta {
  margin: 12px 0 0;
  font-size: 0.8rem;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.wake__btn {
  padding: 10px 18px;
  font: inherit;
  font-weight: 600;
  color: var(--bg);
  background: var(--text);
  border: 0;
  border-radius: 8px;
  cursor: pointer;
}
.wake__btn:hover { background: #fff; }
.wake__btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
}
@keyframes wake-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.8); }
}
@media (prefers-reduced-motion: reduce) {
  .wake__dot { animation: none; }
  .wake__fill { transition: none; }
}
`;
