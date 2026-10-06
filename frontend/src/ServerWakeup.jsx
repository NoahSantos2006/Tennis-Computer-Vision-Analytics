import React from "react";
import { useEffect, useRef, useState, useCallback } from "react";
import "./ServerWakeup.css";

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
  giveUpAfter = 300000,
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
    <main className="server-wakeup-page" role="status" aria-live="polite">
      <div className="server-wakeup-card">
        <div className={`server-wakeup-dot ${failed ? "failed" : ""}`} aria-hidden="true" />
        <h1 className="server-wakeup-title">{message}</h1>
        <p className="server-wakeup-detail">{detail}</p>

        {!failed && (
          <>
            <div
              className="server-wakeup-progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(pct)}
              aria-label="Server startup progress"
            >
              <div className="server-wakeup-progress-bar" style={{ width: `${pct}%` }} />
            </div>
            <p className="server-wakeup-meta">
              {elapsed}s elapsed, {attempts} {attempts === 1 ? "check" : "checks"} sent
            </p>
          </>
        )}

        {failed && (
          <button className="server-wakeup-button" onClick={start}>
            Try again
          </button>
        )}
      </div>
    </main>
  );
}
