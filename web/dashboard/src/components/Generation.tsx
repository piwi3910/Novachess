import { useEffect, useState } from "react";
import type { GenProgress, Snapshot } from "../types";
import { eta, rate, selfplayStatus } from "../lib";
import { post } from "../api";

function fmtEta(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `~${h}h ${m}m`;
}

const ACTION_DONE: Record<string, string> = {
  pause: "Paused — workers scaled to 0; in-flight units return to the queue",
  resume: "Resumed — workers scaling back up",
  stop: "Stopped — workers and coordinator scaled to 0",
  start: "Started — coordinator and workers scaling up",
};

export default function Generation({
  generation,
  selfplay,
  samples,
}: {
  generation: GenProgress;
  selfplay: Snapshot["selfplay"];
  samples: Array<[number, number]>;
}) {
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(t);
  }, [notice]);

  const r = rate(samples);
  const perHour = r * 3600;
  const etaSeconds = generation.target ? eta(generation.positions, generation.target, r) : null;
  const pct = generation.target ? Math.min(100, (generation.positions / generation.target) * 100) : null;

  async function run(action: string) {
    setPending(action);
    setError(null);
    setNotice(null);
    try {
      await post(`/api/selfplay/${action}`);
      setNotice(ACTION_DONE[action] ?? `${action} done`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(null);
    }
  }

  const status = selfplayStatus(selfplay?.workers, selfplay?.coordinator);

  return (
    <section>
      <h2>
        Generation {generation.generation}
        {status && <span className="selfplay-status">{status}</span>}
      </h2>
      {pct !== null && (
        <div className="progress-bar">
          <div style={{ width: `${pct}%` }} />
        </div>
      )}
      <div className="gen-stats">
        <span>
          positions <strong className="num">{generation.positions.toLocaleString()}</strong>
          {generation.target ? <span className="num"> / {generation.target.toLocaleString()}</span> : null}
        </span>
        <span>
          rate <strong className="num">{Math.round(perHour).toLocaleString()}</strong>/h
        </span>
        <span>
          ETA <strong className="num">{etaSeconds !== null ? fmtEta(etaSeconds) : "—"}</strong>
        </span>
        <span>
          batches <strong className="num">{generation.batches.toLocaleString()}</strong>
        </span>
      </div>
      <div className="controls">
        <button disabled={pending !== null} onClick={() => run("pause")}>
          Pause
        </button>
        <button disabled={pending !== null} onClick={() => run("resume")}>
          Resume
        </button>
        <button disabled={pending !== null} onClick={() => run("stop")}>
          Stop
        </button>
        <button disabled={pending !== null} onClick={() => run("start")}>
          Start
        </button>
      </div>
      {notice && (
        <div className="notice-banner">
          <span>{notice}</span>
          <button onClick={() => setNotice(null)}>×</button>
        </div>
      )}
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}
    </section>
  );
}
