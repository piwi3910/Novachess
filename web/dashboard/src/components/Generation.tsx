import { useState } from "react";
import type { GenProgress } from "../types";
import { eta, rate } from "../lib";
import { post } from "../api";

function fmtEta(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `~${h}h ${m}m`;
}

export default function Generation({
  generation,
  samples,
}: {
  generation: GenProgress;
  samples: Array<[number, number]>;
}) {
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const r = rate(samples);
  const perHour = r * 3600;
  const etaSeconds = generation.target ? eta(generation.positions, generation.target, r) : null;
  const pct = generation.target ? Math.min(100, (generation.positions / generation.target) * 100) : null;

  async function run(action: string) {
    setPending(action);
    setError(null);
    try {
      await post(`/api/selfplay/${action}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(null);
    }
  }

  return (
    <section>
      <h2>Generation {generation.generation}</h2>
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
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}
    </section>
  );
}
