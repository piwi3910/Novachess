import { useState, type ReactNode } from "react";
import type { HistoryRecord } from "../types";
import { post } from "../api";

const CHART_W = 480;
const CHART_H = 160;
const PAD = 28;

// Points are parsed once, on arrival, by the caller (App.tsx) via
// lib.ts's parseEpochLine - this component only ever sees the already
// bounded, already parsed series, never the raw log.
function LossChart({ points }: { points: Array<{ epoch: number; loss: number }> }) {
  if (points.length < 2) return null;

  const maxEpoch = Math.max(...points.map((p) => p.epoch));
  const minEpoch = Math.min(...points.map((p) => p.epoch));
  const maxLoss = Math.max(...points.map((p) => p.loss));
  const minLoss = Math.min(...points.map((p) => p.loss));
  const epochSpan = maxEpoch - minEpoch || 1;
  const lossSpan = maxLoss - minLoss || 1;

  const x = (e: number) => PAD + ((e - minEpoch) / epochSpan) * (CHART_W - 2 * PAD);
  const y = (l: number) => CHART_H - PAD - ((l - minLoss) / lossSpan) * (CHART_H - 2 * PAD);

  const path = points.map((p) => `${x(p.epoch).toFixed(1)},${y(p.loss).toFixed(1)}`).join(" ");

  return (
    <svg className="loss-chart" width={CHART_W} height={CHART_H} viewBox={`0 0 ${CHART_W} ${CHART_H}`}>
      <line x1={PAD} y1={CHART_H - PAD} x2={CHART_W - PAD} y2={CHART_H - PAD} stroke="#333a47" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={CHART_H - PAD} stroke="#333a47" />
      <text x={PAD} y={CHART_H - 6} fill="#6c7684" fontSize="10">
        epoch {minEpoch}
      </text>
      <text x={CHART_W - PAD} y={CHART_H - 6} fill="#6c7684" fontSize="10" textAnchor="end">
        epoch {maxEpoch}
      </text>
      <text x={4} y={PAD} fill="#6c7684" fontSize="10">
        {maxLoss.toFixed(4)}
      </text>
      <text x={4} y={CHART_H - PAD} fill="#6c7684" fontSize="10">
        {minLoss.toFixed(4)}
      </text>
      <polyline points={path} fill="none" stroke="#4f8cff" strokeWidth={1.5} />
    </svg>
  );
}

function verdictCell(gate: HistoryRecord | undefined): ReactNode {
  if (!gate) return "—";
  const record = `+${gate.wins ?? 0} -${gate.losses ?? 0} =${gate.draws ?? 0}`;
  const elo = gate.elo_delta !== undefined ? ` (${gate.elo_delta >= 0 ? "+" : ""}${gate.elo_delta.toFixed(0)} elo)` : "";
  return (
    <span className={`verdict ${gate.promoted ? "promoted" : "rejected"}`}>
      {record}
      {elo} {gate.promoted ? "promoted ✓" : "rejected ✗"}
    </span>
  );
}

export default function Training({
  generation,
  history,
  epochs,
  tailLines,
}: {
  generation: number;
  history: HistoryRecord[];
  epochs: Array<{ epoch: number; loss: number }>;
  tailLines: string[];
}) {
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmForce, setConfirmForce] = useState<{ generation: number } | null>(null);

  const generations = Array.from(new Set(history.map((h) => h.generation))).sort((a, b) => a - b);

  const latestGateForCurrent = [...history]
    .filter((h) => h.type === "gate" && h.generation === generation)
    .sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime())
    .pop();
  const canAdvance = latestGateForCurrent?.promoted === true;

  async function runTrainer(gen: number, force: boolean) {
    setPending("trainer");
    setError(null);
    try {
      await post("/api/trainer/start", { generation: gen, force });
      setConfirmForce(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!force && /coordinator/i.test(msg)) {
        setConfirmForce({ generation: gen });
      }
      setError(msg);
    } finally {
      setPending(null);
    }
  }

  async function runGatekeeper() {
    setPending("gatekeeper");
    setError(null);
    try {
      await post("/api/gatekeeper/start", { generation });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(null);
    }
  }

  async function advance() {
    setPending("advance");
    setError(null);
    try {
      await post("/api/generation/advance", { to: generation + 1 });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(null);
    }
  }

  return (
    <section>
      <h2>Training</h2>
      {tailLines.length > 0 && (
        <>
          <LossChart points={epochs} />
          <div className="log-lines">{tailLines.join("\n")}</div>
        </>
      )}

      <table className="history">
        <thead>
          <tr>
            <th>gen</th>
            <th>positions</th>
            <th>final loss</th>
            <th>gate verdict</th>
          </tr>
        </thead>
        <tbody>
          {generations.map((gen) => {
            const dataset = history.find((h) => h.type === "dataset" && h.generation === gen);
            const training = history.find((h) => h.type === "training" && h.generation === gen);
            const gate = [...history]
              .filter((h) => h.type === "gate" && h.generation === gen)
              .sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime())
              .pop();
            return (
              <tr key={gen}>
                <td className="num">{gen}</td>
                <td className="num">{dataset?.positions?.toLocaleString() ?? "—"}</td>
                <td className="num">{training?.final_loss?.toFixed(4) ?? "—"}</td>
                <td>{verdictCell(gate)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="controls">
        <button disabled={pending !== null} onClick={() => runTrainer(generation, false)}>
          Run trainer
        </button>
        <button disabled={pending !== null} onClick={runGatekeeper}>
          Run gatekeeper
        </button>
        {canAdvance && (
          <button className="primary" disabled={pending !== null} onClick={advance}>
            Advance to gen {generation + 1}
          </button>
        )}
      </div>

      {confirmForce && (
        <div className="error-banner">
          <span>{error} — coordinator is still producing, run anyway?</span>
          <button onClick={() => runTrainer(confirmForce.generation, true)}>Run anyway</button>
          <button onClick={() => setConfirmForce(null)}>Cancel</button>
        </div>
      )}

      {error && !confirmForce && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}
    </section>
  );
}
