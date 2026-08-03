import { useEffect, useRef } from "react";
import type { Board } from "../types";
import { fmtNps } from "../lib";

const SPARK_W = 100;
const SPARK_H = 24;
const SPARK_LEN = 30;

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) {
    return <svg className="sparkline" width={SPARK_W} height={SPARK_H} />;
  }
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const step = SPARK_W / (SPARK_LEN - 1);
  const points = values
    .map((v, i) => {
      const x = i * step;
      const y = SPARK_H - ((v - min) / span) * SPARK_H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="sparkline" width={SPARK_W} height={SPARK_H} viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}>
      <polyline points={points} fill="none" stroke="#4f8cff" strokeWidth={1.5} />
    </svg>
  );
}

export default function Boards({ boards }: { boards: Board[] }) {
  const historyRef = useRef<Map<string, number[]>>(new Map());

  useEffect(() => {
    const history = historyRef.current;
    const seen = new Set<string>();
    for (const b of boards) {
      seen.add(b.worker_id);
      const series = history.get(b.worker_id) ?? [];
      series.push(b.nodes_per_second);
      if (series.length > SPARK_LEN) series.shift();
      history.set(b.worker_id, series);
    }
    for (const id of Array.from(history.keys())) {
      if (!seen.has(id)) history.delete(id);
    }
  }, [boards]);

  return (
    <section>
      <h2>Self-play workers</h2>
      {boards.length === 0 ? (
        <p className="no-boards">No workers reporting.</p>
      ) : (
        <div className="board-grid">
          {boards.map((b) => (
            <div key={b.worker_id} className={`board-card${b.stale ? " stale" : ""}`}>
              <div className="worker-name">{b.node_name}</div>
              <div className="board-row">
                <span>nps</span>
                <span className="num">{fmtNps(b.nodes_per_second)}</span>
              </div>
              <div className="board-row">
                <span>unit</span>
                <span className="mono">{b.current_unit ?? "idle"}</span>
              </div>
              <div className="board-row">
                <span>engine</span>
                <span className="mono">{b.engine_version}</span>
              </div>
              <Sparkline values={historyRef.current.get(b.worker_id) ?? []} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
