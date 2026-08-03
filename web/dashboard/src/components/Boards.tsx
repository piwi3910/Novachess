import { useEffect, useRef } from "react";
import type { Board } from "../types";
import { fmtBytes, fmtMillicores, fmtNps, metricScale } from "../lib";

const SPARK_W = 100;
const SPARK_H = 24;
const SPARK_LEN = 30;

// Resource ceilings the cpu/mem sparklines are scaled against: the 1-core
// request and the 512Mi limit from deploy/dashboard.yaml.
const CPU_CEILING_MILLICORES = 1000;
const MEM_CEILING_BYTES = 512 * 1024 * 1024;

interface WorkerHistory {
  nps: number[];
  cpu: number[];
  mem: number[];
}

function Sparkline({ values, max }: { values: number[]; max?: number }) {
  if (values.length < 2) {
    return <svg className="sparkline" width={SPARK_W} height={SPARK_H} />;
  }
  const hi = max ?? Math.max(...values, 1);
  const lo = max !== undefined ? 0 : Math.min(...values, 0);
  const span = hi - lo || 1;
  const step = SPARK_W / (SPARK_LEN - 1);
  const points = values
    .map((v, i) => {
      const x = i * step;
      const frac = max !== undefined ? metricScale(v, max) : (v - lo) / span;
      const y = SPARK_H - frac * SPARK_H;
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
  const historyRef = useRef<Map<string, WorkerHistory>>(new Map());

  useEffect(() => {
    const history = historyRef.current;
    const seen = new Set<string>();
    for (const b of boards) {
      seen.add(b.worker_id);
      const entry = history.get(b.worker_id) ?? { nps: [], cpu: [], mem: [] };
      entry.nps.push(b.nodes_per_second);
      if (entry.nps.length > SPARK_LEN) entry.nps.shift();
      if (b.cpu_millicores !== undefined) {
        entry.cpu.push(b.cpu_millicores);
        if (entry.cpu.length > SPARK_LEN) entry.cpu.shift();
      }
      if (b.memory_bytes !== undefined) {
        entry.mem.push(b.memory_bytes);
        if (entry.mem.length > SPARK_LEN) entry.mem.shift();
      }
      history.set(b.worker_id, entry);
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
          {boards.map((b) => {
            const history = historyRef.current.get(b.worker_id);
            return (
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
                <Sparkline values={history?.nps ?? []} />
                {b.cpu_millicores !== undefined && (
                  <div className="metric-row">
                    <Sparkline values={history?.cpu ?? []} max={CPU_CEILING_MILLICORES} />
                    <div className="metric-label">
                      <span>cpu</span>
                      <span className="num">{fmtMillicores(b.cpu_millicores)}</span>
                    </div>
                  </div>
                )}
                {b.memory_bytes !== undefined && (
                  <div className="metric-row">
                    <Sparkline values={history?.mem ?? []} max={MEM_CEILING_BYTES} />
                    <div className="metric-label">
                      <span>mem</span>
                      <span className="num">{fmtBytes(b.memory_bytes)}</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
