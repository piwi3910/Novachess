import { useEffect, useRef, useState } from "react";
import type { HistoryRecord, Snapshot } from "./types";
import { getHistory, getState, stream } from "./api";
import { parseEpochLine } from "./lib";
import Boards from "./components/Boards";
import Generation from "./components/Generation";
import Training from "./components/Training";

const SAMPLE_WINDOW = 30;
// Bounds the retained trainlog state for the life of the tab across a
// multi-hour run: the epoch series is capped generously above any real
// trainer -epochs value, and only the tail of raw lines is kept for display.
const MAX_EPOCH_POINTS = 500;
const TAIL_LINES = 10;

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [epochs, setEpochs] = useState<Array<{ epoch: number; loss: number }>>([]);
  const [tailLines, setTailLines] = useState<string[]>([]);
  const samplesRef = useRef<Array<[number, number]>>([]);
  const [, forceRender] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getState().then((s) => !cancelled && setSnapshot(s)).catch(() => {});
    getHistory().then((h) => !cancelled && setHistory(h)).catch(() => {});

    const unsubscribe = stream(
      (s) => {
        setSnapshot(s);
        const samples = samplesRef.current;
        samples.push([Date.now(), s.generation.positions]);
        if (samples.length > SAMPLE_WINDOW) samples.shift();
        forceRender((n) => n + 1);
      },
      (line) => {
        // Parse once on arrival with the single lib.ts parser; keep only the
        // parsed point (bounded series) and the raw line's tail, never the
        // unbounded raw log.
        const point = parseEpochLine(line);
        if (point) {
          setEpochs((prev) => {
            const next = [...prev, point];
            return next.length > MAX_EPOCH_POINTS ? next.slice(-MAX_EPOCH_POINTS) : next;
          });
        }
        setTailLines((prev) => {
          const next = [...prev, line];
          return next.length > TAIL_LINES ? next.slice(-TAIL_LINES) : next;
        });
      },
    );

    // History gains new records (dataset/training/gate/promotion) as jobs
    // complete server-side; there is no push event for it, so poll on a
    // light interval rather than refetching per trainlog line.
    const historyPoll = setInterval(() => {
      getHistory().then((h) => !cancelled && setHistory(h)).catch(() => {});
    }, 5000);

    return () => {
      cancelled = true;
      clearInterval(historyPoll);
      unsubscribe();
    };
  }, []);

  return (
    <div className="app">
      <h1>Novachess training dashboard</h1>
      <Boards boards={snapshot?.boards ?? []} />
      {snapshot && (
        <Generation
          generation={snapshot.generation}
          selfplay={snapshot.selfplay}
          samples={samplesRef.current}
        />
      )}
      <Training
        generation={snapshot?.generation.generation ?? 0}
        history={history}
        epochs={epochs}
        tailLines={tailLines}
      />
    </div>
  );
}
