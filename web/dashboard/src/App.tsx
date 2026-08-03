import { useEffect, useRef, useState } from "react";
import type { HistoryRecord, Snapshot } from "./types";
import { getHistory, getState, stream } from "./api";
import Boards from "./components/Boards";
import Generation from "./components/Generation";
import Training from "./components/Training";

const SAMPLE_WINDOW = 30;

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [trainLog, setTrainLog] = useState<string[]>([]);
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
        setTrainLog((prev) => [...prev, line]);
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
      {snapshot && <Generation generation={snapshot.generation} samples={samplesRef.current} />}
      <Training
        generation={snapshot?.generation.generation ?? 0}
        history={history}
        trainLog={trainLog}
      />
    </div>
  );
}
