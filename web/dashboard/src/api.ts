import type { HistoryRecord, Snapshot } from "./types";

export async function getState(): Promise<Snapshot> {
  const r = await fetch("/api/state");
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getHistory(): Promise<HistoryRecord[]> {
  const r = await fetch("/api/history");
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function post(path: string, body?: unknown): Promise<unknown> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error ?? r.statusText);
  return data;
}

export function stream(onState: (s: Snapshot) => void, onTrainLog: (line: string) => void): () => void {
  const es = new EventSource("/api/stream");
  es.addEventListener("state", (e) => onState(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("trainlog", (e) => onTrainLog(JSON.parse((e as MessageEvent).data).line));
  // EventSource reconnects on its own; nothing to do on error but log.
  return () => es.close();
}
