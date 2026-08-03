export interface Board {
  worker_id: string;
  node_name: string;
  current_unit?: string;
  engine_version: string;
  nodes_per_second: number;
  games_completed: number;
  last_seen: string;
  stale: boolean;
  cpu_millicores?: number;
  memory_bytes?: number;
}
export interface GenProgress {
  generation: number;
  positions: number;
  batches: number;
  target?: number;
  updated_at: string;
}
export interface Snapshot {
  boards: Board[];
  generation: GenProgress;
  selfplay?: { workers: number | string; coordinator: number | string };
}
export interface HistoryRecord {
  type: "dataset" | "training" | "gate" | "promotion";
  generation: number;
  positions?: number;
  promoted?: boolean;
  wins?: number; losses?: number; draws?: number;
  elo_delta?: number; los?: number;
  reason?: string; network_uri?: string; job_name?: string;
  final_loss?: number;
  at: string;
}
