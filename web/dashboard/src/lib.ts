// Estimated completion from the observed rate over the sampled window.
export function eta(positions: number, target: number, ratePerSec: number): number | null {
  if (!target || ratePerSec <= 0 || positions >= target) return null;
  return (target - positions) / ratePerSec;
}

// Rate from a rolling window of (time, positions) samples.
export function rate(samples: Array<[number, number]>): number {
  if (samples.length < 2) return 0;
  const [t0, p0] = samples[0];
  const [t1, p1] = samples[samples.length - 1];
  if (t1 <= t0) return 0;
  return ((p1 - p0) / (t1 - t0)) * 1000;
}

export function parseEpochLine(line: string): { epoch: number; loss: number } | null {
  const m = line.match(/epoch\s+(\d+)\s+loss\s+([\d.]+)/);
  return m ? { epoch: Number(m[1]), loss: Number(m[2]) } : null;
}

export function fmtNps(nps: number): string {
  return nps >= 1e6 ? (nps / 1e6).toFixed(2) + "M" : Math.round(nps / 1e3) + "k";
}
