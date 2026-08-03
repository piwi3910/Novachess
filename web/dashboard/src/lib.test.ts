import { describe, expect, it } from "vitest";
import { eta, fmtBytes, fmtMillicores, metricScale, parseEpochLine, rate, selfplayStatus } from "./lib";

describe("eta", () => {
  it("is null when done or rate unknown", () => {
    expect(eta(10, 10, 5)).toBeNull();
    expect(eta(0, 10, 0)).toBeNull();
    expect(eta(0, 0, 5)).toBeNull();
  });
  it("divides remaining by rate", () => {
    expect(eta(400, 1000, 60)).toBe(10);
  });
});

describe("rate", () => {
  it("computes positions per second across the window", () => {
    expect(rate([[0, 0], [10_000, 5000]])).toBe(500);
  });
  it("is 0 with one sample", () => {
    expect(rate([[0, 0]])).toBe(0);
  });
});

describe("parseEpochLine", () => {
  it("parses the trainer's live epoch line", () => {
    expect(parseEpochLine("  epoch  3  loss 0.014205")).toEqual({ epoch: 3, loss: 0.014205 });
  });
  it("rejects other lines", () => {
    expect(parseEpochLine("final training loss 0.01, validation loss 0.02")).toBeNull();
  });
});

describe("selfplayStatus", () => {
  it("is empty until counts arrive", () => {
    expect(selfplayStatus(undefined, undefined)).toBe("");
  });
  it("names the paused state plainly", () => {
    expect(selfplayStatus(0, 0)).toBe("self-play paused");
  });
  it("shows workers and a parked coordinator", () => {
    expect(selfplayStatus(8, 0)).toBe("8 workers · coordinator parked");
  });
  it("shows workers and a running coordinator", () => {
    expect(selfplayStatus(8, 1)).toBe("8 workers · coordinator running");
  });
  it("surfaces error strings verbatim", () => {
    expect(selfplayStatus("boom", 0)).toBe("boom");
  });
});

describe("metricScale", () => {
  it("clamps to [0,1]", () => {
    expect(metricScale(500, 1000)).toBe(0.5);
    expect(metricScale(2000, 1000)).toBe(1);
    expect(metricScale(-5, 1000)).toBe(0);
    expect(metricScale(5, 0)).toBe(0);
  });
});

describe("fmtMillicores", () => {
  it("prints millicores under a core and cores above", () => {
    expect(fmtMillicores(970)).toBe("970m");
    expect(fmtMillicores(1500)).toBe("1.50");
  });
});

describe("fmtBytes", () => {
  it("prints Mi and Gi", () => {
    expect(fmtBytes(212 * 1024 * 1024)).toBe("212Mi");
    expect(fmtBytes(3 * 1024 * 1024 * 1024)).toBe("3.00Gi");
  });
});
