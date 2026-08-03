import { describe, expect, it } from "vitest";
import { eta, parseEpochLine, rate } from "./lib";

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
