import { describe, expect, it } from "vitest";
import { aiInputForMode, basename, componentProgressPercent, containedVideoFrame, isPipelineCancellation, isSupportedVideo, originalSrtExport, outputFilename, parseSummary, pipelineAiHandoff, regionBoxInFrame, resultQuality, updateRegionBox } from "./pipeline";

describe("pipeline helpers", () => {
  it("recognizes only the explicit pipeline cancellation message", () => {
    expect(isPipelineCancellation("Processing was cancelled.")).toBe(true);
    expect(isPipelineCancellation("Could not open C:\\Films\\cancelled_movie.mp4")).toBe(false);
    expect(isPipelineCancellation("cancelled while loading model")).toBe(false);
  });

  it("maps wide and tall video frames inside a letterboxed preview", () => {
    const wide = containedVideoFrame(400, 400, 1920, 1080);
    expect(wide).toEqual({ x: 0, y: 0.21875, w: 1, h: 0.5625 });
    const tall = containedVideoFrame(400, 400, 1080, 1920);
    expect(tall).toEqual({ x: 0.21875, y: 0, w: 0.5625, h: 1 });
    const sourceRegion = { x: 0.1, y: 0.2, w: 0.8, h: 0.6 };
    const wideSelection = regionBoxInFrame(sourceRegion, wide);
    const tallSelection = regionBoxInFrame(sourceRegion, tall);
    expect(wideSelection.x).toBeCloseTo(0.1);
    expect(wideSelection.y).toBeCloseTo(0.33125);
    expect(wideSelection.w).toBeCloseTo(0.8);
    expect(wideSelection.h).toBeCloseTo(0.3375);
    expect(tallSelection.x).toBeCloseTo(0.275);
    expect(tallSelection.y).toBeCloseTo(0.2);
    expect(tallSelection.w).toBeCloseTo(0.45);
    expect(tallSelection.h).toBeCloseTo(0.6);
  });

  it("clamps dragged crop regions to source-frame bounds", () => {
    const moved = updateRegionBox({ x: 0.1, y: 0.2, w: 0.8, h: 0.6 }, "move", 1, -1);
    expect(moved.x).toBeCloseTo(0.2);
    expect(moved.y).toBe(0);
    expect(moved.w).toBeCloseTo(0.8);
    expect(moved.h).toBeCloseTo(0.6);
    const resized = updateRegionBox({ x: 0.1, y: 0.2, w: 0.8, h: 0.6 }, "se", 1, 1);
    expect(resized.x + resized.w).toBe(1);
    expect(resized.y + resized.h).toBe(1);
  });

  it("parses only the result block and keeps values containing equals", () => {
    const result = parseSummary([
      "noise=ignored",
      "--- SUBHOOPER PIPELINE 0.3.7 RESULT START ---",
      "Pipeline=COMPLETE",
      "SRT=C:\\Video=One\\result.srt",
      "SuspiciousShortCues=0",
      "Error=",
      "--- SUBHOOPER PIPELINE 0.3.7 RESULT END ---",
    ].join("\r\n"));
    expect(result.Pipeline).toBe("COMPLETE");
    expect(result.SRT).toBe("C:\\Video=One\\result.srt");
    expect(result).not.toHaveProperty("noise");
    expect(resultQuality(result)).toBe("good");
  });

  it("validates supported paths and extracts Windows names", () => {
    expect(isSupportedVideo("C:\\Films\\sample.MP4")).toBe(true);
    expect(isSupportedVideo("C:\\Films\\sample.txt")).toBe(false);
    expect(basename("C:\\Films\\sample.mp4")).toBe("sample.mp4");
  });

  it("marks suspicious or failed outputs for review", () => {
    expect(resultQuality({ Pipeline: "COMPLETE", SuspiciousShortCues: "2", Error: "" }))
      .toBe("review");
    expect(resultQuality({ Pipeline: "FAILED", Error: "boom" })).toBe("error");
  });

  it("creates export names without duplicating the video extension", () => {
    expect(outputFilename("C:\\Films\\sample.cut.mp4", "ttml")).toBe("sample.cut.ttml");
    expect(outputFilename("video.mkv", "md")).toBe("video.md");
  });

  it("uses a cleaned result only when it is available and selected for translation", () => {
    expect(aiInputForMode("original", "cleaned", "translate", "cleaned")).toBe("cleaned");
    expect(aiInputForMode("original", "", "translate", "cleaned")).toBe("original");
    expect(aiInputForMode("original", "cleaned", "translate", "original")).toBe("original");
    expect(aiInputForMode("original", "cleaned", "clean", "cleaned")).toBe("original");
  });

  it("hands a completed pipeline SRT directly to AI as the original source", () => {
    expect(pipelineAiHandoff("1\n00:00:01,000 --> 00:00:02,000\nHello\n", "C:\\results\\film.srt", "C:\\films\\film.mp4"))
      .toEqual({ content: "1\n00:00:01,000 --> 00:00:02,000\nHello\n", name: "film.srt" });
    expect(pipelineAiHandoff("subtitle", "", "C:\\films\\fallback.mkv"))
      .toEqual({ content: "subtitle", name: "fallback.srt" });
  });

  it("exports the original SRT verbatim when SRT is selected", () => {
    const source = "1\r\n00:00:01,000 --> 00:00:02,000\r\nOCR text & punctuation\r\n";
    expect(originalSrtExport(source, "C:\\results\\film.srt")).toEqual({
      content: source,
      filename: "film.srt",
      format: "srt",
    });
    expect(originalSrtExport(source, "")).toEqual({ content: source, filename: "subtitles.srt", format: "srt" });
  });

  it("uses the selected export format for the original subtitle without replacing its source text", () => {
    const original = "1\n00:00:01,000 --> 00:00:02,000\nOriginal OCR text\n";
    for (const format of ["srt", "txt", "ttml", "md"] as const) {
      expect(originalSrtExport(original, "C:\\results\\film.cut.srt", format)).toEqual({
        content: original, filename: `film.cut.${format}`, format,
      });
    }
  });

  it("shows only installer-reported component download percentages", () => {
    expect(componentProgressPercent("Component download: 24% (12 MB of 50 MB)")).toBe(24);
    expect(componentProgressPercent("OCR model detector.onnx: 42.5% (x MB)")).toBe(42.5);
    expect(componentProgressPercent("Installing pinned native OCR CPU packages...")).toBeNull();
    expect(componentProgressPercent("pip progress 100% during setup")).toBeNull();
    expect(componentProgressPercent("Component download: 101% (x MB)")).toBeNull();
  });

});
