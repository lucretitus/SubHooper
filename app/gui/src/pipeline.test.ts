import { describe, expect, it } from "vitest";
import { aiInputForMode, basename, estimateSrtTextCharacters, isSupportedVideo, outputFilename, parseSummary, pipelineAiHandoff, resultQuality } from "./pipeline";

describe("pipeline helpers", () => {
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

  it("estimates only subtitle text sent to an online translator", () => {
    const srt = "1\n00:00:01,000 --> 00:00:02,000\nHello\nworld\n\n2\n00:00:03,000 --> 00:00:04,000\nİyi.\n";
    expect(estimateSrtTextCharacters(srt)).toBe(Array.from("Hello\nworldİyi.").length);
  });
});
