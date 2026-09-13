export type Summary = Record<string, string>;

export type PipelineStage = {
  key: "idle" | "prepare" | "vsf" | "ocr" | "finalize" | "complete" | "cancelled" | "failed";
  label: string;
  percent: number | null;
};

export const IDLE_STAGE: PipelineStage = {
  key: "idle",
  label: "Ready to start",
  percent: null,
};

export type ExportFormat = "srt" | "ttml" | "txt" | "md";
export type AiMode = "clean" | "translate";
export type AiTranslateSource = "original" | "cleaned";

export type PipelineAiHandoff = {
  content: string;
  name: string;
};

export function pipelineAiHandoff(srtText: string, srtPath: string, videoPath: string): PipelineAiHandoff {
  return {
    content: srtText,
    name: basename(srtPath) || outputFilename(videoPath, "srt"),
  };
}

export function aiInputForMode(
  original: string,
  cleaned: string,
  mode: AiMode,
  translateSource: AiTranslateSource,
): string {
  return mode === "translate" && translateSource === "cleaned" && cleaned
    ? cleaned
    : original;
}

export function estimateSrtTextCharacters(content: string): number {
  let total = 0;
  for (const block of content.replaceAll("\r\n", "\n").replaceAll("\r", "\n").trim().split(/\n{2,}/)) {
    const lines = block.split("\n");
    const timing = lines.findIndex((line) => line.includes("-->"));
    if (timing < 0) continue;
    const text = lines.slice(timing + 1).join("\n").trim();
    total += Array.from(text).length;
  }
  return total;
}

export function outputFilename(videoPath: string, format: ExportFormat): string {
  const stem = basename(videoPath).replace(/\.[^.]+$/, "") || "subtitles";
  return `${stem}.${format}`;
}

export function parseSummary(text: string): Summary {
  const summary: Summary = {};
  let inside = false;
  for (const rawLine of text.replaceAll("\0", "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line.includes("SUBHOOPER PIPELINE") && line.endsWith("RESULT START ---")) {
      inside = true;
      continue;
    }
    if (inside && line.includes("RESULT END ---")) break;
    if (!inside) continue;
    const separator = line.indexOf("=");
    if (separator > 0) summary[line.slice(0, separator)] = line.slice(separator + 1);
  }
  return summary;
}

export function isSupportedVideo(path: string): boolean {
  return /\.(mp4|mkv|avi|mov|webm|ts|m2ts|wmv|m4v)$/i.test(path);
}

export function basename(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}

export function resultQuality(summary: Summary): "good" | "review" | "error" {
  if (summary.Pipeline !== "COMPLETE" || summary.Error) return "error";
  return Number(summary.SuspiciousShortCues || 0) > 0 ? "review" : "good";
}
