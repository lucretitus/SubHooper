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
export type RegionBox = { x: number; y: number; w: number; h: number };
export type RegionDragMode = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";
export type NormalizedRect = RegionBox;

export function updateRegionBox(initial: RegionBox, mode: RegionDragMode, dx: number, dy: number): RegionBox {
  const min = .08;
  if (mode === "move") return { ...initial, x: Math.min(1 - initial.w, Math.max(0, initial.x + dx)), y: Math.min(1 - initial.h, Math.max(0, initial.y + dy)) };
  let left = initial.x;
  let top = initial.y;
  let right = initial.x + initial.w;
  let bottom = initial.y + initial.h;
  if (mode.includes("w")) left = Math.min(right - min, Math.max(0, initial.x + dx));
  if (mode.includes("e")) right = Math.min(1, Math.max(left + min, initial.x + initial.w + dx));
  if (mode.includes("n")) top = Math.min(bottom - min, Math.max(0, initial.y + dy));
  if (mode.includes("s")) bottom = Math.min(1, Math.max(top + min, initial.y + initial.h + dy));
  return { x: left, y: top, w: right - left, h: bottom - top };
}

export function containedVideoFrame(containerWidth: number, containerHeight: number, videoWidth: number, videoHeight: number): NormalizedRect {
  if (containerWidth <= 0 || containerHeight <= 0 || videoWidth <= 0 || videoHeight <= 0) {
    return { x: 0, y: 0, w: 1, h: 1 };
  }
  const scale = Math.min(containerWidth / videoWidth, containerHeight / videoHeight);
  const w = videoWidth * scale / containerWidth;
  const h = videoHeight * scale / containerHeight;
  return { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
}

export function regionBoxInFrame(region: RegionBox, frame: NormalizedRect): NormalizedRect {
  return {
    x: frame.x + region.x * frame.w,
    y: frame.y + region.y * frame.h,
    w: region.w * frame.w,
    h: region.h * frame.h,
  };
}

export type PipelineAiHandoff = {
  content: string;
  name: string;
};

export type OriginalSrtExport = {
  content: string;
  filename: string;
  format: ExportFormat;
};

export function isPipelineCancellation(reason: unknown): boolean {
  const message = reason instanceof Error ? reason.message : String(reason);
  return message.trim() === "Processing was cancelled.";
}

export function originalSrtExport(content: string, sourceName: string, format: ExportFormat = "srt"): OriginalSrtExport {
  const name = basename(sourceName) || "subtitles.srt";
  return {
    content,
    filename: `${name.replace(/\.(srt|ttml|txt|md)$/i, "")}.${format}`,
    format,
  };
}

export function componentProgressPercent(message: string): number | null {
  const match = message.match(/^(?:Component download:|OCR model .+:)\s*(\d{1,3}(?:\.\d+)?)\s*%/);
  if (!match) return null;
  const percent = Number(match[1]);
  return percent >= 0 && percent <= 100 ? percent : null;
}

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
