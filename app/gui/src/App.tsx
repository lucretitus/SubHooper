import { PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { aiInputForMode, AiMode, AiTranslateSource, basename, componentProgressPercent, containedVideoFrame, ExportFormat, IDLE_STAGE, isPipelineCancellation, isSupportedVideo, NormalizedRect, originalSrtExport, outputFilename, pipelineAiHandoff, PipelineStage, regionBoxInFrame, RegionBox, RegionDragMode, Summary, updateRegionBox } from "./pipeline";

type RegionPreset = "Bottom" | "Top" | "Full" | "Custom";
type DragMode = RegionDragMode;
type PipelineResult = { summary: Summary; srtText: string; srtPath: string; resultDir: string };
type Settings = { defaultDirectory: string; defaultFormat: ExportFormat };
type AiModel = { id: string; name: string; sizeLabel: string; memoryLabel: string; recommendation: string; minimumGpuMib: number; installed: boolean; modelCached?: boolean; gpuEligible: boolean; gpuReason: string };
type AiProgress = { phase: string; label: string; received: number; total: number | null };
type AiResult = { srtText: string; modelName: string; cueCount: number; sourceCueCount: number; droppedCueCount: number };
type AiComputeBackend = "cuda" | "cpu";
type OcrComputeMode = "mixed" | "cpu";
type AiCleanupStrength = "light" | "balanced" | "strong";
type ComponentStatus = { ocrRuntime: boolean; gpuRuntime: boolean; gpuError: string | null; ready: boolean; installRoot: string };
type AvailableUpdate = Awaited<ReturnType<typeof check>>;

const VIDEO_FILTER = [{ name: "Video files", extensions: ["mp4", "mkv", "avi", "mov", "webm", "ts", "m2ts", "wmv", "m4v"] }];
const SUBTITLE_FILTER = [{ name: "SRT subtitles", extensions: ["srt"] }];
const AI_MODEL_FILTER = [{ name: "GGUF models", extensions: ["gguf"] }];
const PRESETS: Record<Exclude<RegionPreset, "Custom">, RegionBox> = {
  Bottom: { x: .03, y: .58, w: .94, h: .40 },
  Top: { x: .03, y: .02, w: .94, h: .40 },
  Full: { x: 0, y: 0, w: 1, h: 1 },
};
const PRESET_LABELS: Record<Exclude<RegionPreset, "Custom">, string> = {
  Bottom: "Bottom",
  Top: "Top",
  Full: "Fullscreen",
};
const FORMAT_LABELS: Record<ExportFormat, string> = { srt: "SRT", ttml: "TTML", txt: "TXT", md: "MD" };
const STEP_LABELS = ["Insert", "Subtitles", "Results"];
const QWEN_MODEL_ORDER = ["qwen3-4b-q4km", "qwen3-8b-q4km", "qwen3-14b-q4km"];
const AI_LANGUAGES = ["English", "Turkish", "German", "French", "Spanish", "Italian", "Portuguese", "Arabic", "Japanese", "Korean", "Chinese"];
const CLEANUP_STRENGTH_LABELS: Record<AiCleanupStrength, string> = { light: "Light", balanced: "Balanced", strong: "Strong" };
const CLEANUP_STRENGTH_HELP: Record<AiCleanupStrength, string> = {
  light: "Fix clear text errors and keep uncertain fragments for review.",
  balanced: "Normalize the language and remove only clear OCR noise.",
  strong: "Aggressively remove symbol clusters, broken fragments, and unrelated text.",
};
type IconName = "video" | "folder" | "cpu" | "gpu" | "spark" | "download" | "settings" | "check" | "chevron" | "edit";

function Icon({ name }: { name: IconName }) {
  const paths = {
    video: <><rect x="3.5" y="5.5" width="17" height="13" rx="2" /><path d="M8 5.5v13m8-13v13M3.5 9h4.5m8 0h4.5M3.5 15h4.5m8 0h4.5" /></>,
    folder: <><path d="M3.5 7.5h6l2-2h9v13h-17z" /></>,
    cpu: <><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9.5 1v4m5-4v4m-5 14v4m5-4v4M1 9.5h4m-4 5h4m14-5h4m-4 5h4M10 10h4v4h-4z" /></>,
    gpu: <><rect x="3" y="6" width="18" height="12" rx="2"/><circle cx="9" cy="12" r="3"/><path d="M14.5 10h3m-3 4h3M7 3v3m4-3v3" /></>,
    spark: <><path d="m11.5 2.6 1.8 5.2 5.2 1.8-5.2 1.8-1.8 5.2-1.8-5.2-5.2-1.8 5.2-1.8 1.8-5.2Z" /><path d="m18.5 14.2 .9 2.6 2.6 .9-2.6 .9-.9 2.6-.9-2.6-2.6-.9 2.6-.9 .9-2.6Z" /></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4M4 19h16" /></>,
    settings: <><path d="M19.43 12.98c.04-.32.07-.65.07-.98s-.02-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.37-.31-.6-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98L14.5 2.42A.488.488 0 0 0 14 2h-4c-.25 0-.46.18-.5.42L9.12 5.07c-.61.25-1.18.59-1.69.98l-2.49-1a.485.485 0 0 0-.6.22l-2 3.46a.49.49 0 0 0 .12.64l2.11 1.65c-.04.32-.08.66-.08.98s.03.66.08.98l-2.11 1.65a.49.49 0 0 0-.12.64l2 3.46c.12.22.37.31.6.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.04.24.25.42.5.42h4c.25 0 .46-.18.5-.42l.38-2.65c.61-.25 1.18-.58 1.69-.98l2.49 1c.23.08.48 0 .6-.22l2-3.46a.49.49 0 0 0-.12-.64l-2.11-1.65ZM12 15.5A3.5 3.5 0 1 1 12 8a3.5 3.5 0 0 1 0 7.5Z" /></>,
    check: <><path d="m5 12.5 4.2 4.2L19 7" /></>,
    chevron: <><path d="m8 10 4 4 4-4" /></>,
    edit: <><path d="m4 16-.8 4 4-.8L18 8.4 14.6 5zM13 6.5l3.5 3.5" /></>,
  };
  return <svg className={`icon-${name}`} viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function joinPath(directory: string, filename: string) { return directory ? directory.replace(/[\\\/]$/, "") + "\\" + filename : filename; }
function secondsLabel(seconds: number) { return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`; }
export function defaultOcrComputeMode(recommendedMode: string | undefined): OcrComputeMode {
  // A previous GUI recommendation/selection may still say "cuda".
  return recommendedMode === "mixed" || recommendedMode === "cuda" ? "mixed" : "cpu";
}

export default function App() {
  const [videoPath, setVideoPath] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const [preset, setPreset] = useState<RegionPreset>("Bottom");
  const [regionBox, setRegionBox] = useState<RegionBox>(PRESETS.Bottom);
  const [videoFrame, setVideoFrame] = useState<NormalizedRect>({ x: 0, y: 0, w: 1, h: 1 });
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [ocrComputeMode, setOcrComputeMode] = useState<OcrComputeMode>("cpu");
  const [ocrRecommendedMode, setOcrRecommendedMode] = useState<OcrComputeMode>("cpu");
  const [busy, setBusy] = useState(false);
  const [draggingFile, setDraggingFile] = useState(false);
  const [dragState, setDragState] = useState<{ mode: DragMode; x: number; y: number; box: RegionBox; width: number; height: number } | null>(null);
  const [stage, setStage] = useState<PipelineStage>(IDLE_STAGE);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState("");
  const [settings, setSettings] = useState<Settings>(() => ({ defaultDirectory: localStorage.getItem("defaultDirectory") || "", defaultFormat: (localStorage.getItem("defaultFormat") as ExportFormat) || "srt" }));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [componentStatus, setComponentStatus] = useState<ComponentStatus | null>(null);
  const [componentConsent, setComponentConsent] = useState(false);
  const [componentBusy, setComponentBusy] = useState(false);
  const [componentError, setComponentError] = useState("");
  const [componentProgressMessage, setComponentProgressMessage] = useState("");
  const [updateBusy, setUpdateBusy] = useState(false);
  const [updateMessage, setUpdateMessage] = useState("Not checked");
  const [availableUpdate, setAvailableUpdate] = useState<AvailableUpdate>(null);
  const [elapsed, setElapsed] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [aiModels, setAiModels] = useState<AiModel[]>([]);
  const [aiModelId, setAiModelId] = useState("qwen3-8b-q4km");
  const [aiComputeBackend, setAiComputeBackend] = useState<AiComputeBackend>("cuda");
  const [aiComputeBackendReady, setAiComputeBackendReady] = useState(false);
  const [aiSource, setAiSource] = useState("");
  const [aiSourceName, setAiSourceName] = useState("");
  const [aiOutput, setAiOutput] = useState("");
  const [aiCleanedSource, setAiCleanedSource] = useState("");
  const [aiMode, setAiMode] = useState<AiMode>("clean");
  const [aiTranslateSource, setAiTranslateSource] = useState<AiTranslateSource>("original");
  const [aiCleanupStrength, setAiCleanupStrength] = useState<AiCleanupStrength>("balanced");
  const [aiLanguage, setAiLanguage] = useState("English");
  const [aiFormat, setAiFormat] = useState<ExportFormat>("srt");
  const [aiBusy, setAiBusy] = useState(false);
  const [aiCancelling, setAiCancelling] = useState(false);
  const aiCancellationRef = useRef(false);
  const [aiProgress, setAiProgress] = useState<AiProgress | null>(null);
  const [aiError, setAiError] = useState("");
  const [aiSavedPath, setAiSavedPath] = useState("");
  const [aiResultMeta, setAiResultMeta] = useState<AiResult | null>(null);
  const [aiResultMode, setAiResultMode] = useState<AiMode | null>(null);
  const [aiResultLanguage, setAiResultLanguage] = useState("");
  const [aiResultSource, setAiResultSource] = useState<AiTranslateSource | null>(null);
  const [aiResultStrength, setAiResultStrength] = useState<AiCleanupStrength | null>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let disposed = false;
    const cleanups: Array<() => void> = [];
    Promise.all([
      listen<string>("pipeline-output", ({ payload }) => setLogs((current) => [...current.slice(-399), payload])),
      listen<PipelineStage>("pipeline-stage", ({ payload }) => setStage(payload)),
      getCurrentWebview().onDragDropEvent(({ payload }) => {
        if (payload.type === "over") setDraggingFile(true);
        if (payload.type === "leave") setDraggingFile(false);
        if (payload.type === "drop") {
          setDraggingFile(false);
          const candidate = payload.paths.find(isSupportedVideo);
          if (candidate && !busy) selectVideo(candidate);
        }
      }),
    ]).then((unlisteners) => disposed ? unlisteners.forEach((unlisten) => unlisten()) : cleanups.push(...unlisteners)).catch((reason) => setError(String(reason)));
    return () => { disposed = true; cleanups.forEach((unlisten) => unlisten()); };
  }, [busy]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    listen<AiProgress>("ai-progress", ({ payload }) => setAiProgress(payload))
      .then((cleanup) => { if (disposed) cleanup(); else unlisten = cleanup; })
      .catch((reason) => setAiError(String(reason)));
    invoke<AiModel[]>("ai_catalog")
      .then((models) => { if (!disposed) {
        const ordered = [...models].sort((left, right) => QWEN_MODEL_ORDER.indexOf(left.id) - QWEN_MODEL_ORDER.indexOf(right.id));
        setAiModels(ordered);
        const recommended = [...ordered].reverse().find((model) => model.gpuEligible) || ordered[0];
        setAiModelId(recommended?.id || QWEN_MODEL_ORDER[1]);
      } })
      .catch((reason) => { if (!disposed) setAiError(String(reason)); });
    invoke<ComponentStatus>("component_status")
      .then((status) => { if (!disposed) setComponentStatus(status); })
      .catch((reason) => { if (!disposed) setComponentError(String(reason)); });
    invoke<{ recommendedMode: string; reason: string }>("ocr_hardware_recommendation")
      .then((recommendation) => { if (!disposed) { const mode = defaultOcrComputeMode(recommendation.recommendedMode); setOcrComputeMode(mode); setOcrRecommendedMode(mode); } })
      .catch(() => { if (!disposed) { setOcrComputeMode("cpu"); setOcrRecommendedMode("cpu"); } });
    return () => { disposed = true; unlisten?.(); };
  }, []);

  useEffect(() => {
    const preview = previewRef.current;
    if (!preview) return;
    const update = () => {
      setVideoFrame(containedVideoFrame(preview.clientWidth, preview.clientHeight,
        videoDimensions.width, videoDimensions.height));
    };
    const observer = new ResizeObserver(update);
    observer.observe(preview);
    update();
    return () => observer.disconnect();
  }, [previewUrl, videoDimensions]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    listen<string>("component-progress", ({ payload }) => {
      const message = payload.trim();
      if (message) setComponentProgressMessage(message);
    })
      .then((cleanup) => { if (disposed) cleanup(); else unlisten = cleanup; })
      .catch((reason) => { if (!disposed) setComponentError(String(reason)); });
    return () => { disposed = true; unlisten?.(); };
  }, []);

  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    if (!dragState) return;
    const move = (event: PointerEvent) => {
      setRegionBox(updateRegionBox(dragState.box, dragState.mode, (event.clientX - dragState.x) / dragState.width, (event.clientY - dragState.y) / dragState.height));
      setPreset("Custom");
    };
    const stop = () => setDragState(null);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
    return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", stop); };
  }, [dragState]);

  const lastLog = logs.at(-1) || "";
  const currentLog = /^OCRProgress=\d+$/i.test(lastLog) ? stage.label : lastLog || stage.label;
  const displayedRegion = regionBoxInFrame(regionBox, videoFrame);

  function selectVideo(path: string) {
    if (!isSupportedVideo(path)) { setError("Select a supported video file."); return; }
    setVideoPath(path);
    setPreviewUrl(convertFileSrc(path));
    setVideoFrame({ x: 0, y: 0, w: 1, h: 1 });
    setVideoDimensions({ width: 0, height: 0 });
    setDuration(0); setCurrentTime(0); setPlaying(false); setResult(null); setError(""); setLogs([]); setStage(IDLE_STAGE);
  }

  async function chooseVideo() {
    const selected = await open({ multiple: false, directory: false, filters: VIDEO_FILTER });
    if (typeof selected === "string") selectVideo(selected);
  }

  function choosePreset(value: Exclude<RegionPreset, "Custom">) { setRegionBox(PRESETS[value]); setPreset(value); }

  function beginRegionDrag(event: ReactPointerEvent, mode: DragMode) {
    if (busy || !previewRef.current) return;
    event.preventDefault(); event.stopPropagation();
    const bounds = previewRef.current;
    setDragState({ mode, x: event.clientX, y: event.clientY, box: regionBox,
      width: bounds.clientWidth * videoFrame.w, height: bounds.clientHeight * videoFrame.h });
  }

  async function togglePreview() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) await video.play(); else video.pause();
  }

  function seekPreview(value: number) { if (videoRef.current) videoRef.current.currentTime = value; setCurrentTime(value); }

  async function start() {
    if (!videoPath || busy) return;
    try {
      const status = await invoke<ComponentStatus>("component_status");
      setComponentStatus(status);
      if (!status.ready) {
        setComponentError("Video extraction components must be installed before processing.");
        setSettingsOpen(true);
        return;
      }
      if (ocrComputeMode === "mixed" && !status.gpuRuntime) {
        setComponentError(`GPU + CPU components need verification. Select Verify Components, then retry. ${status.gpuError || ""}`);
        setSettingsOpen(true);
        return;
      }
    } catch (reason) {
      setComponentError(String(reason));
      setSettingsOpen(true);
      return;
    }
    await releaseAiRuntime();
    setBusy(true); setResult(null); setError(""); setLogs([]); setStage({ key: "prepare", label: "Preparing video", percent: null });
    const namedRegion = preset === "Top" || preset === "Custom" ? "Custom" : preset;
    try {
      const response = await invoke<PipelineResult>("start_pipeline", { request: {
        videoPath, region: namedRegion, regionTop: 1 - regionBox.y, regionBottom: 1 - regionBox.y - regionBox.h, regionLeft: regionBox.x, regionRight: regionBox.x + regionBox.w,
        compute: ocrComputeMode === "cpu" ? "cpu" : "auto", ocrCompute: ocrComputeMode,
      } });
      const aiHandoff = pipelineAiHandoff(response.srtText, response.srtPath, videoPath);
      setResult(response); setStage({ key: "complete", label: "Subtitles ready", percent: 100 });
      setAiSource(aiHandoff.content); setAiSourceName(aiHandoff.name); setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); setAiError(""); setAiSavedPath(""); setAiMode("clean"); setActiveStep(2);
    } catch (reason) {
      const message = String(reason);
      if (!isPipelineCancellation(reason)) { setError(message); setStage({ key: "failed", label: "Processing failed", percent: null }); }
    } finally { setBusy(false); }
  }

  async function cancel() {
    if (!busy) return;
    try { await invoke("cancel_pipeline"); setStage({ key: "cancelled", label: "Processing cancelled", percent: null }); } catch (reason) { setError(String(reason)); }
  }

  async function chooseDefaultDirectory() {
    const selected = await open({ multiple: false, directory: true });
    if (typeof selected === "string") setSettings((current) => ({ ...current, defaultDirectory: selected }));
  }

  function storeSettings() {
    localStorage.setItem("defaultDirectory", settings.defaultDirectory); localStorage.setItem("defaultFormat", settings.defaultFormat); setSettingsOpen(false);
  }

  async function installRequiredComponents() {
    if (!componentConsent || componentBusy) return;
    setComponentBusy(true); setComponentError(""); setComponentProgressMessage("Preparing Python 3.13 runtime, OCR packages, and verified PP-OCRv6 models…");
    try {
      const status = await invoke<ComponentStatus>("install_components");
      setComponentStatus(status);
      if (ocrComputeMode === "mixed" && !status.gpuRuntime) {
        setComponentError(`CPU components are ready, but GPU + CPU validation failed. ${status.gpuError || ""}`);
      }
    } catch (reason) {
      setComponentError(String(reason));
      setComponentProgressMessage("Component setup stopped. See the error details below.");
    } finally {
      setComponentBusy(false);
    }
  }

  async function checkForUpdates() {
    if (updateBusy) return;
    setUpdateBusy(true); setUpdateMessage("Checking GitHub Releases...");
    try {
      const update = await check();
      setAvailableUpdate(update);
      setUpdateMessage(update ? `Version ${update.version} is available` : "SubHooper is up to date");
    } catch (reason) {
      setUpdateMessage(`Update check failed: ${String(reason)}`);
    } finally {
      setUpdateBusy(false);
    }
  }

  async function installAvailableUpdate() {
    if (!availableUpdate || updateBusy) return;
    setUpdateBusy(true); setUpdateMessage(`Installing version ${availableUpdate.version}...`);
    try {
      await availableUpdate.downloadAndInstall();
      await relaunch();
    } catch (reason) {
      setUpdateMessage(`Update failed: ${String(reason)}`);
      setUpdateBusy(false);
    }
  }

  async function chooseAiSubtitle() {
    const selected = await open({ multiple: false, directory: false, filters: SUBTITLE_FILTER });
    if (typeof selected !== "string") return;
    try {
      const content = await invoke<string>("read_subtitle_file", { path: selected });
      setAiSource(content); setAiSourceName(basename(selected)); setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); setAiError(""); setAiSavedPath("");
    } catch (reason) { setAiError(String(reason)); }
  }

  async function releaseAiRuntime() {
    await invoke("release_ai_runtime");
    setAiComputeBackendReady(false);
  }

  async function importAiModel(modelId: string) {
    const selected = await open({ multiple: false, directory: false, filters: AI_MODEL_FILTER });
    if (typeof selected !== "string") return;
    try {
      await releaseAiRuntime();
      await invoke("import_ai_model", { modelId, path: selected });
      const models = await invoke<AiModel[]>("ai_catalog");
      const ordered = [...models].sort((left, right) => QWEN_MODEL_ORDER.indexOf(left.id) - QWEN_MODEL_ORDER.indexOf(right.id));
      setAiModels(ordered);
      setAiModelId(modelId);
      setAiError("");
    } catch (reason) { setAiError(String(reason)); }
  }

  async function selectAiModel(modelId: string) {
    if (modelId === aiModelId) return;
    await releaseAiRuntime();
    setAiModelId(modelId);
  }

  async function selectAiBackend(backend: AiComputeBackend) {
    if (backend === aiComputeBackend) return;
    await releaseAiRuntime();
    setAiComputeBackend(backend);
    setAiComputeBackendReady(false);
    setAiError("");
    const recommended = [...aiModels].reverse().find((model) => backend === "cpu" || model.gpuEligible);
    if (recommended) setAiModelId(recommended.id);
  }

  async function downloadAiModel(modelId: string, continueWithProcessing = false) {
    if (aiBusy) return;
    aiCancellationRef.current = false;
    setAiBusy(true); setAiCancelling(false); setAiModelId(modelId); setAiError(""); setAiProgress({ phase: "starting", label: `Preparing ${aiComputeBackend === "cuda" ? "GPU (CUDA)" : "CPU"} engine`, received: 0, total: null });
    try {
      const installed = await invoke<AiModel>("download_ai_model", { modelId, computeBackend: aiComputeBackend });
      if (aiCancellationRef.current) throw "AI processing was cancelled.";
      setAiModels((models) => models.map((model) => model.id === installed.id ? installed : model));
      setAiComputeBackendReady(true);
      if (continueWithProcessing) await runAiProcessing(true);
    } catch (reason) { setAiError(String(reason)); setAiProgress(null); }
    finally { setAiBusy(false); setAiCancelling(false); }
  }

  async function runAiProcessing(afterPreparation = false) {
    if (!aiSource || aiBusy && !afterPreparation) return;
    if (afterPreparation && aiCancellationRef.current) return;
    if (!afterPreparation) aiCancellationRef.current = false;
    const input = aiInputForMode(aiSource, aiCleanedSource, aiMode, aiTranslateSource);
    setAiBusy(true); setAiCancelling(false); setAiSavedPath(""); setAiError("");
    setAiProgress({ phase: "starting", label: "Starting local AI", received: 0, total: null });
    try {
      const response = await invoke<AiResult>("run_ai_cleaning", { request: {
        content: input, modelId: aiModelId, computeBackend: aiComputeBackend, mode: aiMode, cleanupLanguage: aiMode === "clean" ? "Auto detect" : null, cleanupStrength: aiMode === "clean" ? aiCleanupStrength : null, sourceLanguage: aiMode === "translate" ? "Auto detect" : null, targetLanguage: aiMode === "translate" ? aiLanguage : null,
        guidance: null,
      } });
      setAiOutput(response.srtText); setAiResultMeta(response);
      setAiResultMode(aiMode); setAiResultLanguage(aiMode === "translate" ? aiLanguage : "");
      setAiResultSource(aiMode === "translate" ? (aiCleanedSource ? aiTranslateSource : "original") : null);
      setAiResultStrength(aiMode === "clean" ? aiCleanupStrength : null);
      if (aiMode === "clean") { setAiCleanedSource(response.srtText); setAiTranslateSource("original"); }
      setAiProgress({ phase: "complete", label: response.droppedCueCount ? `AI result ready · ${response.droppedCueCount} noise cues removed` : "AI result ready", received: 1, total: 1 });
    } catch (reason) { setAiError(String(reason)); setAiProgress(null); }
    finally { setAiBusy(false); setAiCancelling(false); }
  }

  async function runAiAction() {
    if (!aiSource) {
      await chooseAiSubtitle();
      return;
    }
    if (!selectedAiModel) {
      setAiError("Select a local model.");
      return;
    }
    if (aiComputeBackend === "cuda" && !selectedAiModel.gpuEligible) {
      setAiError(selectedAiModel.gpuReason || "This model is unavailable on the selected GPU. Choose CPU or another model.");
      return;
    }
    if (!selectedAiModel.installed || !aiComputeBackendReady) {
      await downloadAiModel(selectedAiModel.id, true);
      return;
    }
    await runAiProcessing();
  }

  async function cancelAi() {
    if (!aiBusy || aiCancellationRef.current) return;
    aiCancellationRef.current = true;
    setAiCancelling(true);
    try { await invoke("cancel_ai"); } catch (reason) { setAiError(String(reason)); setAiCancelling(false); }
  }

  async function exportAiResult() {
    if (!aiOutput) return;
    const stem = (aiSourceName || "subtitles.srt").replace(/\.[^.]+$/, "");
    const suffix = aiResultMode === "translate" ? `-${(aiResultLanguage || aiLanguage).toLowerCase()}-ai` : "-cleaned-ai";
    const selected = await save({ defaultPath: joinPath(settings.defaultDirectory, `${stem}${suffix}.${aiFormat}`), filters: [{ name: FORMAT_LABELS[aiFormat], extensions: [aiFormat] }] });
    if (!selected) return;
    const destination = selected.toLowerCase().endsWith(`.${aiFormat}`) ? selected : `${selected}.${aiFormat}`;
    try { setAiSavedPath(await invoke<string>("save_export", { destination, content: aiOutput, format: aiFormat })); setAiError(""); } catch (reason) { setAiError(String(reason)); }
  }

  async function exportOriginalSrt() {
    if (!aiSource || aiBusy) return;
    const source = originalSrtExport(aiSource, aiSourceName, aiFormat);
    const selected = await save({ defaultPath: joinPath(settings.defaultDirectory, source.filename), filters: [{ name: FORMAT_LABELS[source.format], extensions: [source.format] }] });
    if (!selected) return;
    const destination = selected.toLowerCase().endsWith(`.${source.format}`) ? selected : `${selected}.${source.format}`;
    try { setAiSavedPath(await invoke<string>("save_export", { destination, content: source.content, format: source.format })); setAiError(""); } catch (reason) { setAiError(String(reason)); }
  }

  const selectedAiModel = aiModels.find((model) => model.id === aiModelId);
  const recommendedAiModelId = [...aiModels].reverse().find((model) => aiComputeBackend === "cpu" || model.gpuEligible)?.id;
  const selectedModelUnavailableOnGpu = aiComputeBackend === "cuda" && Boolean(selectedAiModel && !selectedAiModel.gpuEligible);
  const aiPercent = aiProgress?.phase === "complete" ? aiBusy ? 99 : 100 : aiProgress?.total ? Math.min(99, Math.round(aiProgress.received / aiProgress.total * 100)) : null;
  const componentPercent = componentProgressPercent(componentProgressMessage);
  const aiActionLabel = !aiSource
    ? "Open SRT to continue"
    : aiMode === "clean" ? "Clean subtitles" : `Translate to ${aiLanguage}`;
  const aiActionStatus = !aiSource
    ? "An SRT subtitle is required before processing"
    : selectedModelUnavailableOnGpu
      ? selectedAiModel?.gpuReason || "This model exceeds the available GPU memory. Choose CPU or another model."
      : selectedAiModel?.installed
      ? aiMode === "translate" && aiCleanedSource
        ? `Will translate the ${aiTranslateSource === "cleaned" ? "cleaned result" : "original SRT"}`
        : aiMode === "clean"
          ? `${CLEANUP_STRENGTH_LABELS[aiCleanupStrength]} cleanup runs locally on this PC`
          : "Runs locally on this PC"
      : `First use downloads the verified ${selectedAiModel?.name || "local model"} and ${aiComputeBackend === "cuda" ? "NVIDIA CUDA" : "CPU"} engine, then starts processing`;

  const canOpenStep = (index: number) => index === 0 || index === 2 || index === 1 && Boolean(videoPath);

  return <main className="app-shell">
    <header className="topbar">
      <img className="brand-logo" src="/subhooper-logo.svg" alt="SubHooper" />
      <nav className="step-tabs" aria-label="Workflow steps">{STEP_LABELS.map((label, index) => <button type="button" className={`${index === activeStep ? "active" : ""} ${index < activeStep || index === 1 && Boolean(result) ? "completed" : ""}`} disabled={!canOpenStep(index) || busy || aiBusy} onClick={() => setActiveStep(index)} key={label}><span>{index + 1}</span>{label}</button>)}</nav>
      <button className="icon-button" type="button" aria-label="Settings" onClick={() => setSettingsOpen(true)}><Icon name="settings" /></button>
    </header>

    <section className="page-frame">
      {activeStep === 0 && <section className="source-page page-enter">
        <button className={`drop-zone ${draggingFile ? "dragging" : ""}`} type="button" disabled={busy} onClick={chooseVideo}>
          <span className="drop-icon"><Icon name="folder" /></span>
          <span className="drop-copy"><strong>{videoPath ? basename(videoPath) : "Upload a Video File"}</strong><span>{videoPath ? "Ready to continue" : "Drag and drop a video here, or click to browse"}</span></span>
          {videoPath && <small>{videoPath}</small>}
        </button>
        <button className="page-action" type="button" disabled={!videoPath} onClick={() => setActiveStep(1)}>Next Step</button>
      </section>}

      {activeStep === 1 && <section className="region-page page-enter">
        <div className="region-card">
          <div className="preview-column">
            <div className="video-preview" ref={previewRef}>
              {previewUrl ? <video ref={videoRef} src={previewUrl} preload="metadata" onLoadedMetadata={(event) => { setDuration(event.currentTarget.duration || 0); setVideoDimensions({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight }); }} onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} /> : <div className="video-fallback"><Icon name="video" /></div>}
              <div className="selection-box" style={{ left: `${displayedRegion.x * 100}%`, top: `${displayedRegion.y * 100}%`, width: `${displayedRegion.w * 100}%`, height: `${displayedRegion.h * 100}%` }} onPointerDown={(event) => beginRegionDrag(event, "move")}>{(["n", "s", "e", "w", "ne", "nw", "se", "sw"] as DragMode[]).map((mode) => <i className={`handle ${mode}`} onPointerDown={(event) => beginRegionDrag(event, mode)} key={mode} />)}</div>
            </div>
            <div className="video-controls"><button type="button" onClick={togglePreview} aria-label={playing ? "Pause" : "Play"}>{playing ? "Ⅱ" : "▶"}</button><input type="range" min="0" max={duration || 0} step="0.05" value={Math.min(currentTime, duration || 0)} onChange={(event) => seekPreview(Number(event.target.value))} /><time>{secondsLabel(Math.floor(currentTime))}</time></div>
          </div>
          <div className="preset-row">{(Object.keys(PRESETS) as Array<Exclude<RegionPreset, "Custom">>).map((value) => <button type="button" className={preset === value ? "selected" : ""} onClick={() => choosePreset(value)} key={value}><span className={`preset-icon preset-${value.toLowerCase()}`} />{PRESET_LABELS[value]}</button>)}</div>
          <div className="compute-options">
            <div className="compute-row" role="group" aria-label="OCR processing mode">
              {(["mixed", "cpu"] as OcrComputeMode[]).map((mode) => <button type="button" className={ocrComputeMode === mode ? "selected" : ""} aria-pressed={ocrComputeMode === mode} disabled={busy} onClick={() => setOcrComputeMode(mode)} key={mode}><strong>{mode === "mixed" ? "GPU + CPU" : "CPU"}</strong>{ocrRecommendedMode === mode && <small className="compute-recommended">Recommended</small>}</button>)}
            </div>
          </div>
        </div>
        {error && <div className="error-card"><strong>Processing failed</strong><p>{error}</p></div>}
        <div className="extract-row"><button className={`extract-action ${busy ? "running" : ""}`} type="button" disabled={!videoPath || busy} onClick={start}><i className={`extract-progress ${busy && stage.percent === null ? "indeterminate" : ""}`} style={stage.percent === null ? undefined : { width: `${stage.percent}%` }} /><i className="extract-sweep" />{busy ? <><span className="extract-spinner" /><span className="extract-copy"><strong>{stage.label} · {secondsLabel(elapsed)}</strong><small title={currentLog}>{currentLog}</small></span><b className={stage.percent === null ? "stage-activity-label" : ""}>{stage.percent === null ? "In progress" : `${stage.percent}%`}</b></> : <span className="extract-label">Extract Subtitles</span>}</button>{busy && <button className="cancel-action" type="button" onClick={cancel}>Cancel</button>}</div>
      </section>}

      {activeStep === 2 && <section className="ai-page page-enter">
        <div className="ai-controls">
          <div className="ai-mode-control" role="group" aria-label="AI task">
            <button type="button" className={aiMode === "clean" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiMode("clean")}>Clean</button>
            <button type="button" className={aiMode === "translate" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiMode("translate")}>Translate</button>
            {aiMode === "translate" && <label><span>Language</span><select value={aiLanguage} disabled={aiBusy} onChange={(event) => setAiLanguage(event.target.value)}>{AI_LANGUAGES.map((language) => <option key={language}>{language}</option>)}</select></label>}
          </div>
            <div className="ai-source-control">
            <button className="ai-source-pick" type="button" onClick={chooseAiSubtitle} disabled={aiBusy}><Icon name="folder" /><span><strong>{aiSourceName || "Open an SRT Subtitle"}</strong><small>{aiSource ? "Ready" : "Choose Source"}</small></span></button>
          </div>
          {aiMode === "clean" ? <fieldset className="ai-clean-strength"><legend>Cleanup strength</legend><span>Strength</span>{(Object.keys(CLEANUP_STRENGTH_LABELS) as AiCleanupStrength[]).map((strength) => <button type="button" key={strength} className={aiCleanupStrength === strength ? "selected" : ""} disabled={aiBusy} aria-pressed={aiCleanupStrength === strength} title={CLEANUP_STRENGTH_HELP[strength]} onClick={() => setAiCleanupStrength(strength)}>{CLEANUP_STRENGTH_LABELS[strength]}</button>)}</fieldset> : <div className="ai-clean-strength ai-translate-from">{aiCleanedSource ? <><span>Translate from</span><button type="button" className={aiTranslateSource === "original" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiTranslateSource("original")}>Original</button><button type="button" className={aiTranslateSource === "cleaned" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiTranslateSource("cleaned")}>Cleaned</button></> : <span>Translate the original SRT</span>}</div>}
        </div>
        <div className="ai-workspace">
          <section className="ai-editors" aria-label="Subtitle editors">
            <div className="ai-editor"><header><span>Original SRT</span><small>{aiSourceName || "No file selected"}</small></header><textarea value={aiSource} onChange={(event) => { setAiSource(event.target.value); setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); }} placeholder="Open an SRT file to begin." spellCheck={false} /></div>
            <div className="ai-editor output"><header><span>AI Result</span><small>{aiResultMeta ? `${aiResultMode === "clean" ? `Cleaned · ${aiResultStrength ? CLEANUP_STRENGTH_LABELS[aiResultStrength] : "Balanced"}` : `Translated from ${aiResultSource === "cleaned" ? "cleaned result" : "original"}`} · ${aiResultMeta.cueCount} cues${aiResultMeta.droppedCueCount ? ` · ${aiResultMeta.droppedCueCount} noise removed` : ""} · ${aiResultMeta.modelName}` : "Result"}</small></header><textarea value={aiOutput} onChange={(event) => { const value = event.target.value; setAiOutput(value); if (aiResultMode === "clean") setAiCleanedSource(value); }} placeholder="The result will appear here for review." spellCheck={false} /></div>
            <div className="ai-task-row" aria-live="polite">
              <div className={`ai-feedback ${aiError ? "has-error" : ""}`} role="status">
                <div className="ai-feedback-line"><strong>{aiError ? "AI task stopped" : aiProgress?.label || (aiSource ? "Ready to process subtitles" : "Open an SRT subtitle to begin")}</strong><span>{aiPercent === null ? aiBusy ? "Working" : "" : `${aiPercent}%`}</span></div>
                {aiError && <p title={aiError}>{aiError}</p>}
                <div className="ai-feedback-track"><b className={aiBusy && aiPercent === null ? "indeterminate" : ""} style={{ width: `${aiPercent ?? 0}%` }} /></div>
              </div>
              <div className="ai-task-actions"><button className={`ai-run ai-primary-action ${aiBusy ? "ai-cancel" : ""}`} type="button" aria-label={aiBusy ? "Cancel AI task" : aiActionLabel} title={aiBusy ? "Cancel the current AI task" : aiActionStatus} disabled={aiBusy ? aiCancelling : Boolean(aiSource) && (!selectedAiModel || selectedModelUnavailableOnGpu)} onClick={aiBusy ? cancelAi : runAiAction}>{aiBusy ? aiCancelling ? "Cancelling…" : "Cancel" : aiMode === "clean" ? "Clean Subtitles" : "Translate Subtitles"}</button></div>
            </div>
          </section>
          <aside className="ai-model-section" aria-label="Local processing models">
          <label className="ai-compute-select"><span>Hardware</span><select value={aiComputeBackend} disabled={aiBusy} onChange={(event) => void selectAiBackend(event.target.value as AiComputeBackend)}><option value="cuda">NVIDIA GPU</option><option value="cpu">CPU</option></select></label>
            <div className="ai-model-grid">{aiModels.map((model) => {
              const gpuUnavailable = aiComputeBackend === "cuda" && !model.gpuEligible;
              const unavailableReason = model.gpuReason || "Insufficient GPU VRAM. Choose CPU or a smaller model.";
              return <article className={`${model.id === aiModelId ? "selected" : ""} ${gpuUnavailable ? "unavailable" : ""}`} key={model.id} title={gpuUnavailable ? unavailableReason : model.recommendation} aria-disabled={gpuUnavailable || aiBusy} aria-checked={model.id === aiModelId} role="radio" tabIndex={gpuUnavailable || aiBusy ? -1 : 0} onKeyDown={(event) => { if ((event.key === "Enter" || event.key === " ") && !gpuUnavailable && !aiBusy) { event.preventDefault(); void selectAiModel(model.id); } }} onClick={() => { if (!aiBusy && !gpuUnavailable) void selectAiModel(model.id); }}>
                <div className="ai-model-title"><h3>{model.name}</h3><small>{model.sizeLabel} Model · Min. GPU VRAM {model.minimumGpuMib / 1024} GB</small></div>
                <p className="ai-model-description">{model.recommendation}</p>
                {gpuUnavailable ? <div className="ai-model-limit"><small>{unavailableReason}</small><b>Insufficient GPU VRAM</b></div> : <div className="ai-model-flags">{model.id === recommendedAiModelId && <b>Recommended</b>}{model.modelCached || model.installed ? <small>Downloaded</small> : <button type="button" onClick={(event) => { event.stopPropagation(); void importAiModel(model.id); }}>Import GGUF</button>}</div>}
              </article>;
            })}</div>
          </aside>
        </div>
        <div className="export-dock ai-export">
          <div className="save-location"><Icon name={aiSavedPath ? "check" : "folder"} /><span title={aiSavedPath || settings.defaultDirectory || "Choose when saving"}>{aiSavedPath ? `Saved to ${aiSavedPath}` : "AI output is always saved as a new file"}</span></div>
          <div className="export-actions"><label><select value={aiFormat} onChange={(event) => setAiFormat(event.target.value as ExportFormat)} aria-label="AI export format">{(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((value) => <option value={value} key={value}>{FORMAT_LABELS[value]}</option>)}</select><Icon name="chevron" /></label><button type="button" disabled={!aiSource || aiBusy} onClick={exportOriginalSrt}><Icon name="download" /> Download Original Result</button><button type="button" disabled={!aiOutput || aiBusy} onClick={exportAiResult}><Icon name="download" /> Download AI Result</button></div>
        </div>
      </section>}
    </section>

    {settingsOpen && <div className="modal-backdrop" role="presentation" onMouseDown={() => { if (!componentBusy && !updateBusy) setSettingsOpen(false); }}><section className="settings-modal" role="dialog" aria-modal="true" aria-label="Settings" onMouseDown={(event) => event.stopPropagation()}><div className="modal-heading"><h2>Settings</h2><button type="button" disabled={componentBusy || updateBusy} aria-label="Close settings" onClick={() => setSettingsOpen(false)}>×</button></div><label className="setting-field"><span>Default folder</span><button type="button" onClick={chooseDefaultDirectory}><Icon name="folder" /><b>{settings.defaultDirectory || "Choose"}</b></button></label><label className="setting-field"><span>Default format</span><select value={settings.defaultFormat} onChange={(event) => setSettings((current) => ({ ...current, defaultFormat: event.target.value as ExportFormat }))}>{(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((value) => <option value={value} key={value}>{FORMAT_LABELS[value]}</option>)}</select></label>
      <section className="component-panel"><div className="settings-section-heading"><strong>Video Extraction Components</strong><span className={componentStatus?.ready ? "ready" : "missing"}>{componentStatus?.ready ? componentStatus.gpuRuntime ? "Ready" : "CPU Ready" : "Setup Required"}</span></div><p>Downloads the verified OCR runtime and models. GPU + CPU uses NVIDIA CUDA or DirectML on compatible AMD/Intel graphics, including integrated GPUs. CPU is also available.</p><div className="component-checks"><span>{componentStatus?.ocrRuntime ? "✓" : "○"} CPU OCR Runtime</span><span>{componentStatus?.gpuRuntime ? "✓" : "○"} GPU + CPU OCR Runtime</span></div>{componentBusy && <div className="component-activity" role="status" aria-live="polite"><div className="component-activity-label"><i /> <span>{componentProgressMessage || "Preparing OCR components…"}</span>{componentPercent !== null && <strong>Download: {componentPercent}%</strong>}</div><div className="component-progress-track"><b className={componentPercent === null ? "indeterminate" : ""} style={componentPercent === null ? undefined : { width: `${componentPercent}%` }} /></div></div>}{<label className="component-consent"><input type="checkbox" checked={componentConsent} disabled={componentBusy} onChange={(event) => setComponentConsent(event.target.checked)} /><span>Accept the runtime and OCR dependency licenses to download.</span></label>}{componentError && <p className="settings-error">{componentError}</p>}<button className="settings-action" type="button" disabled={componentBusy || !componentConsent} onClick={installRequiredComponents}>{componentBusy ? "Installing…" : componentStatus?.ready ? "Verify Components" : "Download Components"}</button><small className="component-path" title={componentStatus?.installRoot}>{componentStatus?.installRoot || "%LOCALAPPDATA%\\SubHooper"}</small></section>
      <section className="update-panel"><div className="settings-section-heading"><strong>Application Updates</strong><span>0.4.3 Beta</span></div><p>Signed updates are downloaded from this project's public GitHub Releases page and replace the installed application after approval.</p><p className="update-status">{updateMessage}</p><div className="update-actions"><button type="button" disabled={updateBusy} onClick={checkForUpdates}>{updateBusy ? "Please Wait..." : "Check for Updates"}</button>{availableUpdate && <button type="button" disabled={updateBusy} onClick={installAvailableUpdate}>Install {availableUpdate.version}</button>}</div></section>
      <button className="modal-save" type="button" disabled={componentBusy || updateBusy} onClick={storeSettings}>Save Settings</button></section></div>}
  </main>;
}
