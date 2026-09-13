import { PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { aiInputForMode, AiMode, AiTranslateSource, basename, estimateSrtTextCharacters, ExportFormat, IDLE_STAGE, isSupportedVideo, outputFilename, pipelineAiHandoff, PipelineStage, resultQuality, Summary } from "./pipeline";

type RegionPreset = "Bottom" | "Top" | "Full" | "Custom";
type RegionBox = { x: number; y: number; w: number; h: number };
type DragMode = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";
type PipelineResult = { summary: Summary; srtText: string; srtPath: string; resultDir: string };
type Settings = { defaultDirectory: string; defaultFormat: ExportFormat };
type AiModel = { id: string; name: string; sizeLabel: string; memoryLabel: string; recommendation: string; installed: boolean };
type AiProgress = { phase: string; label: string; received: number; total: number | null };
type AiResult = { srtText: string; modelName: string; cueCount: number; sourceCueCount: number; droppedCueCount: number };
type AiProvider = "local" | "deepl";
type DeepLPlan = "free" | "pro";
type AiCleanupStrength = "light" | "balanced" | "strong";
type ComponentStatus = { videoSubFinder: boolean; ocrRuntime: boolean; ready: boolean; installRoot: string };
type AvailableUpdate = Awaited<ReturnType<typeof check>>;

const VIDEO_FILTER = [{ name: "Video files", extensions: ["mp4", "mkv", "avi", "mov", "webm", "ts", "m2ts", "wmv", "m4v"] }];
const SUBTITLE_FILTER = [{ name: "SRT subtitles", extensions: ["srt"] }];
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
const STEP_LABELS = ["Insert", "Subtitle", "Process", "AI Cleaning"];
const AI_LANGUAGES = ["English", "Turkish", "German", "French", "Spanish", "Italian", "Portuguese", "Arabic", "Japanese", "Korean", "Chinese"];
const CLEANUP_STRENGTH_LABELS: Record<AiCleanupStrength, string> = { light: "Light", balanced: "Balanced", strong: "Strong" };
const CLEANUP_STRENGTH_HELP: Record<AiCleanupStrength, string> = {
  light: "Fix clear text errors and keep uncertain fragments for review.",
  balanced: "Normalize the language and remove only clear OCR noise.",
  strong: "Aggressively remove symbol clusters, broken fragments, and unrelated text.",
};
const STAGE_PROGRESS: Record<PipelineStage["key"], [number, number]> = {
  idle: [0, 0], prepare: [4, 12], vsf: [14, 68], ocr: [70, 91], finalize: [94, 98], complete: [100, 100], cancelled: [0, 0], failed: [0, 0],
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

function clamp(value: number, min: number, max: number) { return Math.min(max, Math.max(min, value)); }

export function updateRegionBox(initial: RegionBox, mode: DragMode, dx: number, dy: number): RegionBox {
  const min = .08;
  if (mode === "move") return { ...initial, x: clamp(initial.x + dx, 0, 1 - initial.w), y: clamp(initial.y + dy, 0, 1 - initial.h) };
  let left = initial.x;
  let top = initial.y;
  let right = initial.x + initial.w;
  let bottom = initial.y + initial.h;
  if (mode.includes("w")) left = clamp(initial.x + dx, 0, right - min);
  if (mode.includes("e")) right = clamp(initial.x + initial.w + dx, left + min, 1);
  if (mode.includes("n")) top = clamp(initial.y + dy, 0, bottom - min);
  if (mode.includes("s")) bottom = clamp(initial.y + initial.h + dy, top + min, 1);
  return { x: left, y: top, w: right - left, h: bottom - top };
}

function joinPath(directory: string, filename: string) { return directory ? directory.replace(/[\\\/]$/, "") + "\\" + filename : filename; }
function secondsLabel(seconds: number) { return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`; }

export default function App() {
  const [videoPath, setVideoPath] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const [preset, setPreset] = useState<RegionPreset>("Bottom");
  const [regionBox, setRegionBox] = useState<RegionBox>(PRESETS.Bottom);
  const [cpuOnly, setCpuOnly] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draggingFile, setDraggingFile] = useState(false);
  const [dragState, setDragState] = useState<{ mode: DragMode; x: number; y: number; box: RegionBox; width: number; height: number } | null>(null);
  const [stage, setStage] = useState<PipelineStage>(IDLE_STAGE);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [srtDraft, setSrtDraft] = useState("");
  const [error, setError] = useState("");
  const [savedPath, setSavedPath] = useState("");
  const [format, setFormat] = useState<ExportFormat>(() => (localStorage.getItem("defaultFormat") as ExportFormat) || "srt");
  const [settings, setSettings] = useState<Settings>(() => ({ defaultDirectory: localStorage.getItem("defaultDirectory") || "", defaultFormat: (localStorage.getItem("defaultFormat") as ExportFormat) || "srt" }));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [componentStatus, setComponentStatus] = useState<ComponentStatus | null>(null);
  const [componentConsent, setComponentConsent] = useState(false);
  const [componentBusy, setComponentBusy] = useState(false);
  const [componentError, setComponentError] = useState("");
  const [updateBusy, setUpdateBusy] = useState(false);
  const [updateMessage, setUpdateMessage] = useState("Not checked");
  const [availableUpdate, setAvailableUpdate] = useState<AvailableUpdate>(null);
  const [elapsed, setElapsed] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [aiModels, setAiModels] = useState<AiModel[]>([]);
  const [aiModelId, setAiModelId] = useState("qwen3-8b-q4km");
  const [aiProvider, setAiProvider] = useState<AiProvider>("local");
  const [aiGuidance, setAiGuidance] = useState("");
  const [deeplApiKey, setDeeplApiKey] = useState("");
  const [deeplPlan, setDeeplPlan] = useState<DeepLPlan>("free");
  const [deeplSettingsOpen, setDeeplSettingsOpen] = useState(false);
  const [allowBilledDeepl, setAllowBilledDeepl] = useState(false);
  const [onlineConsent, setOnlineConsent] = useState(false);
  const [onlineWarningOpen, setOnlineWarningOpen] = useState(false);
  const [aiSource, setAiSource] = useState("");
  const [aiSourceName, setAiSourceName] = useState("");
  const [aiOutput, setAiOutput] = useState("");
  const [aiCleanedSource, setAiCleanedSource] = useState("");
  const [aiMode, setAiMode] = useState<AiMode>("clean");
  const [aiTranslateSource, setAiTranslateSource] = useState<AiTranslateSource>("original");
  const [aiCleanupLanguage, setAiCleanupLanguage] = useState("Auto detect");
  const [aiCleanupStrength, setAiCleanupStrength] = useState<AiCleanupStrength>("balanced");
  const [aiSourceLanguage, setAiSourceLanguage] = useState("Auto detect");
  const [aiLanguage, setAiLanguage] = useState("English");
  const [aiFormat, setAiFormat] = useState<ExportFormat>("srt");
  const [aiBusy, setAiBusy] = useState(false);
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
  const deeplShelfRef = useRef<HTMLElement>(null);

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
      .then((models) => { if (!disposed) setAiModels(models); })
      .catch((reason) => { if (!disposed) setAiError(String(reason)); });
    invoke<ComponentStatus>("component_status")
      .then((status) => { if (!disposed) setComponentStatus(status); })
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
    if (!deeplSettingsOpen || aiProvider !== "deepl" || aiMode !== "translate") return;
    const frame = window.requestAnimationFrame(() => deeplShelfRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    return () => window.cancelAnimationFrame(frame);
  }, [deeplSettingsOpen, aiProvider, aiMode]);

  useEffect(() => {
    if (!busy) {
      if (stage.key === "complete") setProgress(100);
      return;
    }
    const [floor, ceiling] = STAGE_PROGRESS[stage.key];
    setProgress((value) => Math.max(value, floor));
    const timer = window.setInterval(() => setProgress((value) => value < ceiling ? value + 1 : value), 1400);
    return () => window.clearInterval(timer);
  }, [busy, stage.key]);

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

  const quality = useMemo(() => result ? resultQuality(result.summary) : null, [result]);
  const currentLog = logs.at(-1) || stage.label;
  const summary = result?.summary ?? {};

  function selectVideo(path: string) {
    if (!isSupportedVideo(path)) { setError("Select a supported video file."); return; }
    setVideoPath(path);
    setPreviewUrl(convertFileSrc(path));
    setDuration(0); setCurrentTime(0); setPlaying(false); setResult(null); setSrtDraft(""); setError(""); setSavedPath(""); setLogs([]); setStage(IDLE_STAGE); setProgress(0);
  }

  async function chooseVideo() {
    const selected = await open({ multiple: false, directory: false, filters: VIDEO_FILTER });
    if (typeof selected === "string") selectVideo(selected);
  }

  function choosePreset(value: Exclude<RegionPreset, "Custom">) { setRegionBox(PRESETS[value]); setPreset(value); }

  function beginRegionDrag(event: ReactPointerEvent, mode: DragMode) {
    if (busy || !previewRef.current) return;
    event.preventDefault(); event.stopPropagation();
    const bounds = previewRef.current.getBoundingClientRect();
    setDragState({ mode, x: event.clientX, y: event.clientY, box: regionBox, width: bounds.width, height: bounds.height });
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
    } catch (reason) {
      setComponentError(String(reason));
      setSettingsOpen(true);
      return;
    }
    setBusy(true); setResult(null); setSrtDraft(""); setError(""); setSavedPath(""); setLogs([]); setProgress(4); setStage({ key: "prepare", label: "Preparing video", percent: null });
    const namedRegion = preset === "Top" || preset === "Custom" ? "Custom" : preset;
    try {
      const response = await invoke<PipelineResult>("start_pipeline", { request: {
        videoPath, region: namedRegion, regionTop: 1 - regionBox.y, regionBottom: 1 - regionBox.y - regionBox.h, regionLeft: regionBox.x, regionRight: regionBox.x + regionBox.w,
        compute: cpuOnly ? "CPU" : "Auto", ocrCompute: cpuOnly ? "CPU" : "Auto",
      } });
      const aiHandoff = pipelineAiHandoff(response.srtText, response.srtPath, videoPath);
      setResult(response); setSrtDraft(response.srtText); setProgress(100); setStage({ key: "complete", label: "Subtitles ready", percent: 100 });
      setAiSource(aiHandoff.content); setAiSourceName(aiHandoff.name); setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); setAiError(""); setAiSavedPath(""); setAiMode("clean"); setAiProvider("local"); setActiveStep(3);
    } catch (reason) {
      const message = String(reason);
      if (!message.toLowerCase().includes("cancel")) { setError(message); setStage({ key: "failed", label: "Processing failed", percent: null }); }
    } finally { setBusy(false); }
  }

  async function cancel() {
    if (!busy) return;
    try { await invoke("cancel_pipeline"); setStage({ key: "cancelled", label: "Processing cancelled", percent: null }); setProgress(0); } catch (reason) { setError(String(reason)); }
  }

  async function chooseDefaultDirectory() {
    const selected = await open({ multiple: false, directory: true });
    if (typeof selected === "string") setSettings((current) => ({ ...current, defaultDirectory: selected }));
  }

  function storeSettings() {
    localStorage.setItem("defaultDirectory", settings.defaultDirectory); localStorage.setItem("defaultFormat", settings.defaultFormat); setFormat(settings.defaultFormat); setSettingsOpen(false);
  }

  async function installRequiredComponents() {
    if (!componentConsent || componentBusy) return;
    setComponentBusy(true); setComponentError("");
    try {
      setComponentStatus(await invoke<ComponentStatus>("install_components"));
    } catch (reason) {
      setComponentError(String(reason));
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

  async function exportResult() {
    if (!result) return;
    const selected = await save({ defaultPath: joinPath(settings.defaultDirectory, outputFilename(videoPath, format)), filters: [{ name: FORMAT_LABELS[format], extensions: [format] }] });
    if (!selected) return;
    const destination = selected.toLowerCase().endsWith(`.${format}`) ? selected : `${selected}.${format}`;
    try { setSavedPath(await invoke<string>("save_export", { destination, content: srtDraft, format })); setError(""); } catch (reason) { setError(String(reason)); }
  }

  function openAiFromResult() {
    if (!srtDraft) return;
    setAiSource(srtDraft);
    setAiSourceName(outputFilename(videoPath, "srt"));
    setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); setAiError(""); setAiSavedPath(""); setActiveStep(3);
  }

  async function chooseAiSubtitle() {
    const selected = await open({ multiple: false, directory: false, filters: SUBTITLE_FILTER });
    if (typeof selected !== "string") return;
    try {
      const content = await invoke<string>("read_subtitle_file", { path: selected });
      setAiSource(content); setAiSourceName(basename(selected)); setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); setAiError(""); setAiSavedPath("");
    } catch (reason) { setAiError(String(reason)); }
  }

  async function downloadAiModel(modelId: string) {
    if (aiBusy) return;
    setAiBusy(true); setAiProvider("local"); setAiModelId(modelId); setAiError(""); setAiProgress({ phase: "starting", label: "Preparing secure download", received: 0, total: null });
    try {
      const installed = await invoke<AiModel>("download_ai_model", { modelId });
      setAiModels((models) => models.map((model) => model.id === installed.id ? installed : model));
    } catch (reason) { setAiError(String(reason)); }
    finally { setAiBusy(false); }
  }

  async function runAiProcessing() {
    if (!aiSource || aiBusy) return;
    const input = aiInputForMode(aiSource, aiCleanedSource, aiMode, aiTranslateSource);
    setAiBusy(true); setAiSavedPath(""); setAiError("");
    setAiProgress({ phase: "starting", label: aiProvider === "deepl" ? "Preparing online translation" : "Starting local AI", received: 0, total: null });
    try {
      const response = await invoke<AiResult>("run_ai_cleaning", { request: {
        content: input, modelId: aiModelId, mode: aiMode, cleanupLanguage: aiMode === "clean" ? aiCleanupLanguage : null, cleanupStrength: aiMode === "clean" ? aiCleanupStrength : null, sourceLanguage: aiMode === "translate" ? aiSourceLanguage : null, targetLanguage: aiMode === "translate" ? aiLanguage : null,
        guidance: aiMode === "translate" ? aiGuidance : null,
        provider: aiProvider,
        deeplApiKey: aiProvider === "deepl" ? deeplApiKey : null,
        deeplPlan: aiProvider === "deepl" ? deeplPlan : null,
        allowBilledDeepl: aiProvider === "deepl" && deeplPlan === "pro" && allowBilledDeepl,
      } });
      setAiOutput(response.srtText); setAiResultMeta(response);
      setAiResultMode(aiMode); setAiResultLanguage(aiMode === "translate" ? aiLanguage : "");
      setAiResultSource(aiMode === "translate" ? (aiCleanedSource ? aiTranslateSource : "original") : null);
      setAiResultStrength(aiMode === "clean" ? aiCleanupStrength : null);
      if (aiMode === "clean") { setAiCleanedSource(response.srtText); setAiTranslateSource("original"); }
      setAiProgress({ phase: "complete", label: response.droppedCueCount ? `AI result ready · ${response.droppedCueCount} noise cues removed` : "AI result ready", received: 1, total: 1 });
    } catch (reason) { setAiError(String(reason)); }
    finally { setAiBusy(false); }
  }

  async function runAiAction() {
    if (!aiSource) {
      await chooseAiSubtitle();
      return;
    }
    if (aiProvider === "deepl") {
      if (aiMode !== "translate") {
        setAiProvider("local");
        setAiError("DeepL is available only for translation.");
        return;
      }
      if (!deeplApiKey.trim()) {
        setDeeplSettingsOpen(true);
        setAiError("Add a personal DeepL API key in the DeepL account shelf. API Free also requires its own key.");
        return;
      }
      if (deeplPlan === "pro" && !allowBilledDeepl) {
        setDeeplSettingsOpen(true);
        setAiError("Confirm potentially billed DeepL API Pro access before translating.");
        return;
      }
      if (aiSourceLanguage !== "Auto detect" && aiSourceLanguage === aiLanguage) {
        setAiError("Source and target languages must be different for online translation.");
        return;
      }
      if (!onlineConsent) {
        setOnlineWarningOpen(true);
        return;
      }
      await runAiProcessing();
      return;
    }
    if (!selectedAiModel) {
      setAiError("Select a local model.");
      return;
    }
    if (!selectedAiModel.installed) {
      await downloadAiModel(selectedAiModel.id);
      return;
    }
    await runAiProcessing();
  }

  async function cancelAi() {
    try { await invoke("cancel_ai"); } catch (reason) { setAiError(String(reason)); }
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

  const selectedAiModel = aiModels.find((model) => model.id === aiModelId);
  const aiTranslationInput = aiInputForMode(aiSource, aiCleanedSource, "translate", aiTranslateSource);
  const deeplCharacterEstimate = estimateSrtTextCharacters(aiTranslationInput);
  const aiPercent = aiProgress?.total ? Math.min(100, Math.round(aiProgress.received / aiProgress.total * 100)) : null;
  const aiActionLabel = !aiSource
    ? "Open SRT to continue"
    : aiProvider === "deepl" && aiMode === "translate"
      ? "Translate with DeepL"
      : !selectedAiModel?.installed
      ? `Download ${selectedAiModel?.name || "a model"}`
      : aiMode === "clean" ? "Clean subtitles" : `Translate to ${aiLanguage}`;
  const aiActionStatus = !aiSource
    ? "An SRT subtitle is required before processing"
    : aiProvider === "deepl" && aiMode === "translate"
      ? "Sends subtitle text and scene context to DeepL only after confirmation"
      : selectedAiModel?.installed
      ? aiMode === "translate" && aiCleanedSource
        ? `Will translate the ${aiTranslateSource === "cleaned" ? "cleaned result" : "original SRT"}`
        : aiMode === "clean"
          ? `${CLEANUP_STRENGTH_LABELS[aiCleanupStrength]} cleanup runs locally on this PC`
          : "Runs locally on this PC"
      : "The selected model must be downloaded before processing";

  const canOpenStep = (index: number) => index === 0 || index === 3 || index === 1 && Boolean(videoPath) || index === 2 && Boolean(result);

  return <main className="app-shell">
    <header className="topbar">
      <img className="brand-logo" src="/subhooper-logo.svg" alt="SubHooper" />
      <nav className="step-tabs" aria-label="Workflow steps">{STEP_LABELS.map((label, index) => <button type="button" className={`${index === activeStep ? "active" : ""} ${index < activeStep || index === 2 && Boolean(result) ? "completed" : ""}`} disabled={!canOpenStep(index) || busy || aiBusy} onClick={() => setActiveStep(index)} key={label}><span>{index + 1}</span>{label}</button>)}</nav>
      <button className="icon-button" type="button" aria-label="Settings" onClick={() => setSettingsOpen(true)}><Icon name="settings" /></button>
    </header>

    <section className="page-frame">
      {activeStep === 0 && <section className="source-page page-enter">
        <button className={`drop-zone ${draggingFile ? "dragging" : ""}`} type="button" disabled={busy} onClick={chooseVideo}>
          <span className="drop-icon"><Icon name="folder" /></span>
          <span className="drop-copy"><strong>{videoPath ? basename(videoPath) : "Upload a video file"}</strong><span>{videoPath ? "Ready to continue" : "Drag and drop a video here, or click to browse"}</span></span>
          {videoPath && <small>{videoPath}</small>}
        </button>
        <button className="page-action" type="button" disabled={!videoPath} onClick={() => setActiveStep(1)}>Next Step</button>
      </section>}

      {activeStep === 1 && <section className="region-page page-enter">
        <div className="region-card">
          <div className="preview-column">
            <div className="video-preview" ref={previewRef}>
              {previewUrl ? <video ref={videoRef} src={previewUrl} preload="metadata" onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)} onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} /> : <div className="video-fallback"><Icon name="video" /></div>}
              <div className="selection-box" style={{ left: `${regionBox.x * 100}%`, top: `${regionBox.y * 100}%`, width: `${regionBox.w * 100}%`, height: `${regionBox.h * 100}%` }} onPointerDown={(event) => beginRegionDrag(event, "move")}>{(["n", "s", "e", "w", "ne", "nw", "se", "sw"] as DragMode[]).map((mode) => <i className={`handle ${mode}`} onPointerDown={(event) => beginRegionDrag(event, mode)} key={mode} />)}</div>
            </div>
            <div className="video-controls"><button type="button" onClick={togglePreview} aria-label={playing ? "Pause" : "Play"}>{playing ? "Ⅱ" : "▶"}</button><input type="range" min="0" max={duration || 0} step="0.05" value={Math.min(currentTime, duration || 0)} onChange={(event) => seekPreview(Number(event.target.value))} /><time>{secondsLabel(Math.floor(currentTime))}</time></div>
          </div>
          <div className="preset-row">{(Object.keys(PRESETS) as Array<Exclude<RegionPreset, "Custom">>).map((value) => <button type="button" className={preset === value ? "selected" : ""} onClick={() => choosePreset(value)} key={value}><span className={`preset-icon preset-${value.toLowerCase()}`} />{PRESET_LABELS[value]}</button>)}</div>
          <div className="compute-row" role="group" aria-label="Processing mode">
            <button type="button" className={!cpuOnly ? "selected" : ""} disabled={busy} onClick={() => setCpuOnly(false)}><span><Icon name="gpu" /></span><strong>GPU Acceleration</strong><small>Automatic · NVIDIA when available</small></button>
            <button type="button" className={cpuOnly ? "selected" : ""} disabled={busy} onClick={() => setCpuOnly(true)}><span><Icon name="cpu" /></span><strong>CPU Only</strong><small>Compatibility mode</small></button>
          </div>
        </div>
        {error && <div className="error-card"><strong>Processing failed</strong><p>{error}</p></div>}
        <div className="extract-row"><button className={`extract-action ${busy ? "running" : ""}`} type="button" disabled={!videoPath || busy} onClick={start}><i className="extract-progress" style={{ width: `${progress}%` }} /><i className="extract-sweep" />{busy ? <><span className="extract-spinner" /><span className="extract-copy"><strong>{stage.label} · {secondsLabel(elapsed)}</strong><small title={currentLog}>{currentLog}</small></span><b>~{progress}%</b></> : <span className="extract-label">Extract Subtitles</span>}</button>{busy && <button className="cancel-action" type="button" onClick={cancel}>Cancel</button>}</div>
      </section>}

      {activeStep === 2 && result && <section className="result-page page-enter">
        <div className="result-heading"><div><h1>{summary.Subtitles || 0} subtitles extracted</h1><span className={`quality ${quality}`}>{quality === "good" ? "Ready" : "Review needed"}</span></div></div>
        <div className="result-layout"><section className="editor-card"><div className="editor-toolbar"><span><Icon name="edit" /> Subtitles</span><small>{basename(videoPath)}</small></div><textarea value={srtDraft} onChange={(event) => setSrtDraft(event.target.value)} spellCheck={false} aria-label="Editable subtitles" /></section><aside className="result-sidebar">
          <div className="metrics"><div><span>Scan</span><strong>{summary.VSFSeconds || "—"}s</strong></div><div><span>OCR</span><strong>{summary.OCRSeconds || "—"}s</strong></div><div><span>Device</span><strong>{summary.OCRCompute?.replace("_ACTIVE", "") || "—"}</strong></div><div><span>Flagged</span><strong>{summary.SuspiciousShortCues || "0"}</strong></div></div>
          <button className="ai-action" type="button" onClick={openAiFromResult}><span className="ai-icon"><Icon name="spark" /></span><span className="ai-copy"><strong>Clean subtitles with AI</strong><small>Optional · original stays unchanged</small></span></button>
          <div className="result-log" tabIndex={0}><pre>{logs.length ? logs.join("\n") : "—"}</pre></div>
        </aside></div>
        <div className="export-dock">
          <div className="save-location"><Icon name={savedPath ? "check" : "folder"} /><span title={savedPath || settings.defaultDirectory || "Choose when saving"}>{savedPath ? `Saved to ${savedPath}` : `Save location · ${settings.defaultDirectory || "Choose when saving"}`}</span></div>
          <div className="export-actions"><label><select value={format} onChange={(event) => setFormat(event.target.value as ExportFormat)} aria-label="File format">{(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((value) => <option value={value} key={value}>{FORMAT_LABELS[value]}</option>)}</select><Icon name="chevron" /></label><button type="button" onClick={exportResult}><Icon name="download" /> Download</button></div>
        </div>
      </section>}

      {activeStep === 3 && <section className="ai-page page-enter">
        <div className="ai-page-heading">
          <div className="ai-heading-title"><span className="ai-page-icon"><Icon name="spark" /></span><div className="ai-heading-copy"><p>Local cleanup · local or online translation</p><h1>AI Cleaning</h1><span>Normalize the subtitle language, remove OCR noise, review the result, and export a new copy.</span></div></div>
          <div className="ai-heading-actions">
            <div className="ai-mode-control" role="group" aria-label="AI task">
              <button type="button" className={aiMode === "clean" ? "selected" : ""} disabled={aiBusy} onClick={() => { setAiMode("clean"); setAiProvider("local"); }}>Clean</button>
              <button type="button" className={aiMode === "translate" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiMode("translate")}>Translate</button>
              {aiMode === "clean" && <label><span>Language</span><select value={aiCleanupLanguage} disabled={aiBusy} onChange={(event) => setAiCleanupLanguage(event.target.value)}><option>Auto detect</option>{AI_LANGUAGES.map((language) => <option key={language}>{language}</option>)}</select></label>}
              {aiMode === "translate" && <label><span>Target</span><select value={aiLanguage} disabled={aiBusy} onChange={(event) => setAiLanguage(event.target.value)}>{AI_LANGUAGES.map((language) => <option key={language}>{language}</option>)}</select></label>}
            </div>
            {aiBusy ? <button className="ai-cancel" type="button" onClick={cancelAi}>Cancel</button> : <button className="ai-run" type="button" disabled={Boolean(aiSource) && aiProvider === "local" && !selectedAiModel} onClick={runAiAction}><Icon name={aiProvider === "local" && aiSource && !selectedAiModel?.installed ? "download" : aiSource ? "spark" : "folder"} /> {aiActionLabel}</button>}
            {aiMode === "clean" && <fieldset className="ai-clean-strength"><legend>Cleanup strength</legend><span>Cleanup strength</span>{(Object.keys(CLEANUP_STRENGTH_LABELS) as AiCleanupStrength[]).map((strength) => <button type="button" key={strength} className={aiCleanupStrength === strength ? "selected" : ""} disabled={aiBusy} aria-pressed={aiCleanupStrength === strength} title={CLEANUP_STRENGTH_HELP[strength]} onClick={() => setAiCleanupStrength(strength)}>{CLEANUP_STRENGTH_LABELS[strength]}</button>)}<small>{CLEANUP_STRENGTH_HELP[aiCleanupStrength]}</small></fieldset>}
            {aiMode === "translate" && aiCleanedSource && <fieldset className="ai-translate-source"><legend>Translate from</legend><button type="button" className={aiTranslateSource === "original" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiTranslateSource("original")}><span>Original SRT</span><small>Use the imported subtitle</small></button><button type="button" className={aiTranslateSource === "cleaned" ? "selected" : ""} disabled={aiBusy} onClick={() => setAiTranslateSource("cleaned")}><span>Cleaned result</span><small>Use the latest edited cleanup</small></button></fieldset>}
            <span className="ai-action-status">{aiProvider === "deepl" ? `DeepL API ${deeplPlan === "free" ? "Free" : "Pro"}` : selectedAiModel?.name || "Select a model"} · {aiActionStatus}</span>
          </div>
        </div>

        <section className="ai-source-card">
          <button type="button" onClick={chooseAiSubtitle} disabled={aiBusy}><Icon name="folder" /><span><strong>{aiSourceName || "Open an SRT subtitle"}</strong><small>{aiSource ? "Ready for processing" : "No video or extraction result is required"}</small></span></button>
          {aiMode === "translate" && <div className="ai-guidance"><div className="ai-guidance-head"><span>Context & terminology <small>optional</small></span><label><small>Source</small><select value={aiSourceLanguage} disabled={aiBusy} onChange={(event) => setAiSourceLanguage(event.target.value)}><option>Auto detect</option>{AI_LANGUAGES.map((language) => <option key={language}>{language}</option>)}</select></label></div><textarea aria-label="Context and terminology" value={aiGuidance} maxLength={4000} disabled={aiBusy} onChange={(event) => setAiGuidance(event.target.value)} placeholder="Plot context, character names, relationships, preferred terms, tone, or honorifics. This is used with nearby subtitle cues." /></div>}
        </section>

        <section className="ai-model-section">
          <div className="ai-section-heading"><div><h2>Choose a processing engine</h2><p>Local models stay on this PC. DeepL API Free is optional, free within its quota, and requires a personal API key.</p></div><span>Apache-2.0 models · MIT engine</span></div>
          <div className="ai-model-grid">{aiModels.map((model) => <article className={`${aiProvider === "local" && model.id === aiModelId ? "selected" : ""} ${model.id === "qwen3-8b-q4km" ? "recommended" : ""}`} key={model.id} onClick={() => { if (!aiBusy) { setAiProvider("local"); setAiModelId(model.id); } }}>
            {model.id === "qwen3-8b-q4km" && <b>Recommended</b>}
            <div className="ai-model-title"><span className="ai-model-radio" /><div><h3>{model.name}</h3><small>{model.sizeLabel} · {model.memoryLabel}</small></div></div>
            <p>{model.recommendation}</p>
            <button type="button" disabled={aiBusy || model.installed} onClick={(event) => { event.stopPropagation(); downloadAiModel(model.id); }}>{model.installed ? <><Icon name="check" /> Ready</> : <><Icon name="download" /> Download</>}</button>
          </article>)}
          <article className={`ai-online-card ${aiProvider === "deepl" ? "selected" : ""} ${aiMode !== "translate" ? "unavailable" : ""}`} aria-disabled={aiMode !== "translate"} onClick={() => { if (!aiBusy && aiMode === "translate") { setAiProvider("deepl"); setAiError(""); } }}>
            <b>Free online</b>
            <div className="ai-model-title"><span className="ai-model-radio" /><div><h3>Online Translation</h3><small>DeepL API Free · personal key required</small></div></div>
            <p>Higher-quality translation with nearby scene context. Free mode is locked to DeepL's no-billing API endpoint.</p>
            <button type="button" disabled={aiBusy || aiMode !== "translate"} onClick={(event) => { event.stopPropagation(); if (aiMode === "translate") { setAiProvider("deepl"); setDeeplPlan("free"); setAllowBilledDeepl(false); setOnlineConsent(false); setDeeplSettingsOpen(true); setAiError(""); } }}>{aiMode === "translate" ? "Set up DeepL Free" : "Translation only"}</button>
          </article></div>
        </section>

        {aiProvider === "deepl" && aiMode === "translate" && <section ref={deeplShelfRef} className={`deepl-shelf ${deeplSettingsOpen ? "open" : ""}`}>
          <button className="deepl-shelf-toggle" type="button" aria-expanded={deeplSettingsOpen} onClick={() => setDeeplSettingsOpen((value) => !value)}><span><strong>DeepL account</strong><small>{deeplPlan === "free" ? "API Free · no-billing endpoint" : "API Pro · billed account"}{deeplApiKey ? " · key added for this session" : " · key required"}</small></span><Icon name="chevron" /></button>
          {deeplSettingsOpen && <div className="deepl-settings">
            <div className="deepl-intro"><h2>Personal DeepL API access</h2><p>DeepL requires an account key even on API Free. No shared key is bundled. The key stays in memory for this session and is never written to settings, results, or diagnostic logs.</p></div>
            <label className="deepl-plan"><span>DeepL API plan</span><select value={deeplPlan} disabled={aiBusy} onChange={(event) => { const plan = event.target.value as DeepLPlan; setDeeplPlan(plan); setAllowBilledDeepl(false); setOnlineConsent(false); }}><option value="free">API Free — no billing endpoint</option><option value="pro">API Pro — potentially billed</option></select></label>
            <label className="deepl-key"><span>Personal DeepL API key</span><input type="password" autoComplete="off" value={deeplApiKey} disabled={aiBusy} onChange={(event) => { setDeeplApiKey(event.target.value); setOnlineConsent(false); }} placeholder="Paste API key" /></label>
            <div className="deepl-plan-note"><strong>{deeplPlan === "free" ? "Free endpoint locked" : "Paid plan selected"}</strong><span>Estimated source text: <b>{deeplCharacterEstimate.toLocaleString()}</b> characters. {deeplPlan === "free" ? "Requests can only use api-free.deepl.com. DeepL API Free includes 500,000 source characters per month and this mode cannot call the billed endpoint." : "Requests use the API Pro endpoint only after the separate billed-account confirmation below. DeepL account cost controls remain external to SubHooper."}</span></div>
            {deeplPlan === "pro" && <label className="deepl-billed"><input type="checkbox" checked={allowBilledDeepl} disabled={aiBusy} onChange={(event) => { setAllowBilledDeepl(event.target.checked); setOnlineConsent(false); }} /><span>Allow this personal DeepL API Pro account to be billed under its existing plan and cost controls. SubHooper cannot subscribe, upgrade, or change spending limits.</span></label>}
          </div>}
        </section>}

        {(aiBusy || aiProgress) && <div className={`ai-progress-card ${aiProgress?.phase === "complete" ? "complete" : ""}`}><div><strong>{aiProgress?.label || "Preparing local AI"}</strong><span>{aiPercent === null ? "Please wait" : `${aiPercent}%`}</span></div><i><b style={{ width: `${aiPercent ?? 18}%` }} /></i></div>}
        {aiError && <div className="error-card ai-error"><strong>AI task stopped</strong><p>{aiError}</p></div>}

        <section className="ai-workspace">
          <div className="ai-editor"><header><span>Original SRT</span><small>{aiSourceName || "No file selected"}</small></header><textarea value={aiSource} onChange={(event) => { setAiSource(event.target.value); setAiOutput(""); setAiCleanedSource(""); setAiTranslateSource("original"); setAiResultMeta(null); setAiResultMode(null); setAiResultLanguage(""); setAiResultSource(null); setAiResultStrength(null); }} placeholder="Open an SRT file to begin." spellCheck={false} /></div>
          <div className="ai-editor output"><header><span>AI result</span><small>{aiResultMeta ? `${aiResultMode === "clean" ? `Cleaned · ${aiResultStrength ? CLEANUP_STRENGTH_LABELS[aiResultStrength] : "Balanced"}` : `Translated from ${aiResultSource === "cleaned" ? "cleaned result" : "original"}`} · ${aiResultMeta.cueCount} cues${aiResultMeta.droppedCueCount ? ` · ${aiResultMeta.droppedCueCount} noise removed` : ""} · ${aiResultMeta.modelName}` : "A separate copy will appear here"}</small></header><textarea value={aiOutput} onChange={(event) => { const value = event.target.value; setAiOutput(value); if (aiResultMode === "clean") setAiCleanedSource(value); }} placeholder="The cleaned or translated result will appear here for review." spellCheck={false} /></div>
        </section>

        <div className="export-dock ai-export">
          <div className="save-location"><Icon name={aiSavedPath ? "check" : "folder"} /><span title={aiSavedPath || settings.defaultDirectory || "Choose when saving"}>{aiSavedPath ? `Saved to ${aiSavedPath}` : "AI output is always saved as a new file"}</span></div>
          <div className="export-actions"><label><select value={aiFormat} onChange={(event) => setAiFormat(event.target.value as ExportFormat)} aria-label="AI export format">{(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((value) => <option value={value} key={value}>{FORMAT_LABELS[value]}</option>)}</select><Icon name="chevron" /></label><button type="button" disabled={!aiOutput || aiBusy} onClick={exportAiResult}><Icon name="download" /> Download result</button></div>
        </div>
      </section>}
    </section>

    {settingsOpen && <div className="modal-backdrop" role="presentation" onMouseDown={() => { if (!componentBusy && !updateBusy) setSettingsOpen(false); }}><section className="settings-modal" role="dialog" aria-modal="true" aria-label="Settings" onMouseDown={(event) => event.stopPropagation()}><div className="modal-heading"><h2>Settings</h2><button type="button" disabled={componentBusy || updateBusy} aria-label="Close settings" onClick={() => setSettingsOpen(false)}>×</button></div><label className="setting-field"><span>Default folder</span><button type="button" onClick={chooseDefaultDirectory}><Icon name="folder" /><b>{settings.defaultDirectory || "Choose"}</b></button></label><label className="setting-field"><span>Default format</span><select value={settings.defaultFormat} onChange={(event) => setSettings((current) => ({ ...current, defaultFormat: event.target.value as ExportFormat }))}>{(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((value) => <option value={value} key={value}>{FORMAT_LABELS[value]}</option>)}</select></label>
      <section className="component-panel"><div className="settings-section-heading"><strong>Video extraction components</strong><span className={componentStatus?.ready ? "ready" : "missing"}>{componentStatus?.ready ? "Ready" : "Setup required"}</span></div><p>VideoSubFinder 6.10, its Microsoft Visual C++ runtime, a private Python 3.14.7 runtime, RapidVideOCR, RapidOCR, and ONNX Runtime are downloaded only when this setup is approved. They are stored outside the application folder and are not included in SubHooper.</p><div className="component-checks"><span>{componentStatus?.videoSubFinder ? "✓" : "○"} VideoSubFinder</span><span>{componentStatus?.ocrRuntime ? "✓" : "○"} OCR runtime</span></div>{!componentStatus?.ready && <label className="component-consent"><input type="checkbox" checked={componentConsent} disabled={componentBusy} onChange={(event) => setComponentConsent(event.target.checked)} /><span>Download from Microsoft, SourceForge, Python.org, and Python package repositories; accept the separate GPL-2.0, Python, Apache-2.0, and dependency licenses. A Microsoft-signed runtime may request administrator approval. No telemetry or account is added by SubHooper.</span></label>}{componentError && <p className="settings-error">{componentError}</p>}<button className="settings-action" type="button" disabled={componentBusy || componentStatus?.ready || !componentConsent} onClick={installRequiredComponents}>{componentBusy ? "Installing components..." : componentStatus?.ready ? "Components ready" : "Download and install components"}</button><small className="component-path" title={componentStatus?.installRoot}>{componentStatus?.installRoot || "%LOCALAPPDATA%\\SubHooper"}</small></section>
      <section className="update-panel"><div className="settings-section-heading"><strong>Application updates</strong><span>0.3.7 beta</span></div><p>Signed updates are downloaded from this project's public GitHub Releases page and replace the installed application after approval.</p><p className="update-status">{updateMessage}</p><div className="update-actions"><button type="button" disabled={updateBusy} onClick={checkForUpdates}>{updateBusy ? "Please wait..." : "Check for updates"}</button>{availableUpdate && <button type="button" disabled={updateBusy} onClick={installAvailableUpdate}>Install {availableUpdate.version}</button>}</div></section>
      <button className="modal-save" type="button" disabled={componentBusy || updateBusy} onClick={storeSettings}>Save Settings</button></section></div>}
    {onlineWarningOpen && <div className="modal-backdrop" role="presentation" onMouseDown={() => setOnlineWarningOpen(false)}><section className="online-warning-modal" role="dialog" aria-modal="true" aria-label="Online translation warning" onMouseDown={(event) => event.stopPropagation()}><div className="modal-heading"><h2>Online data transfer</h2><button type="button" aria-label="Close warning" onClick={() => setOnlineWarningOpen(false)}>×</button></div><p>The selected subtitle text, nearby dialogue used as context, and optional terminology guidance will be sent directly to DeepL for translation.</p><ul><li>Video, OCR images, local paths, and SubHooper reports are not sent.</li><li>SubHooper includes no shared API key and receives no payment.</li><li>{deeplPlan === "free" ? "API Free mode is locked to DeepL's no-billing endpoint and stops when the free quota is exhausted." : "API Pro may charge the selected personal account under its existing DeepL plan and cost controls."}</li><li>Confidential material should be sent only when the selected DeepL plan and data policy are acceptable.</li></ul><div className="online-warning-actions"><button type="button" onClick={() => setOnlineWarningOpen(false)}>Cancel</button><button type="button" onClick={() => { setOnlineConsent(true); setOnlineWarningOpen(false); void runAiProcessing(); }}>I understand — translate</button></div></section></div>}
  </main>;
}
