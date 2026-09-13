use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "ts", "m2ts", "wmv", "m4v",
];
const AI_MAX_SUBTITLE_BYTES: usize = 20 * 1024 * 1024;
const AI_USER_AGENT: &str = "SubHooper/0.3.7-beta";
const AI_TARGET_CHUNK_SIZE: usize = 12;
const AI_CONTEXT_CUES: usize = 4;
const AI_DOCUMENT_SAMPLE_CUES: usize = 24;
const DEEPL_TARGET_CHUNK_SIZE: usize = 32;

#[derive(Clone, Copy)]
struct AiModelSpec {
    id: &'static str,
    name: &'static str,
    repository: &'static str,
    size_label: &'static str,
    memory_label: &'static str,
    recommendation: &'static str,
}

const AI_MODELS: &[AiModelSpec] = &[
    AiModelSpec {
        id: "qwen3-4b-q4km",
        name: "Qwen3 4B",
        repository: "Qwen/Qwen3-4B-GGUF",
        size_label: "about 2.5 GB",
        memory_label: "8 GB RAM recommended",
        recommendation:
            "Faster OCR cleanup with dominant-language normalization. Results should be reviewed.",
    },
    AiModelSpec {
        id: "qwen3-8b-q4km",
        name: "Qwen3 8B",
        repository: "Qwen/Qwen3-8B-GGUF",
        size_label: "about 5.2 GB",
        memory_label: "12 GB RAM recommended",
        recommendation:
            "Recommended for language normalization, OCR-noise removal, and contextual consistency.",
    },
];

#[derive(Default)]
struct PipelineState {
    active_pid: Mutex<Option<u32>>,
    cancel_requested: AtomicBool,
    last_result: Mutex<Option<PipelineResult>>,
}

#[derive(Default)]
struct AiState {
    active_pid: Mutex<Option<u32>>,
    cancel_requested: AtomicBool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiModelInfo {
    id: String,
    name: String,
    size_label: String,
    memory_label: String,
    recommendation: String,
    installed: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiProgress {
    phase: String,
    label: String,
    received: u64,
    total: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiRequest {
    content: String,
    model_id: String,
    mode: String,
    target_language: Option<String>,
    #[serde(default)]
    source_language: Option<String>,
    #[serde(default)]
    cleanup_language: Option<String>,
    #[serde(default)]
    cleanup_strength: Option<String>,
    #[serde(default)]
    guidance: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    deepl_api_key: Option<String>,
    #[serde(default)]
    deepl_plan: Option<String>,
    #[serde(default)]
    allow_billed_deepl: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiResult {
    srt_text: String,
    model_name: String,
    cue_count: usize,
    source_cue_count: usize,
    dropped_cue_count: usize,
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    assets: Vec<GithubAsset>,
}

#[derive(Clone, Debug, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    digest: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceTreeEntry {
    path: String,
    #[serde(rename = "type")]
    entry_type: String,
    lfs: Option<HuggingFaceLfs>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceLfs {
    oid: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AiCueOutput {
    id: usize,
    text: String,
}

#[derive(Debug, Serialize)]
struct DeepLTranslateRequest {
    text: Vec<String>,
    target_lang: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_lang: Option<String>,
    context: String,
    split_sentences: &'static str,
    preserve_formatting: bool,
    model_type: &'static str,
}

#[derive(Debug, Deserialize)]
struct DeepLTranslation {
    text: String,
}

#[derive(Debug, Deserialize)]
struct DeepLTranslateResponse {
    translations: Vec<DeepLTranslation>,
}

struct TemporaryPrompt {
    path: PathBuf,
}

struct TemporaryOutput {
    path: PathBuf,
}

impl TemporaryOutput {
    fn create() -> Result<Self, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("Could not read the system clock: {error}"))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "subhooper-ai-output-{}-{timestamp}.txt",
            std::process::id()
        ));
        fs::write(&path, b"")
            .map_err(|error| format!("Could not prepare the local AI output file: {error}"))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl TemporaryPrompt {
    fn create(content: &str) -> Result<Self, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("Could not read the system clock: {error}"))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "subhooper-ai-prompt-{}-{timestamp}.txt",
            std::process::id()
        ));
        fs::write(&path, content.as_bytes())
            .map_err(|error| format!("Could not prepare the local AI prompt: {error}"))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryPrompt {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineRequest {
    video_path: String,
    region: String,
    region_top: f64,
    region_bottom: f64,
    region_left: f64,
    region_right: f64,
    compute: String,
    ocr_compute: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelineResult {
    summary: BTreeMap<String, String>,
    srt_text: String,
    srt_path: String,
    result_dir: String,
}

#[derive(Clone, Debug, Serialize)]
struct PipelineStage {
    key: &'static str,
    label: &'static str,
    percent: Option<u8>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComponentStatus {
    video_sub_finder: bool,
    ocr_runtime: bool,
    ready: bool,
    install_root: String,
}

#[derive(Debug)]
struct SubtitleCue {
    start: String,
    end: String,
    text: Vec<String>,
}

fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, String> {
    mutex
        .lock()
        .map_err(|_| "The processing state lock is unavailable.".to_string())
}

fn validate_choice(value: &str, allowed: &[&str], label: &str) -> Result<(), String> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(format!("Invalid {label}: {value}"))
    }
}

fn resolve_project_root() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(value) = std::env::var_os("SUBTITLE_PROJECT_ROOT") {
        candidates.push(PathBuf::from(value));
    }
    if let Ok(current) = std::env::current_dir() {
        candidates.extend(current.ancestors().map(Path::to_path_buf));
    }
    if let Ok(executable) = std::env::current_exe() {
        candidates.extend(executable.ancestors().map(Path::to_path_buf));
    }
    for candidate in candidates {
        for project in [candidate.clone(), candidate.join("app")] {
            if project.join("run-pipeline.ps1").is_file()
                && project.join("engine").join("pipeline.py").is_file()
            {
                return project
                    .canonicalize()
                    .map_err(|error| format!("Could not resolve the project path: {error}"));
            }
        }
    }
    Err("Could not find the SubHooper project root.".to_string())
}

fn resolve_home_root(project: &Path) -> PathBuf {
    if let Some(configured) = std::env::var_os("SUBHOOPER_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(configured);
    }
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA").filter(|value| !value.is_empty())
    {
        return PathBuf::from(local_app_data).join("SubHooper");
    }
    project.to_path_buf()
}

fn component_status_value(project: &Path) -> ComponentStatus {
    let root = resolve_home_root(project);
    let video_sub_finder = root
        .join("components")
        .join("VideoSubFinder-6.10")
        .join("Release_x64")
        .join("VideoSubFinderWXW.exe")
        .is_file();
    let ocr_runtime = root
        .join("runtime")
        .join("ocr-cpu-py314-auto")
        .join("Scripts")
        .join("python.exe")
        .is_file();
    ComponentStatus {
        video_sub_finder,
        ocr_runtime,
        ready: video_sub_finder && ocr_runtime,
        install_root: root.display().to_string(),
    }
}

#[tauri::command]
fn component_status() -> Result<ComponentStatus, String> {
    let project = resolve_project_root()?;
    Ok(component_status_value(&project))
}

fn install_components_blocking(app: AppHandle) -> Result<ComponentStatus, String> {
    let project = resolve_project_root()?;
    let script = project.join("install-components.ps1");
    if !script.is_file() {
        return Err(
            "The component installer is missing from this SubHooper installation.".to_string(),
        );
    }
    let executable = if cfg!(target_os = "windows") {
        "powershell.exe"
    } else {
        "pwsh"
    };
    let mut command = Command::new(executable);
    command
        .current_dir(&project)
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(&script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    let output = command
        .output()
        .map_err(|error| format!("Could not start component setup: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        let detail = if stderr.is_empty() { stdout } else { stderr };
        return Err(format!("Component setup failed. {detail}"));
    }
    let status = component_status_value(&project);
    if !status.ready {
        return Err("Component setup completed without a usable OCR runtime.".to_string());
    }
    let _ = app.emit("component-progress", "SubHooper components are ready.");
    Ok(status)
}

#[tauri::command]
async fn install_components(app: AppHandle) -> Result<ComponentStatus, String> {
    tauri::async_runtime::spawn_blocking(move || install_components_blocking(app))
        .await
        .map_err(|error| format!("The component setup task stopped unexpectedly: {error}"))?
}

fn resolve_results_root(project: &Path) -> Result<PathBuf, String> {
    let configured = std::env::var_os("SUBTITLE_RESULTS_ROOT")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| resolve_home_root(project).join("results"));
    fs::create_dir_all(&configured)
        .map_err(|error| format!("Could not create the results directory: {error}"))?;
    configured
        .canonicalize()
        .map_err(|error| format!("Could not resolve the results directory: {error}"))
}

fn resolve_reports_root(results_root: &Path) -> Result<PathBuf, String> {
    let configured = std::env::var_os("SUBTITLE_REPORTS_ROOT")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            results_root
                .parent()
                .unwrap_or(results_root)
                .join("reports")
        });
    fs::create_dir_all(&configured)
        .map_err(|error| format!("Could not create the reports directory: {error}"))?;
    configured
        .canonicalize()
        .map_err(|error| format!("Could not resolve the reports directory: {error}"))
}

fn persist_pipeline_log(
    reports_root: &Path,
    lines: &[String],
    code: i32,
) -> Result<PathBuf, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("Could not read the system clock: {error}"))?
        .as_secs();
    let mut content = format!("GeneratedUnix={timestamp}\nExitCode={code}\n");
    if !lines.is_empty() {
        content.push_str(&lines.join("\n"));
        content.push('\n');
    }
    let dated = reports_root.join(format!("pipeline-{timestamp}.log"));
    fs::write(&dated, content.as_bytes())
        .map_err(|error| format!("Could not write the pipeline log: {error}"))?;
    fs::write(reports_root.join("pipeline-latest.log"), content.as_bytes())
        .map_err(|error| format!("Could not write the latest pipeline log: {error}"))?;
    Ok(dated)
}

fn validate_video(value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value)
        .canonicalize()
        .map_err(|error| format!("Could not open the video path: {error}"))?;
    if !path.is_file() {
        return Err("The selected video path is not a file.".to_string());
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !VIDEO_EXTENSIONS.contains(&extension.as_str()) {
        return Err("Select a supported video file.".to_string());
    }
    Ok(path)
}

fn stage_for_line(line: &str) -> Option<PipelineStage> {
    if line.contains("Copying the video to the isolated workspace") {
        Some(PipelineStage {
            key: "prepare",
            label: "Preparing video",
            percent: None,
        })
    } else if line.contains("1/2 VideoSubFinder") {
        Some(PipelineStage {
            key: "vsf",
            label: "Scanning subtitle regions",
            percent: None,
        })
    } else if line.contains("2/2 RapidVideOCR") {
        Some(PipelineStage {
            key: "ocr",
            label: "Recognizing text",
            percent: None,
        })
    } else if line.contains("RESULT START") {
        Some(PipelineStage {
            key: "finalize",
            label: "Preparing results",
            percent: None,
        })
    } else if line == "Pipeline=COMPLETE" {
        Some(PipelineStage {
            key: "complete",
            label: "Subtitles ready",
            percent: Some(100),
        })
    } else {
        None
    }
}

fn spawn_reader<R: Read + Send + 'static>(
    reader: R,
    app: AppHandle,
    lines: Arc<Mutex<Vec<String>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            match reader.read_until(b'\n', &mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buffer)
                        .replace('\0', "")
                        .trim_end_matches(|character| character == '\r' || character == '\n')
                        .to_string();
                    if line.is_empty() {
                        continue;
                    }
                    if let Some(stage) = stage_for_line(&line) {
                        let _ = app.emit("pipeline-stage", stage);
                    }
                    let _ = app.emit("pipeline-output", line.clone());
                    if let Ok(mut captured) = lines.lock() {
                        captured.push(line);
                    }
                }
            }
        }
    })
}

fn parse_summary(lines: &[String]) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let mut inside = false;
    for line in lines {
        if line.contains("SUBHOOPER PIPELINE") && line.ends_with("RESULT START ---") {
            inside = true;
            continue;
        }
        if inside && line.contains("RESULT END ---") {
            break;
        }
        if !inside {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            result.insert(key.to_string(), value.to_string());
        }
    }
    result
}

fn pipeline_failure_message(lines: &[String], code: i32) -> String {
    let detail = lines
        .iter()
        .rev()
        .find(|line| line.starts_with("ERROR:"))
        .or_else(|| lines.iter().rev().find(|line| !line.trim().is_empty()))
        .map(|line| line.trim_start_matches("ERROR:").trim());
    match detail {
        Some(value) if !value.is_empty() => {
            format!("{value} (pipeline code: {code})")
        }
        _ => format!("Pipeline exited with code {code}"),
    }
}

fn clear_active_pid(app: &AppHandle, pid: u32) {
    let state = app.state::<PipelineState>();
    if let Ok(mut active) = state.active_pid.lock() {
        if active.is_some_and(|value| value == pid || value == 0) {
            *active = None;
        }
    };
}

fn kill_process_tree(pid: u32) -> Result<(), String> {
    let pid_text = pid.to_string();
    let mut command = Command::new("taskkill");
    command
        .args(["/PID", pid_text.as_str(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    let status = command
        .status()
        .map_err(|error| format!("Could not run the cancellation command: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("Could not terminate the process tree.".to_string())
    }
}

fn prepare_srt_for_save(content: &str) -> Result<String, String> {
    if content.is_empty() || content.len() > 20 * 1024 * 1024 {
        return Err("SRT content is empty or exceeds the size limit.".to_string());
    }
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.contains(" --> ") {
        return Err("No SRT timecode was found.".to_string());
    }
    Ok(if normalized.ends_with('\n') {
        normalized
    } else {
        normalized + "\n"
    })
}

fn parse_srt_cues(content: &str) -> Result<Vec<SubtitleCue>, String> {
    let normalized = prepare_srt_for_save(content)?;
    let mut cues = Vec::new();
    for block in normalized.trim().split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        let time_index = lines
            .iter()
            .position(|line| line.contains(" --> "))
            .ok_or("No timecode was found in an SRT block.")?;
        let (start, end) = lines[time_index]
            .split_once(" --> ")
            .ok_or("The SRT timecode is invalid.")?;
        let text = lines[time_index + 1..]
            .iter()
            .map(|line| (*line).to_string())
            .collect();
        cues.push(SubtitleCue {
            start: start.to_string(),
            end: end.to_string(),
            text,
        });
    }
    if cues.is_empty() {
        return Err("No subtitles are available for export.".to_string());
    }
    Ok(cues)
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn export_content(content: &str, format: &str) -> Result<String, String> {
    if format == "srt" {
        return prepare_srt_for_save(content);
    }
    let cues = parse_srt_cues(content)?;
    match format {
        "txt" => Ok(cues
            .iter()
            .map(|cue| cue.text.join(" "))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"),
        "md" => {
            let mut output = String::from("# Subtitles\n\n");
            for cue in cues {
                output.push_str(&format!(
                    "## {} → {}\n\n{}\n\n",
                    cue.start,
                    cue.end,
                    cue.text.join("  \n")
                ));
            }
            Ok(output)
        }
        "ttml" => {
            let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<tt xmlns=\"http://www.w3.org/ns/ttml\"><body><div>\n");
            for cue in cues {
                let text = cue
                    .text
                    .iter()
                    .map(|line| escape_xml(line))
                    .collect::<Vec<_>>()
                    .join("<br/>");
                output.push_str(&format!(
                    "  <p begin=\"{}\" end=\"{}\">{}</p>\n",
                    cue.start.replace(',', "."),
                    cue.end.replace(',', "."),
                    text
                ));
            }
            output.push_str("</div></body></tt>\n");
            Ok(output)
        }
        _ => Err(format!("Invalid export format: {format}")),
    }
}

fn ai_storage_root(app: &AppHandle) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_local_data_dir()
        .map_err(|error| format!("Could not resolve the AI data directory: {error}"))?
        .join("ai");
    fs::create_dir_all(&root)
        .map_err(|error| format!("Could not create the AI data directory: {error}"))?;
    Ok(root)
}

fn ai_model_spec(model_id: &str) -> Result<AiModelSpec, String> {
    AI_MODELS
        .iter()
        .copied()
        .find(|model| model.id == model_id)
        .ok_or_else(|| format!("Unknown AI model: {model_id}"))
}

fn find_file(root: &Path, filename: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case(filename))
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file(&path, filename) {
                return Some(found);
            }
        }
    }
    None
}

fn find_gguf(root: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_gguf(&path) {
                return Some(found);
            }
        }
    }
    None
}

fn ai_model_path(root: &Path, model_id: &str) -> Option<PathBuf> {
    find_gguf(&root.join("models").join(model_id))
}

fn ai_model_info(root: &Path, model: AiModelSpec) -> AiModelInfo {
    AiModelInfo {
        id: model.id.to_string(),
        name: model.name.to_string(),
        size_label: model.size_label.to_string(),
        memory_label: model.memory_label.to_string(),
        recommendation: model.recommendation.to_string(),
        installed: ai_model_path(root, model.id).is_some(),
    }
}

fn begin_ai_task(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<AiState>();
    state.cancel_requested.store(false, Ordering::SeqCst);
    let mut active = lock(&state.active_pid)?;
    if active.is_some() {
        return Err("Another AI task is already running.".to_string());
    }
    *active = Some(0);
    Ok(())
}

fn end_ai_task(app: &AppHandle) {
    let state = app.state::<AiState>();
    if let Ok(mut active) = state.active_pid.lock() {
        *active = None;
    };
}

fn ensure_ai_not_cancelled(app: &AppHandle) -> Result<(), String> {
    if app
        .state::<AiState>()
        .cancel_requested
        .load(Ordering::SeqCst)
    {
        Err("AI processing was cancelled.".to_string())
    } else {
        Ok(())
    }
}

fn emit_ai_progress(
    app: &AppHandle,
    phase: &str,
    label: impl Into<String>,
    received: u64,
    total: Option<u64>,
) {
    let _ = app.emit(
        "ai-progress",
        AiProgress {
            phase: phase.to_string(),
            label: label.into(),
            received,
            total,
        },
    );
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("Could not open the downloaded file: {error}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("Could not verify the downloaded file: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn download_verified(
    app: &AppHandle,
    client: &reqwest::blocking::Client,
    url: &str,
    destination: &Path,
    expected_sha256: &str,
    phase: &str,
    label: &str,
) -> Result<(), String> {
    if destination.is_file() && sha256_file(destination)?.eq_ignore_ascii_case(expected_sha256) {
        return Ok(());
    }
    if expected_sha256.len() != 64
        || !expected_sha256
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err("The official download did not provide a valid SHA-256 digest.".to_string());
    }
    let parent = destination
        .parent()
        .ok_or("The download destination is invalid.")?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("Could not create the download directory: {error}"))?;
    let partial = destination.with_extension("download");
    if partial.exists() {
        fs::remove_file(&partial)
            .map_err(|error| format!("Could not clear the incomplete download: {error}"))?;
    }
    let mut response = client
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Download failed: {error}"))?;
    let total = response.content_length();
    let mut output = fs::File::create(&partial)
        .map_err(|error| format!("Could not create the download file: {error}"))?;
    let mut hasher = Sha256::new();
    let mut received = 0_u64;
    let mut buffer = [0_u8; 256 * 1024];
    loop {
        ensure_ai_not_cancelled(app)?;
        let read = response
            .read(&mut buffer)
            .map_err(|error| format!("Could not read the download: {error}"))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| format!("Could not write the download: {error}"))?;
        hasher.update(&buffer[..read]);
        received += read as u64;
        emit_ai_progress(app, phase, label, received, total);
    }
    output
        .sync_all()
        .map_err(|error| format!("Could not finish the download: {error}"))?;
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        let _ = fs::remove_file(&partial);
        return Err(format!("SHA-256 verification failed for {label}."));
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(|error| {
            format!("Could not replace the previous verified download: {error}")
        })?;
    }
    fs::rename(&partial, destination)
        .map_err(|error| format!("Could not install the verified download: {error}"))?;
    Ok(())
}

fn release_cpu_asset(release: &GithubRelease) -> Option<GithubAsset> {
    release
        .assets
        .iter()
        .find(|asset| {
            let name = asset.name.to_ascii_lowercase();
            name.ends_with("bin-win-cpu-x64.zip")
        })
        .cloned()
}

fn fetch_llama_asset(client: &reqwest::blocking::Client) -> Result<GithubAsset, String> {
    let latest_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
    let latest: GithubRelease = client
        .get(latest_url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Could not check the official llama.cpp release: {error}"))?
        .json()
        .map_err(|error| format!("Could not read the llama.cpp release manifest: {error}"))?;
    if let Some(asset) = release_cpu_asset(&latest) {
        return Ok(asset);
    }
    let tag_asset = latest
        .assets
        .iter()
        .find(|asset| asset.name.eq_ignore_ascii_case("nightly-tag.txt"))
        .ok_or("The official llama.cpp release does not contain a Windows CPU runtime.")?;
    let tag = client
        .get(&tag_asset.browser_download_url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Could not resolve the llama.cpp runtime tag: {error}"))?
        .text()
        .map_err(|error| format!("Could not read the llama.cpp runtime tag: {error}"))?;
    let tag = tag.trim();
    if tag.is_empty()
        || !tag
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || ".-_".contains(character))
    {
        return Err("The llama.cpp runtime tag is invalid.".to_string());
    }
    let tagged_url = format!("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}");
    let tagged: GithubRelease = client
        .get(tagged_url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Could not check the tagged llama.cpp release: {error}"))?
        .json()
        .map_err(|error| format!("Could not read the tagged llama.cpp manifest: {error}"))?;
    release_cpu_asset(&tagged).ok_or_else(|| {
        "The tagged llama.cpp release does not contain a Windows CPU runtime.".to_string()
    })
}

fn extract_runtime(
    app: &AppHandle,
    archive: &Path,
    runtime_root: &Path,
) -> Result<PathBuf, String> {
    let staging = runtime_root.join("staging");
    let installed = runtime_root.join("current");
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|error| {
            format!("Could not clear the AI runtime staging directory: {error}")
        })?;
    }
    fs::create_dir_all(&staging)
        .map_err(|error| format!("Could not create the AI runtime staging directory: {error}"))?;
    let file = fs::File::open(archive)
        .map_err(|error| format!("Could not open the AI runtime archive: {error}"))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|error| format!("Could not read the AI runtime archive: {error}"))?;
    for index in 0..zip.len() {
        ensure_ai_not_cancelled(app)?;
        let mut entry = zip
            .by_index(index)
            .map_err(|error| format!("Could not read an AI runtime archive entry: {error}"))?;
        let relative = entry
            .enclosed_name()
            .ok_or("The AI runtime archive contains an unsafe path.")?;
        let destination = staging.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&destination)
                .map_err(|error| format!("Could not create an AI runtime directory: {error}"))?;
            continue;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Could not create an AI runtime directory: {error}"))?;
        }
        let mut output = fs::File::create(&destination)
            .map_err(|error| format!("Could not extract an AI runtime file: {error}"))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|error| format!("Could not extract the AI runtime: {error}"))?;
    }
    let cli = find_file(&staging, "llama-cli.exe")
        .ok_or("The verified AI runtime does not contain llama-cli.exe.")?;
    if installed.exists() {
        fs::remove_dir_all(&installed)
            .map_err(|error| format!("Could not replace the previous AI runtime: {error}"))?;
    }
    fs::rename(&staging, &installed)
        .map_err(|error| format!("Could not install the AI runtime: {error}"))?;
    let relative = cli
        .strip_prefix(runtime_root.join("staging"))
        .map_err(|_| "Could not resolve the installed AI runtime path.".to_string())?;
    Ok(installed.join(relative))
}

fn ensure_llama_runtime(
    app: &AppHandle,
    client: &reqwest::blocking::Client,
    root: &Path,
) -> Result<PathBuf, String> {
    let runtime_root = root.join("runtime");
    let installed = runtime_root.join("current");
    if let Some(cli) = find_file(&installed, "llama-cli.exe") {
        return Ok(cli);
    }
    emit_ai_progress(app, "runtime", "Checking the local AI engine", 0, None);
    let asset = fetch_llama_asset(client)?;
    let digest = asset
        .digest
        .as_deref()
        .and_then(|value| value.strip_prefix("sha256:"))
        .ok_or("The official llama.cpp asset does not provide a SHA-256 digest.")?;
    let archive = runtime_root.join("downloads").join(&asset.name);
    download_verified(
        app,
        client,
        &asset.browser_download_url,
        &archive,
        digest,
        "runtime",
        "Downloading the local AI engine",
    )?;
    emit_ai_progress(app, "runtime", "Installing the local AI engine", 1, Some(1));
    extract_runtime(app, &archive, &runtime_root)
}

fn model_download_url(repository: &str, file_path: &str) -> Result<String, String> {
    let mut url = reqwest::Url::parse(&format!("https://huggingface.co/{repository}/resolve/main"))
        .map_err(|error| format!("Could not construct the model download URL: {error}"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "Could not construct the model download path.".to_string())?;
        for segment in file_path.split('/').filter(|segment| !segment.is_empty()) {
            segments.push(segment);
        }
    }
    url.query_pairs_mut().append_pair("download", "true");
    Ok(url.to_string())
}

fn fetch_model_file(
    client: &reqwest::blocking::Client,
    model: AiModelSpec,
) -> Result<(String, String, String), String> {
    let tree_url = format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=true&expand=false",
        model.repository
    );
    let entries: Vec<HuggingFaceTreeEntry> = client
        .get(tree_url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Could not check the official model manifest: {error}"))?
        .json()
        .map_err(|error| format!("Could not read the official model manifest: {error}"))?;
    let entry = entries
        .into_iter()
        .filter(|entry| {
            let path = entry.path.to_ascii_lowercase();
            entry.entry_type == "file" && path.ends_with(".gguf") && path.contains("q4_k_m")
        })
        .min_by_key(|entry| entry.path.len())
        .ok_or("The official model repository does not contain a Q4_K_M GGUF file.")?;
    let digest = entry
        .lfs
        .as_ref()
        .map(|lfs| lfs.oid.clone())
        .ok_or("The official model file does not provide a SHA-256 LFS digest.")?;
    let url = model_download_url(model.repository, &entry.path)?;
    Ok((entry.path, url, digest))
}

fn install_ai_model_blocking(app: AppHandle, model_id: String) -> Result<AiModelInfo, String> {
    begin_ai_task(&app)?;
    let result = (|| {
        let root = ai_storage_root(&app)?;
        let model = ai_model_spec(&model_id)?;
        let client = reqwest::blocking::Client::builder()
            .user_agent(AI_USER_AGENT)
            .timeout(Duration::from_secs(60 * 60 * 8))
            .build()
            .map_err(|error| format!("Could not initialize secure downloads: {error}"))?;
        ensure_llama_runtime(&app, &client, &root)?;
        if ai_model_path(&root, model.id).is_none() {
            emit_ai_progress(&app, "model", format!("Checking {}", model.name), 0, None);
            let (relative_path, url, digest) = fetch_model_file(&client, model)?;
            let filename = Path::new(&relative_path)
                .file_name()
                .ok_or("The model filename is invalid.")?;
            let destination = root.join("models").join(model.id).join(filename);
            download_verified(
                &app,
                &client,
                &url,
                &destination,
                &digest,
                "model",
                &format!("Downloading {}", model.name),
            )?;
        }
        emit_ai_progress(
            &app,
            "complete",
            format!("{} is ready", model.name),
            1,
            Some(1),
        );
        Ok(ai_model_info(&root, model))
    })();
    end_ai_task(&app);
    result
}

fn repair_smart_quote_terminators(value: &str) -> Option<String> {
    let characters = value.chars().collect::<Vec<_>>();
    let mut repaired = String::with_capacity(value.len() + 8);
    let mut inside_string = false;
    let mut escaped = false;
    let mut changed = false;
    for (index, character) in characters.iter().copied().enumerate() {
        repaired.push(character);
        if !inside_string {
            if character == '"' {
                inside_string = true;
            }
            continue;
        }
        if escaped {
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if character == '"' {
            inside_string = false;
            continue;
        }
        if matches!(character, '”' | '“' | '＂')
            && characters[index + 1..]
                .iter()
                .copied()
                .find(|next| !next.is_whitespace())
                == Some('}')
        {
            repaired.push('"');
            inside_string = false;
            changed = true;
        }
    }
    changed.then_some(repaired)
}

fn parse_ai_json_candidate(value: &str) -> Option<Vec<AiCueOutput>> {
    serde_json::from_str(value).ok().or_else(|| {
        repair_smart_quote_terminators(value)
            .and_then(|repaired| serde_json::from_str(&repaired).ok())
    })
}

fn clean_model_output(value: &str) -> Result<Vec<AiCueOutput>, String> {
    let without_thinking = if let Some(end) = value.rfind("</think>") {
        &value[end + "</think>".len()..]
    } else {
        value
    };
    if let Some(end) = without_thinking.rfind(']') {
        for (start, character) in without_thinking[..=end].char_indices().rev() {
            if character != '[' {
                continue;
            }
            if let Some(parsed) = parse_ai_json_candidate(&without_thinking[start..=end]) {
                return Ok(parsed);
            }
        }
        return Err("The model returned invalid subtitle data.".to_string());
    }
    if let Some(start) = without_thinking.find('[') {
        let mut candidate = without_thinking[start..].trim_end();
        if let Some(without_comma) = candidate.strip_suffix(',') {
            candidate = without_comma.trim_end();
        }
        if candidate.ends_with('}') {
            let repaired = format!("{candidate}]");
            if let Some(parsed) = parse_ai_json_candidate(&repaired) {
                return Ok(parsed);
            }
        }
    }
    Err("The model response was incomplete.".to_string())
}

fn assistant_output(value: &str) -> &str {
    for marker in [
        "\r\nAssistant:\r\n",
        "\nAssistant:\n",
        "Assistant:\r\n",
        "Assistant:\n",
    ] {
        if let Some((_, response)) = value.rsplit_once(marker) {
            return response.trim();
        }
    }
    value.trim()
}

fn ai_engine_generation_error(stdout: &str, stderr: &str) -> Option<String> {
    [stderr, stdout]
        .into_iter()
        .flat_map(str::lines)
        .map(str::trim)
        .find(|line| {
            line.contains("Failed to initialize samplers")
                || line.contains("Unexpected empty grammar stack")
                || line.contains("failed to launch slot with task")
        })
        .map(|line| format!("The local AI engine could not initialize generation: {line}"))
}

fn ai_log_excerpt(value: &str) -> String {
    const LIMIT: usize = 32_000;
    let mut excerpt = value.chars().take(LIMIT).collect::<String>();
    if value.chars().count() > LIMIT {
        excerpt.push_str("\n[truncated]\n");
    }
    excerpt
}

fn persist_ai_diagnostic(
    _app: &AppHandle,
    cli: &Path,
    model_path: &Path,
    output_file: &str,
    stdout: &str,
    stderr: &str,
    error: &str,
) -> Option<PathBuf> {
    let project = resolve_project_root().ok()?;
    let results_root = resolve_results_root(&project).ok()?;
    let reports_root = resolve_reports_root(&results_root).ok()?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_millis();
    let content = format!(
        "SubHooper=0.3.7-beta\nGeneratedUnixMs={timestamp}\nError={error}\nCLI={}\nModel={}\n\n--- OUTPUT FILE ---\n{}\n\n--- STDOUT ---\n{}\n\n--- STDERR ---\n{}\n",
        cli.display(),
        model_path.display(),
        ai_log_excerpt(output_file),
        ai_log_excerpt(stdout),
        ai_log_excerpt(stderr),
    );
    let dated = reports_root.join(format!("ai-{timestamp}.log"));
    fs::write(&dated, content.as_bytes()).ok()?;
    fs::write(reports_root.join("ai-latest.log"), content.as_bytes()).ok()?;
    Some(dated.canonicalize().unwrap_or(dated))
}

fn parse_ai_process_output(
    app: &AppHandle,
    cli: &Path,
    model_path: &Path,
    output_file: &str,
    stdout: &str,
    stderr: &str,
) -> Result<Vec<AiCueOutput>, String> {
    let transcript = if output_file.trim().is_empty() {
        stdout
    } else {
        output_file
    };
    match clean_model_output(assistant_output(transcript)) {
        Ok(parsed) => Ok(parsed),
        Err(error) => {
            let diagnostic =
                persist_ai_diagnostic(app, cli, model_path, output_file, stdout, stderr, &error);
            if let Some(path) = diagnostic {
                Err(format!("{error} Diagnostic log: {}", path.display()))
            } else {
                Err(error)
            }
        }
    }
}

fn ai_validation_error(
    app: &AppHandle,
    cli: &Path,
    model_path: &Path,
    response: &[AiCueOutput],
    error: String,
) -> String {
    let response = serde_json::to_string_pretty(response).unwrap_or_default();
    if let Some(path) = persist_ai_diagnostic(app, cli, model_path, &response, "", "", &error) {
        format!("{error} Diagnostic log: {}", path.display())
    } else {
        error
    }
}

fn validate_translation_language(value: Option<&str>) -> Result<&str, String> {
    let language = value.ok_or("Select a translation language.")?.trim();
    if language.is_empty() || language.len() > 64 || language.chars().any(char::is_control) {
        return Err("The translation language is invalid.".to_string());
    }
    Ok(language)
}

fn cue_payload(cues: &[(usize, &SubtitleCue)]) -> Vec<serde_json::Value> {
    cues.iter()
        .map(|(id, cue)| serde_json::json!({"id": id, "text": cue.text.join("\n")}))
        .collect()
}

fn validate_cleanup_language(value: Option<&str>) -> Result<&str, String> {
    let language = value.unwrap_or("Auto detect").trim();
    if language.is_empty() || language.len() > 64 || language.chars().any(char::is_control) {
        return Err("The cleanup language is invalid.".to_string());
    }
    Ok(language)
}

fn validate_cleanup_strength(value: Option<&str>) -> Result<&str, String> {
    match value.unwrap_or("balanced").trim() {
        "light" => Ok("light"),
        "balanced" => Ok("balanced"),
        "strong" => Ok("strong"),
        _ => Err("Select a valid cleanup strength.".to_string()),
    }
}

fn ai_prompt(
    targets: &[(usize, &SubtitleCue)],
    context_before: &[(usize, &SubtitleCue)],
    context_after: &[(usize, &SubtitleCue)],
    document_samples: &[(usize, &SubtitleCue)],
    previous_results: &[(usize, String)],
    mode: &str,
    source_language: Option<&str>,
    target_language: Option<&str>,
    cleanup_language: Option<&str>,
    cleanup_strength: Option<&str>,
    guidance: Option<&str>,
) -> Result<String, String> {
    let guidance = guidance.unwrap_or("").trim();
    if guidance.chars().count() > 4_000 {
        return Err("Translation guidance must be 4,000 characters or fewer.".to_string());
    }
    let task = match mode {
        "clean" => {
            let language = validate_cleanup_language(cleanup_language)?;
            let strength = validate_cleanup_strength(cleanup_strength)?;
            let language_instruction = if language == "Auto detect" {
                "Infer the dominant subtitle language from document_samples and use that language consistently for every kept target cue.".to_string()
            } else {
                format!("Use {language} as the cleanup language for every kept target cue.")
            };
            let strength_instruction = match strength {
                "light" => "Use light cleanup. Correct clear OCR, spelling, punctuation, capitalization, broken-word, and line-break errors. Translate coherent foreign dialogue into the cleanup language, but remove a cue only when it is unmistakably non-text noise. Preserve uncertain fragments for manual review.",
                "balanced" => "Use balanced cleanup. Correct OCR and subtitle formatting, translate coherent foreign dialogue into the cleanup language, and remove high-confidence non-dialogue OCR noise. Preserve ambiguous material when it may still be meaningful dialogue.",
                "strong" => "Use strong cleanup. Produce a clean, single-language subtitle track. Translate coherent foreign dialogue into the cleanup language. Remove isolated symbols, short numeric or mixed-character debris, corrupted foreign-script fragments that cannot be translated confidently, scenery text, credits, logos, interface text, and fragments unrelated to dialogue. Do not keep meaningless text merely because it is uncertain. Preserve short text only when nearby dialogue makes its meaning clear. Never invent dialogue that is absent from the OCR input.",
                _ => unreachable!(),
            };
            format!(
                "{language_instruction} {strength_instruction} For any cue selected for removal, return its id with an empty text string. Use document_samples and surrounding cues only to determine language, relevance, and context."
            )
        }
        "translate" => {
            let language = validate_translation_language(target_language)?;
            let source_hint = source_language
                .map(str::trim)
                .filter(|value| !value.is_empty() && *value != "Auto detect")
                .map(|value| format!(" The declared source language is {value}."))
                .unwrap_or_default();
            return Ok(format!(
                "/no_think\nTranslate only target_cues into {language}.{source_hint} Use context_before, context_after, previous_results, and guidance to preserve scene meaning, pronouns, names, relationships, register, recurring terminology, and concise natural subtitle phrasing. Treat previous_results as the preferred terminology for continuity. Silently check every translation for natural phrasing and consistency before answering. Context entries are reference material only and must not be returned. The output ids must exactly equal target_ids; never return an id from context_before, context_after, or previous_results. Return only one valid JSON array of objects for target_cues with the exact input id and a text string. Encode subtitle line breaks as \\n inside text strings. Do not use Markdown, omit, merge, reorder, explain, or add cues. Payload:\n{}",
                serde_json::to_string(&serde_json::json!({
                    "target_ids": targets.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    "guidance": guidance,
                    "document_samples": cue_payload(document_samples),
                    "context_before": cue_payload(context_before),
                    "previous_results": previous_results.iter().map(|(id, text)| serde_json::json!({"id": id, "text": text})).collect::<Vec<_>>(),
                    "target_cues": cue_payload(targets),
                    "context_after": cue_payload(context_after),
                })).map_err(|error| format!("Could not prepare subtitles for the model: {error}"))?
            ));
        }
        _ => return Err("Select a valid AI task.".to_string()),
    };
    Ok(format!(
        "/no_think\n{task} Return one object for every target id even when text is empty to mark high-confidence noise for removal. The output ids must exactly equal target_ids; never return an id from document_samples, context_before, context_after, or previous_results. Return only one valid JSON array of objects for target_cues with the exact input id and a text string. Encode subtitle line breaks as \\n inside text strings. Do not use Markdown, omit, merge, reorder, explain, or add cue objects. Payload:\n{}",
        serde_json::to_string(&serde_json::json!({
            "target_ids": targets.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "guidance": guidance,
            "cleanup_language": validate_cleanup_language(cleanup_language)?,
            "cleanup_strength": validate_cleanup_strength(cleanup_strength)?,
            "document_samples": cue_payload(document_samples),
            "context_before": cue_payload(context_before),
            "previous_results": previous_results.iter().map(|(id, text)| serde_json::json!({"id": id, "text": text})).collect::<Vec<_>>(),
            "target_cues": cue_payload(targets),
            "context_after": cue_payload(context_after),
        })).map_err(|error| format!("Could not prepare subtitles for the model: {error}"))?
    ))
}

fn run_ai_chunk(
    app: &AppHandle,
    cli: &Path,
    model_path: &Path,
    prompt: &str,
) -> Result<Vec<AiCueOutput>, String> {
    ensure_ai_not_cancelled(app)?;
    let prompt_file = TemporaryPrompt::create(prompt)?;
    let output_file = TemporaryOutput::create()?;
    let mut command = Command::new(cli);
    command
        .arg("-m")
        .arg(model_path)
        .arg("-f")
        .arg(prompt_file.path())
        .args([
            "-n",
            "4096",
            "-c",
            "8192",
            "--jinja",
            "--single-turn",
            "--no-escape",
            "--reasoning",
            "off",
            "--temp",
            "0.2",
            "--top-k",
            "20",
            "--top-p",
            "0.8",
            "--min-p",
            "0",
            "--presence-penalty",
            "0",
            "--no-display-prompt",
            "--no-show-timings",
            "--log-verbosity",
            "2",
            "--color",
            "off",
            "--simple-io",
        ])
        .arg("--output-file")
        .arg(output_file.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    let child = command
        .spawn()
        .map_err(|error| format!("Could not start the local AI engine: {error}"))?;
    let pid = child.id();
    *lock(&app.state::<AiState>().active_pid)? = Some(pid);
    if app
        .state::<AiState>()
        .cancel_requested
        .load(Ordering::SeqCst)
    {
        let _ = kill_process_tree(pid);
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Could not wait for the local AI engine: {error}"))?;
    *lock(&app.state::<AiState>().active_pid)? = Some(0);
    ensure_ai_not_cancelled(app)?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let file_output = fs::read_to_string(output_file.path()).unwrap_or_default();
    let generation_error = ai_engine_generation_error(&stdout, &stderr);
    if !output.status.success() || generation_error.is_some() {
        let detail = stderr.trim().chars().take(1000).collect::<String>();
        let error =
            generation_error.unwrap_or_else(|| format!("The local AI engine failed: {detail}"));
        let diagnostic =
            persist_ai_diagnostic(app, cli, model_path, &file_output, &stdout, &stderr, &error);
        return Err(if let Some(path) = diagnostic {
            format!("{error} Diagnostic log: {}", path.display())
        } else {
            error
        });
    }
    parse_ai_process_output(app, cli, model_path, &file_output, &stdout, &stderr)
}

fn indexed_cues(cues: &[SubtitleCue], start: usize, end: usize) -> Vec<(usize, &SubtitleCue)> {
    cues[start..end]
        .iter()
        .enumerate()
        .map(|(offset, cue)| (start + offset + 1, cue))
        .collect()
}

fn document_sample_cues(cues: &[SubtitleCue]) -> Vec<(usize, &SubtitleCue)> {
    let indexed = indexed_cues(cues, 0, cues.len());
    let meaningful = indexed
        .iter()
        .copied()
        .filter(|(_, cue)| {
            cue.text
                .iter()
                .flat_map(|line| line.chars())
                .filter(|character| character.is_alphabetic())
                .count()
                >= 3
        })
        .collect::<Vec<_>>();
    let candidates = if meaningful.is_empty() {
        indexed
    } else {
        meaningful
    };
    if candidates.len() <= AI_DOCUMENT_SAMPLE_CUES {
        return candidates;
    }
    (0..AI_DOCUMENT_SAMPLE_CUES)
        .map(|sample| {
            let index = sample * (candidates.len() - 1) / (AI_DOCUMENT_SAMPLE_CUES - 1);
            candidates[index]
        })
        .collect()
}

fn compact_cue_shape(cue: &SubtitleCue) -> (usize, usize, usize) {
    cue.text
        .iter()
        .flat_map(|line| line.chars())
        .filter(|character| !character.is_whitespace())
        .fold((0, 0, 0), |(visible, alphabetic, numeric), character| {
            (
                visible + 1,
                alphabetic + usize::from(character.is_alphabetic()),
                numeric + usize::from(character.is_numeric()),
            )
        })
}

fn weak_ocr_fragment(cue: &SubtitleCue) -> bool {
    let (visible, alphabetic, _) = compact_cue_shape(cue);
    visible <= 4 && alphabetic <= 1
}

fn strong_cleanup_prefilter(cues: &[SubtitleCue], index: usize) -> bool {
    let (visible, alphabetic, numeric) = compact_cue_shape(&cues[index]);
    if visible == 0 || alphabetic == 0 && numeric == 0 {
        return true;
    }
    let nearby_weak_fragment = index
        .checked_sub(1)
        .is_some_and(|previous| weak_ocr_fragment(&cues[previous]))
        || cues.get(index + 1).is_some_and(weak_ocr_fragment);
    nearby_weak_fragment && visible <= 4 && alphabetic <= 1 && (numeric > 0 || alphabetic == 0)
}

fn render_ai_srt(
    cues: &[SubtitleCue],
    results: &BTreeMap<usize, String>,
) -> Result<(String, usize), String> {
    let mut output = String::new();
    let mut output_index = 1;
    for (index, cue) in cues.iter().enumerate() {
        let text = results
            .get(&(index + 1))
            .ok_or("The AI result omitted a subtitle cue decision.")?;
        if text.trim().is_empty() {
            continue;
        }
        output.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            output_index, cue.start, cue.end, text
        ));
        output_index += 1;
    }
    Ok((output, output_index - 1))
}

fn select_ai_target_response(
    response: &[AiCueOutput],
    targets: &[(usize, &SubtitleCue)],
    context_before: &[(usize, &SubtitleCue)],
    context_after: &[(usize, &SubtitleCue)],
    document_samples: &[(usize, &SubtitleCue)],
    allow_empty_text: bool,
) -> Result<Vec<AiCueOutput>, String> {
    let is_target = |id: usize| targets.iter().any(|(target_id, _)| *target_id == id);
    let is_context = |id: usize| {
        context_before
            .iter()
            .chain(context_after.iter())
            .chain(document_samples.iter())
            .any(|(context_id, _)| *context_id == id)
    };
    let mut selected = Vec::with_capacity(targets.len());
    for item in response {
        if is_target(item.id) {
            if selected
                .iter()
                .any(|existing: &AiCueOutput| existing.id == item.id)
            {
                return Err(format!(
                    "The model returned target cue {} more than once. No partial result was saved.",
                    item.id
                ));
            }
            selected.push(item.clone());
        } else if !is_context(item.id) {
            return Err(format!(
                "The model returned unexpected cue {} outside the target and reference context. No partial result was saved.",
                item.id
            ));
        }
    }
    if selected.len() != targets.len() {
        return Err(format!(
            "The model returned {} target cues for a group containing {}. No partial result was saved.",
            selected.len(),
            targets.len()
        ));
    }
    for (expected, item) in targets.iter().zip(&selected) {
        if item.id != expected.0 {
            return Err(
                "The model changed target cue order. No partial result was saved.".to_string(),
            );
        }
        if !allow_empty_text && item.text.trim().is_empty() {
            return Err(
                "The model returned empty translated subtitle text. No partial result was saved."
                    .to_string(),
            );
        }
    }
    Ok(selected)
}

fn should_subdivide_ai_range(error: &str, target_count: usize) -> bool {
    error.starts_with("The model ") && target_count > 1
}

fn process_local_ai_range(
    app: &AppHandle,
    cli: &Path,
    model_path: &Path,
    cues: &[SubtitleCue],
    document_samples: &[(usize, &SubtitleCue)],
    start: usize,
    end: usize,
    results: &mut BTreeMap<usize, String>,
    request: &AiRequest,
    chunk_index: usize,
    total_chunks: usize,
) -> Result<(), String> {
    ensure_ai_not_cancelled(app)?;
    let before_start = start.saturating_sub(AI_CONTEXT_CUES);
    let after_end = (end + AI_CONTEXT_CUES).min(cues.len());
    let targets = indexed_cues(cues, start, end);
    let context_before = indexed_cues(cues, before_start, start);
    let context_after = indexed_cues(cues, end, after_end);
    let previous_results = results
        .range(before_start + 1..start + 1)
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(id, text)| (*id, text.clone()))
        .collect::<Vec<_>>();
    let strong_cleanup = request.mode == "clean"
        && validate_cleanup_strength(request.cleanup_strength.as_deref())? == "strong";
    let mut model_targets = Vec::with_capacity(targets.len());
    for (id, cue) in targets {
        if strong_cleanup && strong_cleanup_prefilter(cues, id - 1) {
            results.insert(id, String::new());
        } else {
            model_targets.push((id, cue));
        }
    }
    if model_targets.is_empty() {
        return Ok(());
    }
    let prompt = ai_prompt(
        &model_targets,
        &context_before,
        &context_after,
        document_samples,
        &previous_results,
        &request.mode,
        request.source_language.as_deref(),
        request.target_language.as_deref(),
        request.cleanup_language.as_deref(),
        request.cleanup_strength.as_deref(),
        request.guidance.as_deref(),
    )?;
    let attempt = run_ai_chunk(app, cli, model_path, &prompt).and_then(|response| {
        select_ai_target_response(
            &response,
            &model_targets,
            &context_before,
            &context_after,
            document_samples,
            request.mode == "clean",
        )
        .map_err(|error| ai_validation_error(app, cli, model_path, &response, error))
    });
    match attempt {
        Ok(selected) => {
            for item in selected {
                results.insert(item.id, item.text.trim().to_string());
            }
            Ok(())
        }
        Err(error) if should_subdivide_ai_range(&error, model_targets.len()) => {
            let midpoint = start + (end - start) / 2;
            let action = if request.mode == "translate" {
                "translation"
            } else {
                "cleanup"
            };
            emit_ai_progress(
                app,
                "processing",
                format!(
                    "Retrying {action} cues {}-{} as smaller groups",
                    start + 1,
                    end
                ),
                chunk_index as u64,
                Some(total_chunks as u64),
            );
            process_local_ai_range(
                app,
                cli,
                model_path,
                cues,
                document_samples,
                start,
                midpoint,
                results,
                request,
                chunk_index,
                total_chunks,
            )?;
            process_local_ai_range(
                app,
                cli,
                model_path,
                cues,
                document_samples,
                midpoint,
                end,
                results,
                request,
                chunk_index,
                total_chunks,
            )
        }
        Err(error) => Err(error),
    }
}

fn run_local_ai(app: &AppHandle, request: &AiRequest) -> Result<AiResult, String> {
    let model = ai_model_spec(&request.model_id)?;
    let root = ai_storage_root(app)?;
    let model_path = ai_model_path(&root, model.id)
        .ok_or_else(|| format!("Download {} before running local AI.", model.name))?;
    let cli = find_file(&root.join("runtime").join("current"), "llama-cli.exe")
        .ok_or("The local AI engine is missing. Download the selected model again to repair it.")?;
    let cues = parse_srt_cues(&request.content)?;
    let document_samples = document_sample_cues(&cues);
    let mut results: BTreeMap<usize, String> = BTreeMap::new();
    let total_chunks = cues.len().div_ceil(AI_TARGET_CHUNK_SIZE);
    for (chunk_index, start) in (0..cues.len()).step_by(AI_TARGET_CHUNK_SIZE).enumerate() {
        ensure_ai_not_cancelled(app)?;
        let end = (start + AI_TARGET_CHUNK_SIZE).min(cues.len());
        let action = if request.mode == "translate" {
            "Translating"
        } else {
            "Cleaning"
        };
        emit_ai_progress(
            app,
            "processing",
            format!(
                "{action} contextual group {} of {}",
                chunk_index + 1,
                total_chunks
            ),
            chunk_index as u64,
            Some(total_chunks as u64),
        );
        process_local_ai_range(
            app,
            &cli,
            &model_path,
            &cues,
            &document_samples,
            start,
            end,
            &mut results,
            request,
            chunk_index,
            total_chunks,
        )?;
    }
    let (output, cue_count) = render_ai_srt(&cues, &results)?;
    emit_ai_progress(
        app,
        "complete",
        "AI result is ready",
        total_chunks as u64,
        Some(total_chunks as u64),
    );
    Ok(AiResult {
        srt_text: output,
        model_name: model.name.to_string(),
        cue_count,
        source_cue_count: cues.len(),
        dropped_cue_count: cues.len() - cue_count,
    })
}

fn deepl_target_code(language: &str) -> Result<&'static str, String> {
    match language.trim() {
        "English" => Ok("EN"),
        "Turkish" => Ok("TR"),
        "German" => Ok("DE"),
        "French" => Ok("FR"),
        "Spanish" => Ok("ES"),
        "Italian" => Ok("IT"),
        "Portuguese" => Ok("PT-PT"),
        "Arabic" => Ok("AR"),
        "Japanese" => Ok("JA"),
        "Korean" => Ok("KO"),
        "Chinese" => Ok("ZH-HANS"),
        _ => Err("The selected language is not supported by the DeepL integration.".to_string()),
    }
}

fn deepl_source_code(language: Option<&str>) -> Result<Option<&'static str>, String> {
    match language.unwrap_or("Auto detect").trim() {
        "" | "Auto detect" => Ok(None),
        "English" => Ok(Some("EN")),
        "Turkish" => Ok(Some("TR")),
        "German" => Ok(Some("DE")),
        "French" => Ok(Some("FR")),
        "Spanish" => Ok(Some("ES")),
        "Italian" => Ok(Some("IT")),
        "Portuguese" => Ok(Some("PT")),
        "Arabic" => Ok(Some("AR")),
        "Japanese" => Ok(Some("JA")),
        "Korean" => Ok(Some("KO")),
        "Chinese" => Ok(Some("ZH")),
        _ => Err(
            "The selected source language is not supported by the DeepL integration.".to_string(),
        ),
    }
}

fn deepl_endpoint(
    api_key: &str,
    plan: &str,
    allow_billed: bool,
) -> Result<(&'static str, bool), String> {
    let key = api_key.trim();
    if key.is_empty() || key.len() > 512 || key.chars().any(char::is_whitespace) {
        return Err("Enter a valid DeepL API key.".to_string());
    }
    match plan {
        "free" => Ok(("https://api-free.deepl.com/v2/translate", true)),
        "pro" if allow_billed => Ok(("https://api.deepl.com/v2/translate", false)),
        "pro" => Err("DeepL API Pro requires explicit billed-account confirmation.".to_string()),
        _ => Err("Select DeepL API Free or DeepL API Pro.".to_string()),
    }
}

fn deepl_context(cues: &[SubtitleCue], start: usize, end: usize, guidance: Option<&str>) -> String {
    let window_start = start.saturating_sub(AI_CONTEXT_CUES);
    let window_end = (end + AI_CONTEXT_CUES).min(cues.len());
    let mut context = String::from("Film subtitle scene context. Preserve names, relationships, register, and recurring terminology.\n");
    if let Some(value) = guidance.map(str::trim).filter(|value| !value.is_empty()) {
        context.push_str("Project guidance: ");
        context.push_str(value);
        context.push('\n');
    }
    context.push_str("Nearby dialogue:\n");
    for (offset, cue) in cues[window_start..window_end].iter().enumerate() {
        context.push_str(&format!(
            "{}: {}\n",
            window_start + offset + 1,
            cue.text.join(" / ")
        ));
    }
    context
}

fn deepl_error(status: u16) -> String {
    match status {
        403 => "DeepL rejected the API key. Check the key and account endpoint.".to_string(),
        413 => "The subtitle group is too large for DeepL. No partial result was saved.".to_string(),
        429 => "DeepL rate limit reached. Try again later; no partial result was saved.".to_string(),
        456 => "DeepL character quota has been exhausted. No paid upgrade or charge was started by SubHooper.".to_string(),
        _ => format!("DeepL translation failed with HTTP status {status}. No partial result was saved."),
    }
}

fn run_deepl_translation(app: &AppHandle, request: &AiRequest) -> Result<AiResult, String> {
    if request.mode != "translate" {
        return Err("DeepL is available only for translation.".to_string());
    }
    let language = validate_translation_language(request.target_language.as_deref())?;
    let target_lang = deepl_target_code(language)?;
    let source_lang = deepl_source_code(request.source_language.as_deref())?;
    if request
        .source_language
        .as_deref()
        .is_some_and(|value| value.trim() == language)
    {
        return Err(
            "Source and target languages must be different for online translation.".to_string(),
        );
    }
    let api_key = request.deepl_api_key.as_deref().unwrap_or("").trim();
    let plan = request.deepl_plan.as_deref().unwrap_or("free");
    let (endpoint, free_endpoint) = deepl_endpoint(api_key, plan, request.allow_billed_deepl)?;
    let guidance = request.guidance.as_deref().unwrap_or("").trim();
    if guidance.chars().count() > 4_000 {
        return Err("Translation guidance must be 4,000 characters or fewer.".to_string());
    }
    let cues = parse_srt_cues(&request.content)?;
    let client = reqwest::blocking::Client::builder()
        .user_agent(AI_USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|error| format!("Could not prepare the DeepL connection: {error}"))?;
    let mut results: BTreeMap<usize, String> = BTreeMap::new();
    let total_chunks = cues.len().div_ceil(DEEPL_TARGET_CHUNK_SIZE);
    for (chunk_index, start) in (0..cues.len()).step_by(DEEPL_TARGET_CHUNK_SIZE).enumerate() {
        ensure_ai_not_cancelled(app)?;
        let end = (start + DEEPL_TARGET_CHUNK_SIZE).min(cues.len());
        emit_ai_progress(
            app,
            "processing",
            format!(
                "Translating online group {} of {}",
                chunk_index + 1,
                total_chunks
            ),
            chunk_index as u64,
            Some(total_chunks as u64),
        );
        let payload = DeepLTranslateRequest {
            text: cues[start..end]
                .iter()
                .map(|cue| cue.text.join("\n"))
                .collect(),
            target_lang: target_lang.to_string(),
            source_lang: source_lang.map(str::to_string),
            context: deepl_context(&cues, start, end, request.guidance.as_deref()),
            split_sentences: "nonewlines",
            preserve_formatting: true,
            model_type: "prefer_quality_optimized",
        };
        let encoded_size = serde_json::to_vec(&payload)
            .map_err(|error| format!("Could not prepare the DeepL request: {error}"))?
            .len();
        if encoded_size > 120 * 1024 {
            return Err(
                "A subtitle group exceeds the DeepL request limit. No partial result was saved."
                    .to_string(),
            );
        }
        let response = client
            .post(endpoint)
            .header("Authorization", format!("DeepL-Auth-Key {api_key}"))
            .json(&payload)
            .send()
            .map_err(|error| format!("Could not reach DeepL: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(deepl_error(status.as_u16()));
        }
        let response: DeepLTranslateResponse = response
            .json()
            .map_err(|error| format!("DeepL returned an unreadable response: {error}"))?;
        ensure_ai_not_cancelled(app)?;
        if response.translations.len() != end - start {
            return Err(
                "DeepL returned an incomplete subtitle group. No partial result was saved."
                    .to_string(),
            );
        }
        for (offset, item) in response.translations.into_iter().enumerate() {
            if item.text.trim().is_empty() {
                return Err(
                    "DeepL returned an empty subtitle cue. No partial result was saved."
                        .to_string(),
                );
            }
            results.insert(start + offset + 1, item.text.trim().to_string());
        }
    }
    let (output, cue_count) = render_ai_srt(&cues, &results)?;
    emit_ai_progress(
        app,
        "complete",
        "Online translation is ready",
        total_chunks as u64,
        Some(total_chunks as u64),
    );
    let model_name = if free_endpoint {
        "DeepL API Free"
    } else {
        "DeepL API Pro"
    };
    Ok(AiResult {
        srt_text: output,
        model_name: model_name.to_string(),
        cue_count,
        source_cue_count: cues.len(),
        dropped_cue_count: 0,
    })
}

fn run_ai_blocking(app: AppHandle, request: AiRequest) -> Result<AiResult, String> {
    begin_ai_task(&app)?;
    let result = (|| {
        validate_choice(&request.mode, &["clean", "translate"], "AI task")?;
        let provider = request.provider.as_deref().unwrap_or("local");
        validate_choice(provider, &["local", "deepl"], "AI provider")?;
        if provider == "deepl" {
            run_deepl_translation(&app, &request)
        } else {
            run_local_ai(&app, &request)
        }
    })();
    end_ai_task(&app);
    result
}

fn validate_custom_region(request: &PipelineRequest) -> Result<(), String> {
    let values = [
        request.region_top,
        request.region_bottom,
        request.region_left,
        request.region_right,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("The subtitle region bounds are invalid.".to_string());
    }
    if request.region_top - request.region_bottom < 0.08
        || request.region_right - request.region_left < 0.08
    {
        return Err("The subtitle region is too small or invalid.".to_string());
    }
    Ok(())
}

fn run_pipeline_blocking(
    app: AppHandle,
    request: PipelineRequest,
) -> Result<PipelineResult, String> {
    validate_choice(
        &request.region,
        &["Bottom", "LowerHalf", "Full", "Custom"],
        "region",
    )?;
    if request.region == "Custom" {
        validate_custom_region(&request)?;
    }
    validate_choice(
        &request.compute,
        &["Auto", "CUDA", "CPU"],
        "VSF compute mode",
    )?;
    validate_choice(
        &request.ocr_compute,
        &["Auto", "CUDA", "CPU"],
        "OCR compute mode",
    )?;
    let video = validate_video(&request.video_path)?;
    let project = resolve_project_root()?;
    let results_root = resolve_results_root(&project)?;
    let reports_root = resolve_reports_root(&results_root)?;
    let script = project.join("run-pipeline.ps1");
    let state = app.state::<PipelineState>();
    state.cancel_requested.store(false, Ordering::SeqCst);
    {
        let mut active = lock(&state.active_pid)?;
        if active.is_some() {
            return Err("Another subtitle extraction is already running.".to_string());
        }
        *active = Some(0);
    }

    let executable = if cfg!(target_os = "windows") {
        "powershell.exe"
    } else {
        "pwsh"
    };
    let mut command = Command::new(executable);
    command
        .current_dir(&project)
        .env("SUBHOOPER_HOME", resolve_home_root(&project))
        .env("SUBTITLE_HOME", resolve_home_root(&project))
        .env("SUBTITLE_PROJECT_ROOT", &project)
        .env("SUBTITLE_RESULTS_ROOT", &results_root)
        .env("SUBTITLE_REPORTS_ROOT", &reports_root)
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(&script)
        .arg(&video)
        .arg("-SubtitleRegion")
        .arg(&request.region)
        .arg("-RegionTop")
        .arg(request.region_top.to_string())
        .arg("-RegionBottom")
        .arg(request.region_bottom.to_string())
        .arg("-RegionLeft")
        .arg(request.region_left.to_string())
        .arg("-RegionRight")
        .arg(request.region_right.to_string())
        .arg("-Compute")
        .arg(&request.compute)
        .arg("-OCRCompute")
        .arg(&request.ocr_compute)
        .arg("-Client")
        .arg("GUI")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);

    let mut child = command.spawn().map_err(|error| {
        if let Ok(mut active) = state.active_pid.lock() {
            *active = None;
        }
        format!("Could not start the pipeline: {error}")
    })?;
    let pid = child.id();
    *lock(&state.active_pid)? = Some(pid);
    if state.cancel_requested.load(Ordering::SeqCst) {
        let _ = kill_process_tree(pid);
    }
    let captured = Arc::new(Mutex::new(Vec::new()));
    let stdout_thread = child
        .stdout
        .take()
        .map(|reader| spawn_reader(reader, app.clone(), captured.clone()));
    let stderr_thread = child
        .stderr
        .take()
        .map(|reader| spawn_reader(reader, app.clone(), captured.clone()));
    let status = child
        .wait()
        .map_err(|error| format!("Could not wait for the pipeline: {error}"));
    if let Some(handle) = stdout_thread {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_thread {
        let _ = handle.join();
    }
    clear_active_pid(&app, pid);

    if state.cancel_requested.load(Ordering::SeqCst) {
        return Err("Processing was cancelled.".to_string());
    }
    let lines = lock(&captured)?.clone();
    let status_code = status
        .as_ref()
        .ok()
        .and_then(|value| value.code())
        .unwrap_or(-1);
    let pipeline_log = persist_pipeline_log(&reports_root, &lines, status_code);
    let status = status?;
    if !status.success() {
        let _ = app.emit(
            "pipeline-stage",
            PipelineStage {
                key: "failed",
                label: "Processing failed",
                percent: None,
            },
        );
        let message = pipeline_failure_message(&lines, status.code().unwrap_or(-1));
        return Err(match pipeline_log {
            Ok(path) => format!("{message}\nLog: {}", path.to_string_lossy()),
            Err(error) => format!("{message}\nCould not save the log: {error}"),
        });
    }
    let summary = parse_summary(&lines);
    if summary.get("Pipeline").map(String::as_str) != Some("COMPLETE") {
        return Err("The pipeline completed without a valid result summary.".to_string());
    }
    let srt_path = PathBuf::from(
        summary
            .get("SRT")
            .ok_or("The pipeline did not report an SRT result.")?,
    );
    let canonical_srt = srt_path
        .canonicalize()
        .map_err(|error| format!("Could not open the SRT result: {error}"))?;
    if !canonical_srt.starts_with(&results_root) || !canonical_srt.is_file() {
        return Err("The SRT result is outside the configured results directory.".to_string());
    }
    let srt_text = fs::read_to_string(&canonical_srt)
        .map_err(|error| format!("Could not read the SRT result: {error}"))?;
    let result = PipelineResult {
        summary,
        srt_text,
        srt_path: canonical_srt.to_string_lossy().into_owned(),
        result_dir: canonical_srt
            .parent()
            .unwrap_or(&results_root)
            .to_string_lossy()
            .into_owned(),
    };
    *lock(&state.last_result)? = Some(result.clone());
    Ok(result)
}

#[tauri::command]
async fn start_pipeline(
    app: AppHandle,
    request: PipelineRequest,
) -> Result<PipelineResult, String> {
    tauri::async_runtime::spawn_blocking(move || run_pipeline_blocking(app, request))
        .await
        .map_err(|error| format!("The pipeline task stopped unexpectedly: {error}"))?
}

#[tauri::command]
fn cancel_pipeline(app: AppHandle) -> Result<(), String> {
    let state = app.state::<PipelineState>();
    let pid = lock(&state.active_pid)?.ok_or("No pipeline is currently running.")?;
    if pid == 0 {
        state.cancel_requested.store(true, Ordering::SeqCst);
        return Ok(());
    }
    state.cancel_requested.store(true, Ordering::SeqCst);
    kill_process_tree(pid)?;
    let _ = app.emit(
        "pipeline-stage",
        PipelineStage {
            key: "cancelled",
            label: "Processing cancelled",
            percent: None,
        },
    );
    Ok(())
}

#[tauri::command]
fn save_export(destination: String, content: String, format: String) -> Result<String, String> {
    let destination = PathBuf::from(destination);
    validate_choice(&format, &["srt", "ttml", "txt", "md"], "export format")?;
    let extension_matches = destination
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case(&format));
    if !extension_matches {
        return Err(format!(
            "The output filename must use the .{format} extension."
        ));
    }
    let output = export_content(&content, &format)?;
    fs::write(&destination, output.as_bytes())
        .map_err(|error| format!("Could not save the file: {error}"))?;
    destination
        .canonicalize()
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(|error| format!("Could not resolve the saved SRT path: {error}"))
}

#[tauri::command]
fn ai_catalog(app: AppHandle) -> Result<Vec<AiModelInfo>, String> {
    let root = ai_storage_root(&app)?;
    Ok(AI_MODELS
        .iter()
        .copied()
        .map(|model| ai_model_info(&root, model))
        .collect())
}

#[tauri::command]
fn read_subtitle_file(path: String) -> Result<String, String> {
    let path = PathBuf::from(path)
        .canonicalize()
        .map_err(|error| format!("Could not open the subtitle file: {error}"))?;
    if !path.is_file()
        || !path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("srt"))
    {
        return Err("Select an SRT subtitle file.".to_string());
    }
    let size = path
        .metadata()
        .map_err(|error| format!("Could not inspect the subtitle file: {error}"))?
        .len();
    if size == 0 || size > AI_MAX_SUBTITLE_BYTES as u64 {
        return Err("The SRT file is empty or exceeds the 20 MB size limit.".to_string());
    }
    let content = fs::read_to_string(&path)
        .map_err(|error| format!("Could not read the SRT file as UTF-8: {error}"))?;
    prepare_srt_for_save(&content)
}

#[tauri::command]
async fn download_ai_model(app: AppHandle, model_id: String) -> Result<AiModelInfo, String> {
    tauri::async_runtime::spawn_blocking(move || install_ai_model_blocking(app, model_id))
        .await
        .map_err(|error| format!("The model download task stopped unexpectedly: {error}"))?
}

#[tauri::command]
async fn run_ai_cleaning(app: AppHandle, request: AiRequest) -> Result<AiResult, String> {
    tauri::async_runtime::spawn_blocking(move || run_ai_blocking(app, request))
        .await
        .map_err(|error| format!("The AI task stopped unexpectedly: {error}"))?
}

#[tauri::command]
fn cancel_ai(app: AppHandle) -> Result<(), String> {
    let state = app.state::<AiState>();
    let pid = lock(&state.active_pid)?.ok_or("No AI task is currently running.")?;
    state.cancel_requested.store(true, Ordering::SeqCst);
    if pid > 0 {
        kill_process_tree(pid)?;
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(PipelineState::default())
        .manage(AiState::default())
        .invoke_handler(tauri::generate_handler![
            start_pipeline,
            cancel_pipeline,
            save_export,
            component_status,
            install_components,
            ai_catalog,
            read_subtitle_file,
            download_ai_model,
            run_ai_cleaning,
            cancel_ai,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.app_handle().state::<PipelineState>();
                state.cancel_requested.store(true, Ordering::SeqCst);
                if let Ok(active) = state.active_pid.lock() {
                    if let Some(pid) = *active {
                        if pid > 0 {
                            let _ = kill_process_tree(pid);
                        }
                    }
                };
                let ai_state = window.app_handle().state::<AiState>();
                ai_state.cancel_requested.store(true, Ordering::SeqCst);
                if let Ok(active) = ai_state.active_pid.lock() {
                    if let Some(pid) = *active {
                        if pid > 0 {
                            let _ = kill_process_tree(pid);
                        }
                    }
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("Could not start the SubHooper GUI");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_parser_keeps_windows_paths_with_equals() {
        let lines = vec![
            "noise=value".to_string(),
            "--- SUBHOOPER PIPELINE 0.3.7 RESULT START ---".to_string(),
            "Pipeline=COMPLETE".to_string(),
            "SRT=C:\\Video=One\\result.srt".to_string(),
            "--- SUBHOOPER PIPELINE 0.3.7 RESULT END ---".to_string(),
        ];
        let result = parse_summary(&lines);
        assert_eq!(result.get("Pipeline").map(String::as_str), Some("COMPLETE"));
        assert_eq!(
            result.get("SRT").map(String::as_str),
            Some("C:\\Video=One\\result.srt")
        );
        assert!(!result.contains_key("noise"));
    }

    #[test]
    fn stage_parser_maps_real_pipeline_markers() {
        assert_eq!(
            stage_for_line("1/2 VideoSubFinder: test").unwrap().key,
            "vsf"
        );
        assert_eq!(
            stage_for_line("2/2 RapidVideOCR: test").unwrap().percent,
            None
        );
        assert!(stage_for_line("ordinary output").is_none());
    }

    #[test]
    fn edited_srt_is_lf_normalized_and_terminated() {
        let value =
            prepare_srt_for_save("1\r\n00:00:01,000 --> 00:00:02,000\r\nText").expect("valid SRT");
        assert_eq!(value, "1\n00:00:01,000 --> 00:00:02,000\nText\n");
    }

    #[test]
    fn pipeline_failure_exposes_actionable_last_error() {
        let lines = vec![
            "ordinary output".to_string(),
            "ERROR: A compatible OCR runtime was not found.".to_string(),
        ];
        assert_eq!(
            pipeline_failure_message(&lines, 1),
            "A compatible OCR runtime was not found. (pipeline code: 1)"
        );
    }

    #[test]
    fn exports_ttml_txt_and_markdown_from_edited_srt() {
        let srt = "1\n00:00:01,000 --> 00:00:02,500\nA & B\nSecond line\n";
        let ttml = export_content(srt, "ttml").expect("TTML export");
        assert!(ttml.contains("begin=\"00:00:01.000\""));
        assert!(ttml.contains("A &amp; B<br/>Second line"));
        assert_eq!(export_content(srt, "txt").unwrap(), "A & B Second line\n");
        assert!(export_content(srt, "md")
            .unwrap()
            .contains("## 00:00:01,000 → 00:00:02,500"));
    }

    #[test]
    fn ai_output_parser_uses_structured_cues_after_thinking_text() {
        let parsed = clean_model_output(
            "<think>private reasoning</think>\n[{\"id\":1,\"text\":\"Clean text\"}]",
        )
        .expect("valid model output");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, 1);
        assert_eq!(parsed[0].text, "Clean text");
    }

    #[test]
    fn ai_prompt_file_is_removed_after_use() {
        let path;
        {
            let prompt = TemporaryPrompt::create("local prompt").expect("temporary prompt");
            path = prompt.path().to_path_buf();
            assert_eq!(fs::read_to_string(&path).unwrap(), "local prompt");
        }
        assert!(!path.exists());
    }

    #[test]
    fn ai_output_file_is_removed_after_use() {
        let path;
        {
            let output = TemporaryOutput::create().expect("temporary output");
            path = output.path().to_path_buf();
            assert!(path.is_file());
        }
        assert!(!path.exists());
    }

    #[test]
    fn ai_output_parser_repairs_only_a_missing_array_terminator() {
        let parsed =
            clean_model_output("[{\"id\":1,\"text\":\"First\"},{\"id\":2,\"text\":\"Second\"}")
                .expect("safely repairable model output");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].id, 2);
        assert!(clean_model_output("[{\"id\":1,\"text\":\"partial\"").is_err());
    }

    #[test]
    fn ai_output_parser_repairs_a_smart_quote_used_as_json_terminator() {
        let transcript =
            "User:\r\n[prompt]\r\n\r\nAssistant:\r\n[{\"id\":66,\"text\":\"槽！”}]\r\n";
        let parsed = clean_model_output(assistant_output(transcript))
            .expect("smart quote terminator should be repaired");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, 66);
        assert_eq!(parsed[0].text, "槽！”");

        let already_valid = clean_model_output(r#"[{"id":66,"text":"槽！”"}]"#)
            .expect("valid smart quote text should remain valid");
        assert_eq!(already_valid[0].text, "槽！”");
    }

    #[test]
    fn local_ai_subdivision_is_limited_to_model_output_failures() {
        assert!(should_subdivide_ai_range(
            "The model returned invalid subtitle data. Diagnostic log: test",
            12,
        ));
        assert!(!should_subdivide_ai_range(
            "The local AI engine failed to start.",
            12,
        ));
        assert!(!should_subdivide_ai_range(
            "The model returned invalid subtitle data.",
            1,
        ));
    }

    #[test]
    fn ai_output_parser_uses_only_the_assistant_part_of_cli_transcripts() {
        let transcript = "User:\n[{\"id\":1,\"text\":\"Original\"}]\n\nAssistant:\n[{\"id\":1,\"text\":\"Cleaned\"}]";
        let parsed = clean_model_output(assistant_output(transcript)).expect("assistant response");
        assert_eq!(parsed[0].text, "Cleaned");
        assert!(clean_model_output(assistant_output(
            "User:\n[{\"id\":1,\"text\":\"Original\"}]\n\nAssistant:\n"
        ))
        .is_err());
    }

    #[test]
    fn local_ai_discards_returned_context_cues_when_all_targets_are_valid() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["Reference".into()],
        };
        let targets = (13..=24).map(|id| (id, &cue)).collect::<Vec<_>>();
        let context_after = (25..=28).map(|id| (id, &cue)).collect::<Vec<_>>();
        let document_samples = vec![(40, &cue)];
        let mut response = (13..=28)
            .map(|id| AiCueOutput {
                id,
                text: format!("Cue {id}"),
            })
            .collect::<Vec<_>>();
        response.push(AiCueOutput {
            id: 40,
            text: "Document sample".into(),
        });
        let selected = select_ai_target_response(
            &response,
            &targets,
            &[],
            &context_after,
            &document_samples,
            false,
        )
        .expect("context leakage should be safely discarded");
        assert_eq!(selected.len(), 12);
        assert_eq!(selected.first().unwrap().id, 13);
        assert_eq!(selected.last().unwrap().id, 24);
    }

    #[test]
    fn local_ai_still_rejects_missing_duplicate_reordered_or_unknown_targets() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["Reference".into()],
        };
        let targets = vec![(13, &cue), (14, &cue)];
        let context_after = vec![(15, &cue)];
        let missing = vec![AiCueOutput {
            id: 13,
            text: "One".into(),
        }];
        let duplicate = vec![
            AiCueOutput {
                id: 13,
                text: "One".into(),
            },
            AiCueOutput {
                id: 13,
                text: "Again".into(),
            },
            AiCueOutput {
                id: 14,
                text: "Two".into(),
            },
        ];
        let reordered = vec![
            AiCueOutput {
                id: 14,
                text: "Two".into(),
            },
            AiCueOutput {
                id: 13,
                text: "One".into(),
            },
        ];
        let unknown = vec![
            AiCueOutput {
                id: 13,
                text: "One".into(),
            },
            AiCueOutput {
                id: 14,
                text: "Two".into(),
            },
            AiCueOutput {
                id: 99,
                text: "Unknown".into(),
            },
        ];
        let empty = vec![
            AiCueOutput {
                id: 13,
                text: "One".into(),
            },
            AiCueOutput {
                id: 14,
                text: "".into(),
            },
        ];
        assert!(
            select_ai_target_response(&missing, &targets, &[], &context_after, &[], false).is_err()
        );
        assert!(
            select_ai_target_response(&duplicate, &targets, &[], &context_after, &[], false)
                .is_err()
        );
        assert!(
            select_ai_target_response(&reordered, &targets, &[], &context_after, &[], false)
                .is_err()
        );
        assert!(
            select_ai_target_response(&unknown, &targets, &[], &context_after, &[], false).is_err()
        );
        assert!(
            select_ai_target_response(&empty, &targets, &[], &context_after, &[], false).is_err()
        );
        let cleaned = select_ai_target_response(&empty, &targets, &[], &context_after, &[], true)
            .expect("cleanup may explicitly drop noise");
        assert!(cleaned[1].text.is_empty());
    }

    #[test]
    fn ai_engine_sampler_failures_are_not_reported_as_model_json_errors() {
        let error = ai_engine_generation_error(
            "Error: Failed to initialize samplers: Unexpected empty grammar stack",
            "",
        )
        .expect("sampler error");
        assert!(error.contains("could not initialize generation"));
    }

    #[test]
    fn translation_prompt_requires_a_target_language() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["Hello".into()],
        };
        assert!(ai_prompt(
            &[(1, &cue)],
            &[],
            &[],
            &[],
            &[],
            "translate",
            None,
            None,
            None,
            None,
            None,
        )
        .is_err());
        let prompt = ai_prompt(
            &[(1, &cue)],
            &[(8, &cue)],
            &[(10, &cue)],
            &[(1, &cue)],
            &[(8, "Previous translation".into())],
            "translate",
            Some("English"),
            Some("Turkish"),
            None,
            None,
            Some("Keep the character name Ada."),
        )
        .unwrap();
        assert!(prompt.contains("Translate only target_cues into Turkish"));
        assert!(prompt.contains("declared source language is English"));
        assert!(prompt.contains("context_before"));
        assert!(prompt.contains("Previous translation"));
        assert!(prompt.contains("Keep the character name Ada."));
    }

    #[test]
    fn cleanup_prompt_normalizes_language_and_marks_only_noise_for_removal() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["Meaningful dialogue".into()],
        };
        let prompt = ai_prompt(
            &[(1, &cue)],
            &[],
            &[],
            &[(1, &cue)],
            &[],
            "clean",
            None,
            None,
            Some("Auto detect"),
            Some("balanced"),
            None,
        )
        .unwrap();
        assert!(prompt.contains("Infer the dominant subtitle language"));
        assert!(prompt.contains("translate coherent foreign dialogue"));
        assert!(prompt.contains("high-confidence non-dialogue OCR noise"));
        assert!(prompt.contains("empty text string"));
        assert!(prompt.contains("document_samples"));
    }

    #[test]
    fn cleanup_strengths_are_explicit_and_strong_is_aggressive() {
        assert_eq!(validate_cleanup_strength(None).unwrap(), "balanced");
        assert_eq!(validate_cleanup_strength(Some("light")).unwrap(), "light");
        assert_eq!(validate_cleanup_strength(Some("strong")).unwrap(), "strong");
        assert!(validate_cleanup_strength(Some("maximum")).is_err());
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["7".into()],
        };
        let prompt = ai_prompt(
            &[(1, &cue)],
            &[],
            &[],
            &[(1, &cue)],
            &[],
            "clean",
            None,
            None,
            Some("English"),
            Some("strong"),
            None,
        )
        .unwrap();
        assert!(prompt.contains("Use strong cleanup"));
        assert!(prompt.contains("short numeric or mixed-character debris"));
        assert!(prompt.contains("Do not keep meaningless text merely because it is uncertain"));
    }

    #[test]
    fn strong_cleanup_prefilter_removes_fragment_cluster_but_keeps_short_words() {
        let cue = |text: &[&str]| SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: text.iter().map(|line| (*line).to_string()).collect(),
        };
        let cues = vec![
            cue(&["."]),
            cue(&["7"]),
            cue(&["1", "0", "R."]),
            cue(&["I"]),
            cue(&["Understand."]),
        ];
        assert!(strong_cleanup_prefilter(&cues, 0));
        assert!(strong_cleanup_prefilter(&cues, 1));
        assert!(strong_cleanup_prefilter(&cues, 2));
        assert!(!strong_cleanup_prefilter(&cues, 3));
        assert!(!strong_cleanup_prefilter(&cues, 4));
    }

    #[test]
    fn cleanup_document_samples_cover_the_full_subtitle() {
        let cues = (1..=40)
            .map(|index| SubtitleCue {
                start: "00:00:01,000".into(),
                end: "00:00:02,000".into(),
                text: vec![format!("English dialogue {index}")],
            })
            .collect::<Vec<_>>();
        let samples = document_sample_cues(&cues);
        assert_eq!(samples.len(), AI_DOCUMENT_SAMPLE_CUES);
        assert_eq!(samples.first().unwrap().0, 1);
        assert_eq!(samples.last().unwrap().0, 40);
    }

    #[test]
    fn cleaned_srt_drops_empty_decisions_and_renumbers_kept_cues() {
        let cues = vec![
            SubtitleCue {
                start: "00:00:01,000".into(),
                end: "00:00:02,000".into(),
                text: vec!["Keep".into()],
            },
            SubtitleCue {
                start: "00:00:03,000".into(),
                end: "00:00:04,000".into(),
                text: vec!["Noise".into()],
            },
            SubtitleCue {
                start: "00:00:05,000".into(),
                end: "00:00:06,000".into(),
                text: vec!["Keep too".into()],
            },
        ];
        let results = BTreeMap::from([
            (1, "First".to_string()),
            (2, "".to_string()),
            (3, "Second".to_string()),
        ]);
        let (srt, count) = render_ai_srt(&cues, &results).unwrap();
        assert_eq!(count, 2);
        assert!(srt.contains("1\n00:00:01,000 --> 00:00:02,000\nFirst"));
        assert!(srt.contains("2\n00:00:05,000 --> 00:00:06,000\nSecond"));
        assert!(!srt.contains("00:00:03,000"));
    }

    #[test]
    fn local_catalog_excludes_tiny_model_and_recommends_eight_billion_parameters() {
        assert_eq!(AI_MODELS.len(), 2);
        assert!(AI_MODELS.iter().all(|model| model.id != "qwen3-1.7b-q4km"));
        assert_eq!(AI_MODELS[1].id, "qwen3-8b-q4km");
    }

    #[test]
    fn deepl_plan_selection_locks_free_and_pro_endpoints() {
        let (endpoint, free) = deepl_endpoint("personal-free-key", "free", false).unwrap();
        assert_eq!(endpoint, "https://api-free.deepl.com/v2/translate");
        assert!(free);
        assert!(deepl_endpoint("personal-pro-key", "pro", false).is_err());
        let (endpoint, free) = deepl_endpoint("personal-pro-key", "pro", true).unwrap();
        assert_eq!(endpoint, "https://api.deepl.com/v2/translate");
        assert!(!free);
        assert!(deepl_endpoint("personal-key", "unknown", true).is_err());
    }

    #[test]
    fn deepl_language_mapping_is_explicit() {
        assert_eq!(deepl_target_code("Turkish").unwrap(), "TR");
        assert_eq!(deepl_target_code("Chinese").unwrap(), "ZH-HANS");
        assert_eq!(deepl_source_code(Some("English")).unwrap(), Some("EN"));
        assert_eq!(deepl_source_code(Some("Auto detect")).unwrap(), None);
        assert!(deepl_target_code("Unknown").is_err());
    }

    #[test]
    fn model_download_url_has_one_separator_after_main() {
        let normal = model_download_url("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf")
            .expect("valid model URL");
        let leading_slash = model_download_url("Qwen/Qwen3-4B-GGUF", "/Qwen3-4B-Q4_K_M.gguf")
            .expect("valid model URL with leading slash");
        for url in [normal, leading_slash] {
            assert!(url.contains("/resolve/main/Qwen3-4B-Q4_K_M.gguf"));
            assert!(!url.contains("/resolve/main//"));
        }
    }

    #[test]
    fn custom_region_rejects_too_small_boxes() {
        let request = PipelineRequest {
            video_path: "video.mp4".into(),
            region: "Custom".into(),
            region_top: 0.50,
            region_bottom: 0.45,
            region_left: 0.10,
            region_right: 0.90,
            compute: "Auto".into(),
            ocr_compute: "Auto".into(),
        };
        assert!(validate_custom_region(&request).is_err());
    }

    #[test]
    fn application_home_uses_local_app_data_when_available() {
        let project = Path::new("C:/Tools/SubHooper/app");
        let expected = std::env::var_os("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .map(|path| path.join("SubHooper"))
            .unwrap_or_else(|| project.to_path_buf());
        assert_eq!(resolve_home_root(project), expected);
    }
}
