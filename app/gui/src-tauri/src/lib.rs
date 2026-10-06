use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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
use std::os::windows::io::AsRawHandle;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(target_os = "windows")]
#[link(name = "bcrypt")]
extern "system" {
    fn BCryptGenRandom(algorithm: *mut std::ffi::c_void, buffer: *mut u8, size: u32, flags: u32) -> i32;
}

#[cfg(target_os = "windows")]
#[link(name = "iphlpapi")]
extern "system" {
    fn GetExtendedTcpTable(table: *mut std::ffi::c_void, size: *mut u32, order: i32,
        address_family: u32, table_class: u32, reserved: u32) -> u32;
}

#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Clone, Copy)]
struct TcpRowOwnerPid {
    state: u32,
    local_address: u32,
    local_port: u32,
    remote_address: u32,
    remote_port: u32,
    owning_pid: u32,
}

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "ts", "m2ts", "wmv", "m4v",
];
const AI_MAX_SUBTITLE_BYTES: usize = 20 * 1024 * 1024;
const AI_USER_AGENT: &str = "SubHooper/0.4.3";
const AI_TARGET_CHUNK_SIZE: usize = 12;
const AI_TRANSLATION_CHUNK_SIZE: usize = 8;
const AI_CONTEXT_CUES: usize = 4;
const AI_DOCUMENT_SAMPLE_CUES: usize = 24;

#[derive(Clone, Copy)]
struct AiModelSpec {
    id: &'static str,
    name: &'static str,
    repository: &'static str,
    revision: &'static str,
    filename: &'static str,
    sha256: &'static str,
    size_label: &'static str,
    memory_label: &'static str,
    recommendation: &'static str,
    minimum_gpu_mib: u64,
    minimum_free_gpu_mib: u64,
}

const AI_MODELS: &[AiModelSpec] = &[
    AiModelSpec {
        id: "qwen3-4b-q4km", name: "Qwen3 4B Q4_K_M",
        repository: "Qwen/Qwen3-4B-GGUF", revision: "a9a60d009fa7ff9606305047c2bf77ac25dbec49",
        filename: "Qwen3-4B-Q4_K_M.gguf",
        sha256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        size_label: "2.5 GB", memory_label: "2.5 GB file; runtime memory is higher",
        recommendation: "Smallest quality-oriented option for limited memory; review the output.",
        minimum_gpu_mib: 4096,
        minimum_free_gpu_mib: 3072,
    },
    AiModelSpec {
        id: "qwen3-8b-q4km", name: "Qwen3 8B Q4_K_M",
        repository: "Qwen/Qwen3-8B-GGUF", revision: "6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
        filename: "Qwen3-8B-Q4_K_M.gguf",
        sha256: "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        size_label: "5.03 GB", memory_label: "5.03 GB file; allow extra VRAM for context/runtime",
        recommendation: "Recommended medium model for OCR cleanup and translation.",
        minimum_gpu_mib: 8192,
        minimum_free_gpu_mib: 6144,
    },
    AiModelSpec {
        id: "qwen3-14b-q4km", name: "Qwen3 14B Q4_K_M",
        repository: "Qwen/Qwen3-14B-GGUF", revision: "c75e7b2d0234068f674a1bacf548ea32e27ccd29",
        filename: "Qwen3-14B-Q4_K_M.gguf",
        sha256: "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        size_label: "9 GB", memory_label: "9 GB file; allow extra VRAM for context/runtime",
        recommendation: "High-capacity option when memory permits; review the output.",
        minimum_gpu_mib: 12288,
        minimum_free_gpu_mib: 10240,
    },
];

const LLAMA_RELEASE_TAG: &str = "b10941";
const CUDA_MIN_TAG: &str = "12.4";
const CUDA_MAX_TAG: &str = "13.3";
const LLAMA_RUNTIME_MANIFEST: &str = ".subhooper-llama-runtime.json";

#[cfg(target_os = "windows")]
#[repr(C)]
struct ByHandleFileInformation {
    file_attributes: u32,
    creation_time_low: u32,
    creation_time_high: u32,
    last_access_time_low: u32,
    last_access_time_high: u32,
    last_write_time_low: u32,
    last_write_time_high: u32,
    volume_serial_number: u32,
    file_size_high: u32,
    file_size_low: u32,
    number_of_links: u32,
    file_index_high: u32,
    file_index_low: u32,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn GetFileInformationByHandle(handle: *mut std::ffi::c_void, information: *mut ByHandleFileInformation) -> i32;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ModelFileFingerprint {
    size: u64,
    modified_seconds: u64,
    modified_nanos: u32,
    #[serde(default)]
    windows_file_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelVerificationCache {
    version: u32,
    sha256: String,
    fingerprint: ModelFileFingerprint,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RuntimeIntegrityManifest {
    release_tag: String,
    backend: String,
    cuda_tag: String,
    archive_sha256: BTreeMap<String, String>,
    files: BTreeMap<String, String>,
}

#[derive(Default)]
struct PipelineState {
    active_pid: Mutex<Option<u32>>,
    active_session: Mutex<Option<PipelineSession>>,
    cancel_requested: AtomicBool,
    closing: AtomicBool,
    last_result: Mutex<Option<PipelineResult>>,
}

struct PipelineSession {
    token: String,
    project: PathBuf,
}

#[derive(Default)]
struct AiState {
    active_pid: Mutex<Option<u32>>,
    cancel_requested: AtomicBool,
    closing: AtomicBool,
    warm: Mutex<Option<WarmAiServer>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiModelInfo {
    id: String,
    name: String,
    size_label: String,
    memory_label: String,
    recommendation: String,
    minimum_gpu_mib: u64,
    installed: bool,
    model_cached: bool,
    gpu_eligible: bool,
    gpu_reason: String,
    gpu_total_mib: Option<u64>,
    gpu_free_mib: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiProgress {
    phase: String,
    label: String,
    received: u64,
    total: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OcrHardwareRecommendation {
    recommended_mode: String,
    reason: String,
}

fn recommended_ocr_mode(nvidia_detected: bool, directml_candidate: bool) -> &'static str {
    if nvidia_detected || directml_candidate { "mixed" } else { "cpu" }
}

fn directml_ocr_device_detected() -> bool {
    let mut command = Command::new("powershell.exe");
    command.args(["-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -like 'PCI*' -and $_.Status -eq 'OK' } | Select-Object -ExpandProperty Name"])
        .stdout(Stdio::piped()).stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    command.output().is_ok_and(|output| {
        output.status.success() && String::from_utf8_lossy(&output.stdout).lines().any(|name| {
            let name = name.to_ascii_lowercase();
            name.contains("amd") || name.contains("radeon") || name.contains("intel")
        })
    })
}

fn nvidia_ocr_device_detected() -> bool {
    let mut command = Command::new("nvidia-smi");
    command.args(["--query-gpu=name", "--format=csv,noheader"])
        .stdout(Stdio::piped()).stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    command.output().is_ok_and(|output| {
        output.status.success() && !String::from_utf8_lossy(&output.stdout).trim().is_empty()
    })
}

#[tauri::command]
fn ocr_hardware_recommendation() -> OcrHardwareRecommendation {
    let nvidia = nvidia_ocr_device_detected();
    let directml = !nvidia && directml_ocr_device_detected();
    if !nvidia && !directml {
        return OcrHardwareRecommendation {
            recommended_mode: recommended_ocr_mode(false, false).into(),
            reason: "A supported GPU candidate was not detected; CPU is recommended.".into(),
        };
    }
    OcrHardwareRecommendation {
        recommended_mode: recommended_ocr_mode(nvidia, directml).into(),
        reason: if nvidia {
            "NVIDIA GPU detected. GPU + CPU uses CUDA; runtime availability is checked at startup."
        } else {
            "AMD/Intel graphics detected. GPU + CPU tries DirectML; model execution is checked at startup. Speed depends on the GPU and driver."
        }.into(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiRequest {
    content: String,
    model_id: String,
    #[serde(default)]
    compute_backend: Option<AiComputeBackend>,
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
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum AiComputeBackend {
    Cuda,
    Cpu,
}

impl AiComputeBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
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
    ocr_runtime: bool,
    gpu_runtime: bool,
    gpu_error: Option<String>,
    ready: bool,
    install_root: String,
}

#[derive(Clone, Debug)]
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

fn probe_ocr_runtime(project: &Path, root: &Path, runtime: &str) -> Result<(), String> {
    let runtime_python = root.join("runtime").join(runtime).join("Scripts").join("python.exe");
    if !runtime_python.is_file() {
        return Err(format!("OCR runtime is not installed: {runtime}."));
    }
    let mut probe = Command::new(runtime_python);
    probe
        .arg(project.join("engine").join("probe.py"))
        .arg(root.join("components").join("native-ocr-models"));
    #[cfg(target_os = "windows")]
    probe.creation_flags(CREATE_NO_WINDOW);
    let output = probe.output().map_err(|error| format!("Could not validate {runtime}: {error}"))?;
    if output.status.success() { return Ok(()); }
    let detail = String::from_utf8_lossy(&output.stderr);
    let detail = detail.lines().rev().find(|line| !line.trim().is_empty())
        .unwrap_or("OCR validation failed.");
    Err(detail.to_string())
}

fn component_status_value(project: &Path) -> ComponentStatus {
    let root = resolve_home_root(project);
    let ocr_runtime = probe_ocr_runtime(project, &root, "native-ocr-cpu-py313-v040").is_ok();
    let mut driver = Command::new("nvidia-smi.exe");
    driver.args(["--query-gpu=name", "--format=csv,noheader"]);
    #[cfg(target_os = "windows")]
    driver.creation_flags(CREATE_NO_WINDOW);
    let cuda_device = driver.output().is_ok_and(|output|
        output.status.success() && !String::from_utf8_lossy(&output.stdout).trim().is_empty());
    let cuda = if cuda_device {
        probe_ocr_runtime(project, &root, "native-ocr-cuda-py313-v040")
    } else {
        Err("NVIDIA GPU detection failed; check the NVIDIA driver.".to_string())
    };
    let gpu = cuda.or_else(|cuda_error|
        probe_ocr_runtime(project, &root, "native-ocr-dml-py313-v043")
            .map_err(|dml_error| format!("CUDA: {cuda_error} DirectML: {dml_error}")));
    ComponentStatus {
        ocr_runtime,
        gpu_runtime: gpu.is_ok(),
        gpu_error: gpu.err(),
        ready: ocr_runtime,
        install_root: root.display().to_string(),
    }
}

#[tauri::command]
fn component_status() -> Result<ComponentStatus, String> {
    startup_trace("component-status-enter");
    let project = resolve_project_root()?;
    let status = component_status_value(&project);
    startup_trace("component-status-exit");
    Ok(status)
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
    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not start component setup: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Could not capture component setup progress.")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Could not capture component setup errors.")?;
    let stderr_reader = thread::spawn(move || {
        let mut output = String::new();
        let _ = BufReader::new(stderr).read_to_string(&mut output);
        output
    });
    let mut stdout_text = String::new();
    let mut stdout_reader = BufReader::new(stdout);
    let mut raw_line = Vec::new();
    loop {
        raw_line.clear();
        let read = stdout_reader
            .read_until(b'\n', &mut raw_line)
            .map_err(|error| format!("Could not read component setup output: {error}"))?;
        if read == 0 {
            break;
        }
        let line = String::from_utf8_lossy(&raw_line).trim().to_string();
        if !line.trim().is_empty() {
            let _ = app.emit("component-progress", line.clone());
            stdout_text.push_str(&line);
            stdout_text.push('\n');
        }
    }
    let status = child
        .wait()
        .map_err(|error| format!("Could not wait for component setup: {error}"))?;
    let stderr_text = stderr_reader.join().unwrap_or_default();
    let stderr = stderr_text.trim().to_string();
    if !status.success() {
        let detail = if stderr.is_empty() { stdout_text.trim() } else { stderr.as_str() };
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
    if let Some(value) = line.trim().strip_prefix("OCRProgress=") {
        let percent = value.parse::<u8>().ok()?;
        if percent > 99 {
            return None;
        }
        Some(PipelineStage {
            key: "ocr",
            label: "Scanning and recognizing subtitles",
            percent: Some(percent),
        })
    } else if line.contains("1/2 Native engine") {
        Some(PipelineStage {
            key: "ocr",
            label: "Scanning subtitles with the native OCR engine",
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

fn cleanup_pipeline_workspace(app: &AppHandle, expected_token: Option<&str>) -> Result<(), String> {
    // The caller must first confirm termination. Serialize cleanup and retain
    // ownership on failure so another termination handler can retry safely.
    let state = app.state::<PipelineState>();
    let mut active_session = lock(&state.active_session)?;
    let Some(session) = active_session.as_ref() else { return Ok(()); };
    if expected_token.is_some_and(|token| token != session.token.as_str()) { return Ok(()); }
    let python = resolve_home_root(&session.project)
        .join("runtime").join("native-ocr-cpu-py313-v040")
        .join("Scripts").join("python.exe");
    let mut command = Command::new(python);
    command.arg(session.project.join("engine").join("runtime.py"))
        .args(["--cleanup-session", session.token.as_str()])
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    let status = command.status()
        .map_err(|error| format!("Could not clean the terminated pipeline workspace: {error}"))?;
    if status.success() { *active_session = None; Ok(()) }
    else { Err("Temporary workspace cleanup failed after pipeline termination.".into()) }
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
        prepare_srt_for_save(content)?;
        return Ok(content.to_string());
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
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut seen = 0usize;
    while let Some((directory, depth)) = pending.pop() {
        for entry in fs::read_dir(directory).ok()? {
            let entry = entry.ok()?;
            seen += 1;
            if seen > 8192 {
                return None;
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).ok()?;
            if runtime_reparse_point(&metadata) {
                continue;
            }
            if metadata.is_file()
                && path.file_name().and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case(filename))
            {
                return Some(path);
            }
            if metadata.is_dir() && depth < 12 {
                pending.push((path, depth + 1));
            }
        }
    }
    None
}

fn runtime_reparse_point(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::MetadataExt;
        return metadata.file_attributes() & 0x400 != 0;
    }
    #[cfg(not(target_os = "windows"))]
    false
}

fn ai_model_path(root: &Path, model_id: &str) -> Option<PathBuf> {
    let model = ai_model_spec(model_id).ok()?;
    let path = root.join("models").join(model_id).join(model.filename);
    fs::symlink_metadata(&path).ok().filter(|metadata| metadata.is_file() && !runtime_reparse_point(metadata)).map(|_| path)
}

fn model_digest_sidecar(path: &Path) -> PathBuf {
    let filename = path
        .file_name()
        .map(|value| value.to_string_lossy())
        .unwrap_or_default();
    path.with_file_name(format!("{filename}.sha256"))
}

fn model_verification_cache_path(path: &Path) -> PathBuf {
    let filename = path.file_name().map(|value| value.to_string_lossy()).unwrap_or_default();
    path.with_file_name(format!("{filename}.verified.json"))
}

fn model_file_fingerprint(path: &Path) -> Result<ModelFileFingerprint, String> {
    let path_metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("Could not inspect the local model: {error}"))?;
    if !path_metadata.is_file() || runtime_reparse_point(&path_metadata) {
        return Err("The local model must be a regular file, not a link or reparse point.".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options.open(path).map_err(|error| format!("Could not open the local model: {error}"))?;
    let metadata = file.metadata().map_err(|error| format!("Could not inspect the local model: {error}"))?;
    if !metadata.is_file() || runtime_reparse_point(&metadata) {
        return Err("The local model must be a regular file, not a link or reparse point.".into());
    }
    let modified = metadata.modified().map_err(|error| format!("Could not read the local model modification time: {error}"))?
        .duration_since(UNIX_EPOCH).map_err(|_| "The local model modification time is invalid.")?;
    #[cfg(target_os = "windows")]
    let windows_file_id = {
        let mut information = ByHandleFileInformation {
            file_attributes: 0, creation_time_low: 0, creation_time_high: 0,
            last_access_time_low: 0, last_access_time_high: 0,
            last_write_time_low: 0, last_write_time_high: 0,
            volume_serial_number: 0, file_size_high: 0, file_size_low: 0,
            number_of_links: 0, file_index_high: 0, file_index_low: 0,
        };
        let ok = unsafe { GetFileInformationByHandle(file.as_raw_handle() as *mut std::ffi::c_void, &mut information) };
        if ok == 0 {
            return Err(format!("Could not read the local model file identity: {}", std::io::Error::last_os_error()));
        }
        Some(format!("{:08x}:{:08x}{:08x}", information.volume_serial_number, information.file_index_high, information.file_index_low))
    };
    #[cfg(not(target_os = "windows"))]
    let windows_file_id = None;
    Ok(ModelFileFingerprint {
        size: metadata.len(),
        modified_seconds: modified.as_secs(),
        modified_nanos: modified.subsec_nanos(),
        windows_file_id,
    })
}

fn model_sidecar_matches(path: &Path, expected_sha256: &str) -> bool {
    fs::read_to_string(model_digest_sidecar(path))
        .is_ok_and(|digest| digest.trim().eq_ignore_ascii_case(expected_sha256))
}

fn write_model_verification_cache_with_fingerprint(path: &Path, expected_sha256: &str, fingerprint: ModelFileFingerprint) -> Result<(), String> {
    let cache = ModelVerificationCache {
        version: 1,
        sha256: expected_sha256.to_ascii_lowercase(),
        fingerprint,
    };
    let cache_path = model_verification_cache_path(path);
    let temporary = cache_path.with_extension(format!("verified.{}.tmp", std::process::id()));
    let encoded = serde_json::to_vec(&cache).map_err(|error| format!("Could not encode the model verification record: {error}"))?;
    fs::write(&temporary, encoded).map_err(|error| format!("Could not save the model verification record: {error}"))?;
    if cache_path.exists() { let _ = fs::remove_file(&cache_path); }
    fs::rename(&temporary, &cache_path).map_err(|error| format!("Could not install the model verification record: {error}"))
}

fn write_model_verification_cache(path: &Path, expected_sha256: &str) -> Result<(), String> {
    write_model_verification_cache_with_fingerprint(path, expected_sha256, model_file_fingerprint(path)?)
}

fn invalidate_model_verification(path: &Path) {
    let _ = fs::remove_file(model_digest_sidecar(path));
    let _ = fs::remove_file(model_verification_cache_path(path));
}

fn model_verification_cache_matches(path: &Path, expected_sha256: &str) -> bool {
    if !model_sidecar_matches(path, expected_sha256) { return false; }
    let Ok(fingerprint) = model_file_fingerprint(path) else { return false; };
    fs::read(model_verification_cache_path(path)).ok()
        .and_then(|bytes| serde_json::from_slice::<ModelVerificationCache>(&bytes).ok())
        .is_some_and(|cache| cache.version == 1
            && cache.sha256.eq_ignore_ascii_case(expected_sha256)
            && cache.fingerprint == fingerprint)
}

fn verify_model_integrity_with<F>(path: &Path, model: AiModelSpec, hash_file: F) -> Result<(), String>
where F: FnOnce(&Path) -> Result<String, String> {
    if model_verification_cache_matches(path, model.sha256) { return Ok(()); }
    let before = model_file_fingerprint(path)?;
    let actual = hash_file(path)?;
    if !actual.eq_ignore_ascii_case(model.sha256) {
        invalidate_model_verification(path);
        return Err(format!("{} failed its integrity check. Download the model again before running it.", model.name));
    }
    let after = model_file_fingerprint(path)?;
    if before != after {
        invalidate_model_verification(path);
        return Err("The local model changed while it was being verified. Try again.".into());
    }
    write_model_digest_sidecar(path, model.sha256)?;
    write_model_verification_cache_with_fingerprint(path, model.sha256, after)
}

fn adopt_cached_model_with<F, C>(path: Option<PathBuf>, verify: F, is_cancelled: C) -> Result<bool, String>
where
    F: FnOnce(&Path) -> Result<(), String>,
    C: Fn() -> bool,
{
    let Some(path) = path else { return Ok(false); };
    match verify(&path) {
        Ok(()) if is_cancelled() => Err("AI processing was cancelled.".to_string()),
        Ok(()) => Ok(true),
        Err(error) if is_cancelled() => Err(error),
        Err(_) => Ok(false),
    }
}

fn adopt_cached_model_for_task(app: &AppHandle, root: &Path, model: AiModelSpec) -> Result<bool, String> {
    ensure_ai_not_cancelled(app)?;
    emit_ai_progress(app, "verify", format!("Verifying {}", model.name), 0, None);
    let path = ai_model_path(root, model.id);
    let adopted = adopt_cached_model_with(
        path,
        |path| verify_model_integrity_for_task(app, path, model),
        || app.state::<AiState>().cancel_requested.load(Ordering::SeqCst),
    )?;
    ensure_ai_not_cancelled(app)?;
    Ok(adopted)
}

fn write_model_digest_sidecar(path: &Path, expected_sha256: &str) -> Result<(), String> {
    fs::write(model_digest_sidecar(path), expected_sha256)
        .map_err(|error| format!("Could not save the verified model digest: {error}"))
}

#[cfg(test)]
fn verify_model_integrity(path: &Path, model: AiModelSpec) -> Result<(), String> {
    verify_model_integrity_with(path, model, sha256_file)
}

fn verify_model_integrity_for_task(app: &AppHandle, path: &Path, model: AiModelSpec) -> Result<(), String> {
    verify_model_integrity_with(path, model, |path| sha256_file_for_task(app, path, &format!("Verifying {}", model.name)))
}

fn backend_runtime_path(root: &Path, backend: AiComputeBackend) -> Option<PathBuf> {
    let runtime = root.join("runtime");
    match backend {
        AiComputeBackend::Cpu => Some(runtime.join("cpu").join("current")),
        AiComputeBackend::Cuda => nvidia_cuda_capability()
            .ok()
            .map(|tag| runtime.join(format!("cuda-{tag}")).join("current")),
    }
}

fn backend_runtime_ready(root: &Path, backend: AiComputeBackend) -> bool {
    backend_runtime_path(root, backend)
        .is_some_and(|path| runtime_integrity_manifest(&path).is_ok_and(|manifest| {
            manifest.backend == backend.as_str() && manifest.files.keys().any(|relative| {
                Path::new(relative).components().all(|component| matches!(component, std::path::Component::Normal(_)))
                    && Path::new(relative).file_name().and_then(|name| name.to_str())
                    .is_some_and(|name| name.eq_ignore_ascii_case("llama-server.exe"))
                    && path.join(relative).is_file()
            })
        }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuMemory {
    total_mib: u64,
    free_mib: u64,
}

const GPU_TOTAL_REPORT_TOLERANCE_MIB: u64 = 256;

fn parse_gpu_memory_report(report: &str) -> Option<GpuMemory> {
    report.lines().find_map(|line| {
        let mut fields = line.split(',').map(str::trim);
        let total_mib = fields.next()?.parse().ok()?;
        let free_mib = fields.next()?.parse().ok()?;
        (total_mib >= free_mib).then_some(GpuMemory { total_mib, free_mib })
    })
}

fn gpu_memory() -> Option<GpuMemory> {
    let mut command = Command::new("nvidia-smi");
    command.arg("--query-gpu=memory.total,memory.free")
        .arg("--format=csv,noheader,nounits")
        .stdout(Stdio::piped()).stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    parse_gpu_memory_report(&String::from_utf8_lossy(&output.stdout))
}

fn cuda_layers_for_model(_model: AiModelSpec, _free_mib: Option<u64>) -> u32 {
    // Request all layers only after the free-memory gate; the server must
    // confirm positive offload and can still fail safely on allocation.
    99
}

fn gpu_eligibility(model: AiModelSpec, memory: Option<GpuMemory>) -> (bool, String) {
    // Eligibility describes hardware capacity, not temporary usage by a warm
    // model. Cold loads separately check current free memory before loading.
    match memory {
        Some(memory) if memory.total_mib.saturating_add(GPU_TOTAL_REPORT_TOLERANCE_MIB) >= model.minimum_gpu_mib => (true, String::new()),
        Some(_) => (false, "This model needs more GPU memory. Choose CPU or a smaller model.".into()),
        None => (false, "GPU memory unavailable. Choose CPU.".into()),
    }
}

fn ai_model_info(root: &Path, model: AiModelSpec, backend: Option<AiComputeBackend>, memory: Option<GpuMemory>) -> AiModelInfo {
    let (gpu_eligible, gpu_reason) = gpu_eligibility(model, memory);
    let runtime_ready = backend.map_or_else(
        || backend_runtime_ready(root, AiComputeBackend::Cpu) || backend_runtime_ready(root, AiComputeBackend::Cuda),
        |backend| backend_runtime_ready(root, backend),
    );
    // Catalog is metadata-only. A file at the exact canonical path can be
    // adopted by SHA-256 only after an explicit Clean/Translate action.
    let model_path = ai_model_path(root, model.id);
    let model_cached = model_path.is_some();
    let verified_cached = model_path.as_deref()
        .is_some_and(|path| model_sidecar_matches(path, model.sha256));
    AiModelInfo {
        id: model.id.to_string(),
        name: model.name.to_string(),
        size_label: model.size_label.to_string(),
        memory_label: model.memory_label.to_string(),
        recommendation: model.recommendation.to_string(),
        minimum_gpu_mib: model.minimum_gpu_mib,
        installed: runtime_ready && verified_cached,
        model_cached,
        gpu_eligible,
        gpu_reason,
        gpu_total_mib: memory.map(|memory| memory.total_mib),
        gpu_free_mib: memory.map(|memory| memory.free_mib),
    }
}

fn begin_ai_task(app: &AppHandle) -> Result<(), String> {
    let pipeline = app.state::<PipelineState>();
    let pipeline_active = lock(&pipeline.active_pid)?;
    if pipeline_active.is_some() {
        return Err("Video OCR is running. Start local AI after extraction finishes.".into());
    }
    let state = app.state::<AiState>();
    let mut active = lock(&state.active_pid)?;
    if state.closing.load(Ordering::SeqCst) {
        return Err("The application is closing.".into());
    }
    if active.is_some() {
        return Err("Another AI task is already running.".to_string());
    }
    state.cancel_requested.store(false, Ordering::SeqCst);
    *active = Some(0);
    drop(pipeline_active);
    Ok(())
}

// Close and cancel observe the same lock used to spawn/register a server.
fn register_ai_launch<T, F>(state: &AiState, launch: F) -> Result<T, String>
where F: FnOnce() -> Result<(T, u32), String> {
    let mut active = lock(&state.active_pid)?;
    if state.closing.load(Ordering::SeqCst) || state.cancel_requested.load(Ordering::SeqCst) {
        return Err("AI processing was cancelled.".into());
    }
    let (instance, pid) = launch()?;
    *active = Some(pid);
    Ok(instance)
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

fn sha256_file_for_task(app: &AppHandle, path: &Path, label: &str) -> Result<String, String> {
    ensure_ai_not_cancelled(app)?;
    let mut file = fs::File::open(path)
        .map_err(|error| format!("Could not open the downloaded file: {error}"))?;
    let total = file.metadata()
        .map_err(|error| format!("Could not inspect the downloaded file: {error}"))?.len();
    emit_ai_progress(app, "verify", label, 0, Some(total));
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut received = 0_u64;
    loop {
        ensure_ai_not_cancelled(app)?;
        let read = file.read(&mut buffer)
            .map_err(|error| format!("Could not verify the downloaded file: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        received += read as u64;
        if received % (16 * 1024 * 1024) < read as u64 || received == total {
            emit_ai_progress(app, "verify", label, received, Some(total));
        }
    }
    ensure_ai_not_cancelled(app)?;
    Ok(format!("{:x}", hasher.finalize()))
}

struct PendingDownload {
    path: PathBuf,
    file: Option<fs::File>,
}

impl Drop for PendingDownload {
    fn drop(&mut self) {
        // Close the handle first so Windows can remove the partial file.
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

fn copy_download_limited<R: Read, W: Write, F: FnMut(u64) -> Result<(), String>>(
    input: &mut R, output: &mut W, maximum_bytes: u64, mut progress: F,
) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut received = 0_u64;
    let mut buffer = [0_u8; 256 * 1024];
    progress(0)?;
    loop {
        let read = match input.read(&mut buffer) {
            Ok(read) => read,
            Err(error) => {
                progress(received)?;
                return Err(format!("Could not read the download: {error}"));
            }
        };
        progress(received)?;
        if read == 0 { break; }
        let next = received.checked_add(read as u64)
            .filter(|size| *size <= maximum_bytes)
            .ok_or("The download exceeds the allowed file size.")?;
        output.write_all(&buffer[..read])
            .map_err(|error| format!("Could not write the download: {error}"))?;
        hasher.update(&buffer[..read]);
        received = next;
        progress(received)?;
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
    if destination.is_file() && sha256_file_for_task(app, destination, &format!("Verifying {label}"))?.eq_ignore_ascii_case(expected_sha256) {
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
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|error| format!("Could not create a download token: {error}"))?.as_nanos();
    let partial = destination.with_extension(format!("{}.{}.download", std::process::id(), nonce));
    let mut response = client.get(url).send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("Download failed: {error}"))?;
    ensure_ai_not_cancelled(app)?;
    let total = response.content_length();
    // Pinned models are at most about 9 GB; runtime archives are smaller.
    let maximum_bytes = if phase == "model" { 12_u64 * 1024 * 1024 * 1024 }
        else { 4_u64 * 1024 * 1024 * 1024 };
    if total.is_some_and(|bytes| bytes > maximum_bytes) {
        return Err("The download exceeds the allowed file size.".into());
    }
    let output = fs::OpenOptions::new().write(true).create_new(true).open(&partial)
        .map_err(|error| format!("Could not create the download file: {error}"))?;
    let mut pending = PendingDownload { path: partial.clone(), file: Some(output) };
    let output = pending.file.as_mut().ok_or("The download file is unavailable.")?;
    let actual = copy_download_limited(&mut response, output, maximum_bytes, |received| {
        ensure_ai_not_cancelled(app)?;
        emit_ai_progress(app, phase, label, received, total);
        Ok(())
    })?;
    output.sync_all().map_err(|error| format!("Could not finish the download: {error}"))?;
    drop(pending.file.take());
    ensure_ai_not_cancelled(app)?;
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

fn nvidia_cuda_capability() -> Result<&'static str, String> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map_err(|_| "Local Qwen3 AI requires an NVIDIA GPU and nvidia-smi. Install a supported NVIDIA driver, then try again; CPU fallback is disabled.".to_string())?;
    if !output.status.success() || String::from_utf8_lossy(&output.stdout).trim().is_empty() {
        return Err("Local Qwen3 AI could not detect an NVIDIA GPU through nvidia-smi. Install or repair a supported NVIDIA driver; CPU fallback is disabled.".to_string());
    }
    let output = Command::new("nvidia-smi")
        .output()
        .map_err(|_| "Could not read NVIDIA CUDA driver capability from nvidia-smi.".to_string())?;
    let report = String::from_utf8_lossy(&output.stdout);
    let versions = report.lines().filter_map(|line| {
        let lower = line.to_ascii_lowercase();
        let marker = lower.find("cuda umd version:").map(|at| (at, "cuda umd version:"))
            .or_else(|| lower.find("cuda version:").map(|at| (at, "cuda version:")))?;
        let value = line[marker.0 + marker.1.len()..].trim().split_whitespace().next()?;
        let (major, minor) = value.split_once('.')?;
        Some((major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?, value))
    }).collect::<Vec<_>>();
    let (major, minor, version) = versions.iter().copied().max_by_key(|(major, minor, _)| (*major, *minor))
        .ok_or("Could not read the CUDA version from nvidia-smi. Update the NVIDIA driver; CPU fallback is disabled.")?;
    if (major, minor) >= (13, 3) {
        Ok(CUDA_MAX_TAG)
    } else if (major, minor) >= (12, 4) {
        Ok(CUDA_MIN_TAG)
    } else {
        Err(format!("Local Qwen3 GPU inference requires NVIDIA CUDA 12.4 or newer; nvidia-smi reports {version}. Update the NVIDIA driver; CPU fallback is disabled."))
    }
}

fn fetch_llama_assets(
    client: &reqwest::blocking::Client,
    backend: AiComputeBackend,
    cuda_tag: Option<&str>,
) -> Result<Vec<GithubAsset>, String> {
    let release_url = format!(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_RELEASE_TAG}"
    );
    let release: GithubRelease = client
        .get(release_url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| {
            format!("Could not check the official llama.cpp release: {error}")
        })?
        .json()
        .map_err(|error| {
            format!("Could not read the llama.cpp release manifest: {error}")
        })?;
    let expected_names = match backend {
        AiComputeBackend::Cpu => vec![format!("llama-{LLAMA_RELEASE_TAG}-bin-win-cpu-x64.zip")],
        AiComputeBackend::Cuda => {
            let tag = cuda_tag.ok_or("The llama.cpp CUDA runtime version was not selected.")?;
            if tag != CUDA_MIN_TAG && tag != CUDA_MAX_TAG {
                return Err("The selected llama.cpp CUDA runtime is not supported.".to_string());
            }
            vec![
                format!("llama-{LLAMA_RELEASE_TAG}-bin-win-cuda-{tag}-x64.zip"),
                format!("cudart-llama-bin-win-cuda-{tag}-x64.zip"),
            ]
        }
    };
    expected_names
        .iter()
        .map(|expected_name| {
            let asset = release
                .assets
                .iter()
                .find(|asset| asset.name.eq_ignore_ascii_case(expected_name))
                .cloned()
                .ok_or_else(|| {
                    format!("The pinned llama.cpp release is missing required {backend:?} asset {expected_name}; the selected backend cannot continue.")
                })?;
            let expected_url = format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE_TAG}/{expected_name}"
            );
            if asset.browser_download_url != expected_url {
                return Err("The llama.cpp runtime manifest returned an unexpected download URL.".to_string());
            }
            let digest = asset.digest.as_deref().and_then(|value| value.strip_prefix("sha256:"));
            if digest.map_or(true, |value| !valid_sha256(value)) {
                return Err(format!("The official llama.cpp asset {expected_name} does not provide a valid SHA-256 digest; it was not installed."));
            }
            Ok(asset)
        })
        .collect()
}

fn valid_sha256(digest: &str) -> bool {
    digest.len() == 64 && digest.chars().all(|character| character.is_ascii_hexdigit())
}

fn runtime_manifest_path(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(value) => parts.push(value.to_string_lossy().to_string()),
            _ => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn collect_runtime_digests(root: &Path, current: &Path, app: Option<&AppHandle>) -> Result<BTreeMap<String, String>, String> {
    let mut pending = vec![(current.to_path_buf(), 0usize)];
    let mut seen = 0usize;
    let mut files = BTreeMap::new();
    while let Some((directory, depth)) = pending.pop() {
        let entries = fs::read_dir(directory)
            .map_err(|error| format!("Could not read the installed llama.cpp runtime: {error}"))?;
        for entry in entries {
            if let Some(app) = app { ensure_ai_not_cancelled(app)?; }
            let entry = entry
                .map_err(|error| format!("Could not inspect the llama.cpp runtime: {error}"))?;
            seen += 1;
            if seen > 8192 {
                return Err("The llama.cpp runtime contains too many entries.".to_string());
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("Could not inspect a llama.cpp runtime file: {error}"))?;
            if runtime_reparse_point(&metadata) {
                return Err("The llama.cpp runtime contains a link or reparse point.".to_string());
            }
            if metadata.is_dir() {
                if depth >= 12 {
                    return Err("The llama.cpp runtime directory is too deep.".to_string());
                }
                pending.push((path, depth + 1));
            } else if metadata.is_file() {
                if path.file_name().is_some_and(|name| name == LLAMA_RUNTIME_MANIFEST) {
                    continue;
                }
                let relative = path
                    .strip_prefix(root)
                    .ok()
                    .and_then(runtime_manifest_path)
                    .ok_or("Could not record a llama.cpp runtime file path.")?;
                let digest = if let Some(app) = app {
                    sha256_file_for_task(app, &path, "Verifying AI engine files")?
                } else {
                    sha256_file(&path)?
                };
                files.insert(relative, digest);
            }
        }
    }
    Ok(files)
}

fn write_runtime_integrity_manifest(
    runtime_root: &Path,
    backend: AiComputeBackend,
    cuda_tag: Option<&str>,
    source_archives_sha256: BTreeMap<String, String>,
    app: Option<&AppHandle>,
) -> Result<(), String> {
    let expected_archive_count = if backend == AiComputeBackend::Cuda { 2 } else { 1 };
    if source_archives_sha256.len() != expected_archive_count
        || source_archives_sha256.values().any(|digest| !valid_sha256(digest))
    {
        return Err("A verified llama.cpp runtime archive digest is invalid.".to_string());
    }
    let manifest = RuntimeIntegrityManifest {
        release_tag: LLAMA_RELEASE_TAG.to_string(),
        backend: backend.as_str().to_string(),
        cuda_tag: cuda_tag.unwrap_or_default().to_string(),
        archive_sha256: source_archives_sha256
            .into_iter()
            .map(|(name, digest)| (name, digest.to_ascii_lowercase()))
            .collect(),
        files: collect_runtime_digests(runtime_root, runtime_root, app)?,
    };
    let contents = serde_json::to_vec(&manifest)
        .map_err(|error| format!("Could not prepare the llama.cpp runtime integrity manifest: {error}"))?;
    fs::write(runtime_root.join(LLAMA_RUNTIME_MANIFEST), contents)
        .map_err(|error| format!("Could not save the llama.cpp runtime integrity manifest: {error}"))
}

fn runtime_integrity_manifest(installed: &Path) -> Result<RuntimeIntegrityManifest, String> {
    let manifest_path = installed.join(LLAMA_RUNTIME_MANIFEST);
    let contents = fs::read(manifest_path)
        .map_err(|_| "The llama.cpp AI engine integrity manifest is missing.".to_string())?;
    let manifest: RuntimeIntegrityManifest = serde_json::from_slice(&contents)
        .map_err(|_| "The llama.cpp AI engine integrity manifest is invalid.".to_string())?;
    let expected_archives = if manifest.backend == "cuda" { 2 } else { 1 };
    if manifest.release_tag != LLAMA_RELEASE_TAG
        || !matches!(manifest.backend.as_str(), "cuda" | "cpu")
        || (manifest.backend == "cuda" && manifest.cuda_tag != CUDA_MIN_TAG && manifest.cuda_tag != CUDA_MAX_TAG)
        || (manifest.backend == "cpu" && !manifest.cuda_tag.is_empty())
        || manifest.archive_sha256.len() != expected_archives
        || manifest.archive_sha256.values().any(|digest| !valid_sha256(digest))
        || manifest.files.is_empty()
    {
        return Err("The llama.cpp AI engine integrity manifest is invalid.".to_string());
    }
    Ok(manifest)
}

fn verify_llama_runtime_files(installed: &Path, verify_all: bool, app: Option<&AppHandle>) -> Result<PathBuf, String> {
    let manifest = runtime_integrity_manifest(installed)?;
    for (relative, expected) in &manifest.files {
        if !valid_sha256(expected) {
            return Err("The llama.cpp AI engine integrity manifest is invalid.".to_string());
        }
        if Path::new(relative)
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err("The llama.cpp AI engine integrity manifest contains an unsafe path.".to_string());
        }
    }
    let cli_relative = manifest.files.keys()
        .find(|relative| Path::new(relative).file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("llama-server.exe")))
        .ok_or("The llama.cpp runtime manifest does not contain llama-server.exe.")?;
    let cli = installed.join(cli_relative);
    let mut component_path = installed.to_path_buf();
    for component in Path::new(cli_relative).components() {
        component_path.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&component_path)
            .map_err(|_| "The verified llama.cpp AI engine is missing llama-server.exe.")?;
        if runtime_reparse_point(&metadata) {
            return Err("The llama.cpp AI engine path contains a link or reparse point.".to_string());
        }
    }
    if !fs::symlink_metadata(&cli).is_ok_and(|metadata| metadata.is_file()) {
        return Err("The verified llama.cpp AI engine is missing llama-server.exe.".to_string());
    }
    let expected_cli = &manifest.files[cli_relative];
    if !valid_sha256(expected_cli) {
        return Err("The llama.cpp AI engine integrity manifest is invalid.".to_string());
    }
    if verify_all {
        let actual_files = collect_runtime_digests(installed, installed, app)?;
        if actual_files != manifest.files {
            return Err("A llama.cpp AI engine file failed its integrity check.".to_string());
        }
    } else {
        let actual_cli_digest = if let Some(app) = app {
            sha256_file_for_task(app, &cli, "Verifying AI engine")?
        } else {
            sha256_file(&cli)?
        };
        if !actual_cli_digest.eq_ignore_ascii_case(expected_cli) {
            return Err("The llama.cpp AI engine executable failed its integrity check.".to_string());
        }
    }
    Ok(cli)
}

fn extract_runtime(
    app: &AppHandle,
    archives: &[(&Path, &str)],
    runtime_root: &Path,
    backend: AiComputeBackend,
    cuda_tag: Option<&str>,
    archive_sha256: BTreeMap<String, String>,
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
    for (archive, expected_digest) in archives {
        let file = fs::File::open(archive)
            .map_err(|error| format!("Could not open a llama.cpp runtime archive: {error}"))?;
        if !sha256_file_for_task(app, archive, "Verifying cached AI engine")?.eq_ignore_ascii_case(expected_digest) {
            return Err("A cached llama.cpp runtime archive failed its integrity check.".to_string());
        }
        let mut zip = zip::ZipArchive::new(file)
            .map_err(|error| format!("Could not read a llama.cpp runtime archive: {error}"))?;
        for index in 0..zip.len() {
            ensure_ai_not_cancelled(app)?;
            let mut entry = zip
                .by_index(index)
                .map_err(|error| format!("Could not read a llama.cpp runtime archive entry: {error}"))?;
            let relative = entry
                .enclosed_name()
                .ok_or("A llama.cpp runtime archive contains an unsafe path.")?;
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
    }
    let cli = find_file(&staging, "llama-server.exe")
        .ok_or("The verified llama.cpp AI runtime does not contain llama-server.exe.")?;
    write_runtime_integrity_manifest(&staging, backend, cuda_tag, archive_sha256, Some(app))?;
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
    backend: AiComputeBackend,
) -> Result<PathBuf, String> {
    let cuda_tag = if backend == AiComputeBackend::Cuda {
        Some(nvidia_cuda_capability()?)
    } else {
        None
    };
    let runtime_name = match (backend, cuda_tag) {
        (AiComputeBackend::Cuda, Some(tag)) => format!("cuda-{tag}"),
        (AiComputeBackend::Cpu, None) => "cpu".to_string(),
        _ => return Err("Could not select the requested local AI compute backend.".to_string()),
    };
    let runtime_root = root.join("runtime").join(runtime_name);
    let installed = runtime_root.join("current");
    if runtime_integrity_manifest(&installed).is_ok_and(|manifest| {
        manifest.backend == backend.as_str() && manifest.cuda_tag == cuda_tag.unwrap_or_default()
    }) {
        match verify_llama_runtime_files(&installed, false, Some(app)) {
            Ok(cli) => {
                ensure_ai_not_cancelled(app)?;
                return Ok(cli);
            }
            Err(error) if app.state::<AiState>().cancel_requested.load(Ordering::SeqCst) => return Err(error),
            Err(_) => {}
        }
    }
    ensure_ai_not_cancelled(app)?;
    emit_ai_progress(
        app,
        "runtime",
        "Checking the compatible local AI engine",
        0,
        None,
    );
    let assets = fetch_llama_assets(client, backend, cuda_tag)?;
    let downloads = runtime_root.join("downloads");
    let mut archives = Vec::new();
    let mut digests = BTreeMap::new();
    for (index, asset) in assets.into_iter().enumerate() {
        let digest = asset.digest.as_deref()
            .and_then(|value| value.strip_prefix("sha256:"))
            .ok_or_else(|| format!("The official llama.cpp CUDA asset {} does not provide a SHA-256 digest; GPU inference cannot continue.", asset.name))?;
        let archive = downloads.join(&asset.name);
        download_verified(
            app,
            client,
            &asset.browser_download_url,
            &archive,
            digest,
            "runtime",
            match (backend, index) {
                (AiComputeBackend::Cuda, 0) => "Downloading the llama.cpp CUDA inference engine",
                (AiComputeBackend::Cuda, _) => "Downloading the llama.cpp CUDA runtime libraries",
                (AiComputeBackend::Cpu, _) => "Downloading the llama.cpp CPU inference engine",
            },
        )?;
        digests.insert(asset.name, digest.to_ascii_lowercase());
        archives.push((archive, digest.to_string()));
    }
    emit_ai_progress(
        app,
        "runtime",
        "Installing the compatible local AI engine",
        1,
        Some(1),
    );
    let refs = archives.iter().map(|(path, digest)| (path.as_path(), digest.as_str())).collect::<Vec<_>>();
    extract_runtime(app, &refs, &runtime_root, backend, cuda_tag, digests)
}

#[cfg(test)]
fn is_llama_runtime(installed: &Path) -> bool {
    verify_llama_runtime_files(installed, false, None).is_ok()
}

fn require_llama_runtime(app: &AppHandle, root: &Path, backend: AiComputeBackend) -> Result<(PathBuf, String), String> {
    let runtime_base = root.join("runtime");
    let installed = match backend {
        AiComputeBackend::Cpu => runtime_base.join("cpu").join("current"),
        AiComputeBackend::Cuda => {
            let tag = nvidia_cuda_capability()?;
            runtime_base.join(format!("cuda-{tag}")).join("current")
        }
    };
    let cli = verify_llama_runtime_files(&installed, true, Some(app)).map_err(|error| {
        if app.state::<AiState>().cancel_requested.load(Ordering::SeqCst) {
            return error;
        }
        let _ = fs::write(installed.join(LLAMA_RUNTIME_MANIFEST), b"invalid");
        format!("{error} Download the selected model again to repair the AI engine.")
    })?;
    let manifest = runtime_integrity_manifest(&installed)?;
    if manifest.backend != backend.as_str() {
        return Err(format!("The installed local AI engine is {}, but {} was selected. Download the model with the selected compute backend first.", manifest.backend, backend.as_str()));
    }
    Ok((cli, manifest.cuda_tag))
}

fn model_download_url(
    repository: &str,
    revision: &str,
    file_path: &str,
) -> Result<String, String> {
    let mut url = reqwest::Url::parse(&format!(
        "https://huggingface.co/{repository}/resolve/{revision}"
    ))
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
        "https://huggingface.co/api/models/{}/tree/{}?recursive=true&expand=false",
        model.repository, model.revision
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
            entry.entry_type == "file" && entry.path.eq_ignore_ascii_case(model.filename)
        })
        .next()
        .ok_or("The official model repository does not contain the selected GGUF file.")?;
    // Hugging Face may store a pinned file through either LFS or Xet. Xet tree
    // entries do not always expose an LFS oid; the pinned SHA-256 is still
    // enforced on the downloaded bytes by download_verified.
    if let Some(digest) = entry.lfs.as_ref().map(|lfs| lfs.oid.as_str()) {
        let digest = digest.strip_prefix("sha256:").unwrap_or(digest);
        if !digest.eq_ignore_ascii_case(model.sha256) {
            return Err(format!(
                "The selected official model revision has an unexpected SHA-256 digest for {}.",
                model.filename
            ));
        }
    }
    let url = model_download_url(model.repository, model.revision, &entry.path)?;
    Ok((entry.path, url, model.sha256.to_string()))
}

fn install_ai_model_blocking(
    app: AppHandle,
    model_id: String,
    backend: AiComputeBackend,
) -> Result<AiModelInfo, String> {
    begin_ai_task(&app)?;
    let result = (|| {
        release_warm_ai(&app)?;
        let root = ai_storage_root(&app)?;
        let model = ai_model_spec(&model_id)?;
        if backend == AiComputeBackend::Cuda {
            let (eligible, reason) = gpu_eligibility(model, gpu_memory());
            if !eligible {
                return Err(reason);
            }
        }
        let client = reqwest::blocking::Client::builder()
            .user_agent(AI_USER_AGENT)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|error| format!("Could not initialize secure downloads: {error}"))?;
        ensure_llama_runtime(&app, &client, &root, backend)?;
        // A legacy digest sidecar needs a one-time content hash before the
        // durable metadata fingerprint can enable the cold-run fast path.
        let needs_download = !adopt_cached_model_for_task(&app, &root, model)?;
        if needs_download {
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
            write_model_digest_sidecar(&destination, &digest)?;
            write_model_verification_cache(&destination, &digest)?;
        }
        ensure_ai_not_cancelled(&app)?;
        emit_ai_progress(
            &app,
            "complete",
            format!("{} is ready", model.name),
            1,
            Some(1),
        );
        Ok(ai_model_info(&root, model, Some(backend), gpu_memory()))
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
    }).or_else(|| {
        serde_json::from_str(&escape_json_string_controls(value)).ok()
    }).or_else(|| {
        join_json_string_fragments(value)
            .and_then(|joined| serde_json::from_str(&escape_json_string_controls(&joined)).ok())
    }).or_else(|| {
        let mut candidate = value.to_string();
        let mut changed = false;
        for _ in 0..32 {
            let Some(next) = split_merged_ai_cues(&candidate) else { break; };
            candidate = next;
            changed = true;
        }
        changed.then(|| serde_json::from_str(&escape_json_string_controls(&candidate)).ok()).flatten()
    })
}

fn split_merged_ai_cues(value: &str) -> Option<String> {
    // Qwen3 4B has emitted {"id":23,"text":"","id":24,"text":"..."}.
    // Split only after a complete cue object and before another id field.
    // Exact target IDs and order are validated after parsing.
    let mut object_start = None;
    let (mut depth, mut in_string, mut escaped) = (0_usize, false, false);
    for (index, ch) in value.char_indices() {
        if in_string {
            if escaped { escaped = false; }
            else if ch == '\\' { escaped = true; }
            else if ch == '"' { in_string = false; }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 { object_start = Some(index); }
                depth += 1;
            }
            '}' => { depth = depth.saturating_sub(1); },
            ',' if depth == 1 => {
                let next = value[index + 1..].trim_start();
                if next.starts_with("\"id\"")
                    && next[4..].trim_start().starts_with(':')
                    && object_start.is_some_and(|start| {
                        let candidate = format!("{}}}", &value[start..index]);
                        serde_json::from_str::<AiCueOutput>(&candidate).is_ok()
                    })
                {
                    return Some(format!("{}{}{}", &value[..index], "},{", &value[index + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

fn join_json_string_fragments(value: &str) -> Option<String> {
    // Recover only the observed `"text":"first"+"\\nsecond"` shape. The
    // transformed candidate still must parse as JSON and pass exact cue IDs.
    let chars: Vec<char> = value.chars().collect();
    let (mut index, mut inside, mut escaped, mut changed) = (0, false, false, false);
    let mut result = String::with_capacity(value.len());
    while index < chars.len() {
        let ch = chars[index];
        if inside && !escaped && ch == '"' {
            let mut next = index + 1;
            while next < chars.len() && chars[next].is_whitespace() { next += 1; }
            if next < chars.len() && chars[next] == '+' {
                next += 1;
                while next < chars.len() && chars[next].is_whitespace() { next += 1; }
                if next < chars.len() && chars[next] == '"' {
                    index = next + 1;
                    changed = true;
                    continue;
                }
            }
        }
        result.push(ch);
        if ch == '"' && !escaped { inside = !inside; }
        if ch == '\\' { escaped = !escaped; } else { escaped = false; }
        index += 1;
    }
    changed.then_some(result)
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
    // Small models sometimes return a single JSON object after a group has
    // been subdivided to one cue. Target-count and ID checks still happen in
    // select_ai_target_response, so this cannot silently accept a partial group.
    if let Some(end) = without_thinking.rfind('}') {
        for (start, character) in without_thinking[..=end].char_indices().rev() {
            if character != '{' {
                continue;
            }
            let candidate = &without_thinking[start..=end];
            let parsed = serde_json::from_str::<AiCueOutput>(candidate).ok().or_else(|| {
                repair_smart_quote_terminators(candidate)
                    .and_then(|repaired| serde_json::from_str(&repaired).ok())
            }).or_else(|| {
                serde_json::from_str(&escape_json_string_controls(candidate)).ok()
            });
            if let Some(cue) = parsed {
                return Ok(vec![cue]);
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

fn escape_json_string_controls(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    let (mut in_string, mut after_backslash) = (false, false);
    for character in value.chars() {
        if in_string && !after_backslash {
            match character {
                '\n' => { escaped.push_str("\\n"); continue; }
                '\r' => { escaped.push_str("\\r"); continue; }
                '\t' => { escaped.push_str("\\t"); continue; }
                _ => {}
            }
        }
        if character == '"' && !after_backslash { in_string = !in_string; }
        let was_backslash = character == '\\';
        escaped.push(character);
        after_backslash = was_backslash && !after_backslash;
    }
    escaped
}

fn parse_translation_output(value: &str, id: usize, source: &str) -> Result<String, String> {
    let answer = assistant_output(value);
    let answer = answer.rsplit_once("</think>").map_or(answer, |(_, text)| text).trim();
    if answer.starts_with('{') || answer.starts_with('[') {
        let repaired = escape_json_string_controls(answer);
        let parsed = clean_model_output(&repaired)?;
        if parsed.len() != 1 || parsed[0].id != id || parsed[0].text.trim().is_empty() {
            return Err("The model returned an invalid translated cue.".to_string());
        }
        return Ok(parsed[0].text.trim().to_string());
    }
    if answer.is_empty()
        || answer.starts_with("```")
        || answer.contains("\nUser:")
        || answer.contains("\nAssistant:")
        || answer == format!("Subtitle:\n{}", source.trim())
    {
        return Err("The model returned no usable translated subtitle text.".to_string());
    }
    Ok(answer.to_string())
}

fn parse_single_cleanup_output(value: &str, id: usize) -> Result<String, String> {
    let answer = assistant_output(value);
    let answer = answer.rsplit_once("</think>").map_or(answer, |(_, text)| text).trim();
    if answer == "[[REMOVE]]" {
        return Ok(String::new());
    }
    if answer.starts_with('{') || answer.starts_with('[') {
        let parsed = clean_model_output(&escape_json_string_controls(answer))?;
        if parsed.len() != 1 || parsed[0].id != id {
            return Err("The model returned an invalid cleaned cue.".to_string());
        }
        return Ok(parsed[0].text.trim().to_string());
    }
    if answer.is_empty() || answer.starts_with("```")
        || answer.contains("\nUser:") || answer.contains("\nAssistant:")
        || answer.contains("Subtitle:\n")
    {
        return Err("The model returned no usable cleaned subtitle text.".to_string());
    }
    Ok(answer.to_string())
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
        "SubHooper=0.4.3-beta\nGeneratedUnixMs={timestamp}\nError={error}\nCLI={}\nModel={}\n\n--- OUTPUT FILE ---\n{}\n\n--- STDOUT ---\n{}\n\n--- STDERR ---\n{}\n",
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

fn cue_text_payload(cues: &[(usize, &SubtitleCue)]) -> Vec<String> {
    cues.iter().map(|(_, cue)| cue.text.join("\n")).collect()
}

fn needs_translation(cue: &SubtitleCue) -> bool {
    cue.text.iter().any(|line| line.chars().any(char::is_alphabetic))
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
    _previous_results: &[(usize, String)],
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
    match mode {
        "clean" => {
            let language = validate_cleanup_language(cleanup_language)?;
            let strength = validate_cleanup_strength(cleanup_strength)?;
            let language_instruction = if language == "Auto detect" {
                "Infer the dominant subtitle language from language samples and use it for every kept target cue.".to_string()
            } else {
                format!("Use {language} as the cleanup language for every kept target cue.")
            };
            let strength_instruction = match strength {
                "light" => "Use light cleanup. Correct clear OCR, spelling, punctuation, capitalization, broken-word, and line-break errors. Translate coherent foreign dialogue into the cleanup language, but remove a cue only when it is unmistakably non-text noise. Preserve uncertain fragments for manual review.",
                "balanced" => "Use balanced cleanup. Correct OCR and subtitle formatting, translate coherent foreign dialogue into the cleanup language, and remove high-confidence non-dialogue OCR noise. Preserve ambiguous material when it may still be meaningful dialogue.",
                "strong" => "Use strong cleanup. Produce a clean, single-language subtitle track. Translate coherent foreign dialogue into the cleanup language. Remove isolated symbols, short numeric or mixed-character debris, corrupted foreign-script fragments that cannot be translated confidently, scenery text, credits, logos, interface text, and fragments unrelated to dialogue. Do not keep meaningless text merely because it is uncertain. Preserve short text only when nearby dialogue makes its meaning clear. Never invent dialogue that is absent from the OCR input.",
                _ => unreachable!(),
            };
            let task = format!(
                "{language_instruction} {strength_instruction} Use surrounding cues only to determine language and context."
            );
            if targets.len() == 1 {
                let samples = if language == "Auto detect" {
                    document_samples.iter().take(8).copied().collect::<Vec<_>>()
                } else { Vec::new() };
                return Ok(format!(
                    "{language_instruction} {strength_instruction} Use language samples and surrounding cues only as read-only context; clean only the target subtitle and never return sample or context text. Reply with the cleaned target text only; reply exactly [[REMOVE]] only for clear non-dialogue noise. No JSON, Markdown, or explanation.\nGuidance: {guidance}\nLanguage samples: {}\nContext before: {}\nContext after: {}\nTarget subtitle:\n{}",
                    serde_json::to_string(&cue_text_payload(&samples)).map_err(|error| error.to_string())?,
                    serde_json::to_string(&cue_text_payload(context_before)).map_err(|error| error.to_string())?,
                    serde_json::to_string(&cue_text_payload(context_after)).map_err(|error| error.to_string())?,
                    targets[0].1.text.join("\n")
                ));
            }
            let samples = if language == "Auto detect" {
                document_samples.iter().take(8).copied().collect::<Vec<_>>()
            } else { Vec::new() };
            return Ok(format!(
                "{task} Use language samples and surrounding cues only as read-only context; clean only the input cues and never return sample or context text. Reply ONLY with a JSON array containing one {{\"id\":number,\"text\":string}} object for each input cue, using exactly the input ids in the same order. Use empty text only for clear non-dialogue noise. No wrapper, Markdown, or explanation. Escape line breaks as \\n.\nGuidance: {guidance}\nLanguage samples: {}\nContext before: {}\nContext after: {}\nInput: {}",
                serde_json::to_string(&cue_text_payload(&samples)).map_err(|error| error.to_string())?,
                serde_json::to_string(&cue_text_payload(context_before)).map_err(|error| error.to_string())?,
                serde_json::to_string(&cue_text_payload(context_after)).map_err(|error| error.to_string())?,
                serde_json::to_string(&cue_payload(targets)).map_err(|error| error.to_string())?,
            ));
        }
        "translate" => {
            let language = validate_translation_language(target_language)?;
            let source_hint = source_language
                .map(str::trim)
                .filter(|value| !value.is_empty() && *value != "Auto detect")
                .map(|value| format!(" Source: {value}."))
                .unwrap_or_default();
            let style = if language.eq_ignore_ascii_case("Turkish") {
                " Write natural, idiomatic Turkish with natural word order; avoid literal translations and keep dialogue concise."
            } else {
                " Use natural, idiomatic phrasing and concise subtitle dialogue; avoid literal translations."
            };
            let before = serde_json::to_string(&cue_text_payload(context_before))
                .map_err(|error| format!("Could not prepare subtitle context: {error}"))?;
            let after = serde_json::to_string(&cue_text_payload(context_after))
                .map_err(|error| format!("Could not prepare subtitle context: {error}"))?;
            if targets.len() == 1 {
                let (_, cue) = targets[0];
                return Ok(format!(
                "Translate this subtitle into {language}.{source_hint}{style} Keep names and meaning. Use the surrounding cues only as read-only context; translate only the target subtitle and never return context cues. Reply with translated subtitle text only, preserving line breaks. No JSON, quotation marks, Markdown, or explanation.\nGuidance: {guidance}\nContext before (read-only text): {before}\nContext after (read-only text): {after}\nTarget subtitle:\n{}",
                    cue.text.join("\n"),
                ));
            }
            return Ok(format!(
                "Translate each input subtitle into {language}.{source_hint}{style} Keep names and meaning. Use surrounding cues only as read-only context; translate only the input cues and never return context cues. Return ONLY a JSON array with one {{\"id\":number,\"text\":string}} object per input, using exactly the input ids in the same order. No wrapper, Markdown, or explanation. Escape line breaks inside strings as \\n.\nGuidance: {guidance}\nContext before (read-only text): {before}\nContext after (read-only text): {after}\nInput: {}",
                serde_json::to_string(&cue_payload(targets))
                    .map_err(|error| format!("Could not prepare subtitles for the model: {error}"))?
            ));
        }
        _ => Err("Select a valid AI task.".to_string()),
    }
}

fn new_ai_server_key() -> Result<String, String> {
    let mut random = [0_u8; 32];
    #[cfg(target_os = "windows")]
    {
        // Windows CNG system-preferred RNG; the secret is passed in the child
        // environment instead of appearing in its command line.
        let status = unsafe { BCryptGenRandom(std::ptr::null_mut(), random.as_mut_ptr(), random.len() as u32, 2) };
        if status < 0 { return Err("Could not secure the local AI server.".into()); }
    }
    #[cfg(not(target_os = "windows"))]
    {
        fs::File::open("/dev/urandom")
            .and_then(|mut source| source.read_exact(&mut random))
            .map_err(|error| format!("Could not secure the local AI server: {error}"))?;
    }
    Ok(random.iter().map(|byte| format!("{byte:02x}")).collect::<Vec<_>>().join(""))
}

#[cfg(target_os = "windows")]
fn ai_server_port_owned_by(port: u16, pid: u32) -> Result<bool, String> {
    // TCP_TABLE_OWNER_PID_LISTENER for AF_INET. Fail closed if Windows cannot
    // confirm that the child we spawned owns the loopback listening port.
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
    const AF_INET: u32 = 2;
    const TCP_TABLE_OWNER_PID_LISTENER: u32 = 3;
    let mut bytes = 0_u32;
    let first = unsafe { GetExtendedTcpTable(std::ptr::null_mut(), &mut bytes, 0,
        AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0) };
    if first != ERROR_INSUFFICIENT_BUFFER || bytes < 4 {
        return Err("Could not inspect the local AI server port owner.".into());
    }
    let capacity = (bytes as usize).div_ceil(4) * 4;
    let mut table = vec![0_u32; capacity / 4];
    let status = unsafe { GetExtendedTcpTable(table.as_mut_ptr().cast(), &mut bytes, 0,
        AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0) };
    if status != 0 || bytes as usize > capacity {
        return Err("Could not verify the local AI server port owner.".into());
    }
    let used = bytes as usize;
    let count = table[0] as usize;
    let row_size = std::mem::size_of::<TcpRowOwnerPid>();
    if count > used.saturating_sub(4) / row_size {
        return Err("The local AI server port table was invalid.".into());
    }
    for index in 0..count {
        let row = unsafe {
            std::ptr::read_unaligned((table.as_ptr() as *const u8).add(4 + index * row_size)
                as *const TcpRowOwnerPid)
        };
        if u16::from_be(row.local_port as u16) == port
            && row.local_address == u32::from_ne_bytes([127, 0, 0, 1]) {
            return Ok(row.owning_pid == pid);
        }
    }
    Ok(false)
}

#[cfg(not(target_os = "windows"))]
fn ai_server_port_owned_by(_port: u16, _pid: u32) -> Result<bool, String> {
    // The distributable application is Windows-only.
    Ok(true)
}

struct AiServer {
    child: Child,
    endpoint: String,
    port: u16,
    api_key: String,
    log_path: PathBuf,
    client: reqwest::blocking::Client,
}

struct WarmAiServer {
    model_id: String,
    backend: AiComputeBackend,
    instance: AiServer,
}

fn release_warm_ai(app: &AppHandle) -> Result<(), String> {
    let retired = lock(&app.state::<AiState>().warm)?.take();
    drop(retired);
    Ok(())
}

fn take_warm_ai(app: &AppHandle, model: AiModelSpec, backend: AiComputeBackend) -> Result<Option<AiServer>, String> {
    let warm = lock(&app.state::<AiState>().warm)?.take();
    if let Some(mut warm) = warm {
        if warm.model_id == model.id && warm.backend == backend
            && warm.instance.child.try_wait().ok().flatten().is_none()
            && ai_server_port_owned_by(warm.instance.port, warm.instance.child.id())?
            && warm.instance.client.get(format!("{}/health", warm.instance.endpoint))
                .timeout(Duration::from_secs(2)).send()
                .is_ok_and(|response| response.status().is_success()) {
            let pid = warm.instance.child.id();
            return register_ai_launch(&app.state::<AiState>(), || Ok((Some(warm.instance), pid)));
        }
    }
    Ok(None)
}

fn preserve_warm_ai(app: &AppHandle, model: AiModelSpec, backend: AiComputeBackend, mut instance: AiServer) -> Result<(), String> {
    let state = app.state::<AiState>();
    let mut warm = lock(&state.warm)?;
    if state.closing.load(Ordering::SeqCst) || state.cancel_requested.load(Ordering::SeqCst)
        || instance.child.try_wait().ok().flatten().is_some() {
        return Ok(());
    }
    *warm = Some(WarmAiServer {
        model_id: model.id.into(), backend, instance,
    });
    Ok(())
}

impl Drop for AiServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_ai_server(app: &AppHandle, server: &Path, model: &Path, backend: AiComputeBackend, gpu_layers: u32) -> Result<AiServer, String> {
    // Bind only the loopback interface, and reserve an available ephemeral port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("Could not reserve a local AI port: {error}"))?;
    let port = listener.local_addr().map_err(|error| format!("Could not read the AI port: {error}"))?.port();
    let endpoint = format!("http://127.0.0.1:{port}");
    let api_key = new_ai_server_key()?;
    let project = resolve_project_root()?;
    let results_root = resolve_results_root(&project)?;
    let reports_root = resolve_reports_root(&results_root)?;
    let log_path = reports_root.join(format!("subhooper-ai-server-{}-{}.log", std::process::id(), port));
    let log = fs::File::create(&log_path).map_err(|error| format!("Could not capture the local AI engine log: {error}"))?;
    let stderr = log.try_clone().map_err(|error| format!("Could not capture AI errors: {error}"))?;
    let mut command = Command::new(server);
    let threads = std::thread::available_parallelism().map_or(4, |count| count.get().min(if backend == AiComputeBackend::Cuda { 4 } else { 8 }));
    command.args(["-m"]).arg(model).args(["-ngl", &gpu_layers.to_string(), "-c", "4096", "--jinja", "--host", "127.0.0.1", "--port", &port.to_string(), "--no-webui", "--parallel", "1", "--threads", &threads.to_string(), "--log-verbosity", "4"])
        .env("LLAMA_API_KEY", &api_key)
        .stdin(Stdio::null()).stdout(Stdio::from(log)).stderr(Stdio::from(stderr));
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    drop(listener);
    let client = reqwest::blocking::Client::builder()
        .no_proxy().connect_timeout(Duration::from_secs(1)).timeout(Duration::from_secs(600))
        .build().map_err(|error| format!("Could not create the local AI client: {error}"))?;
    let child = register_ai_launch(&app.state::<AiState>(), || {
        let child = command.spawn().map_err(|error| format!("Could not start the local AI server: {error}"))?;
        let pid = child.id();
        Ok((child, pid))
    })?;
    let mut instance = AiServer { child, endpoint, port, api_key, log_path, client };
    for _ in 0..600 {
        ensure_ai_not_cancelled(app)?;
        if let Ok(Some(status)) = instance.child.try_wait() {
            return Err(format!("The local AI server exited during model load ({status}). Log: {}", instance.log_path.display()));
        }
        if !ai_server_port_owned_by(instance.port, instance.child.id())? {
            thread::sleep(Duration::from_millis(200));
            continue;
        }
        if instance.client.get(format!("{}/health", instance.endpoint)).timeout(Duration::from_millis(500)).send().is_ok_and(|response| response.status().is_success()) {
            // /health and /v1/models are public in llama.cpp. Probe a
            // protected route with an empty request before sending any cue.
            let protected_endpoint = format!("{}/v1/chat/completions", instance.endpoint);
            let unauthenticated = instance.client.post(protected_endpoint.as_str())
                .json(&serde_json::json!({})).timeout(Duration::from_secs(2)).send()
                .map_err(|error| format!("Could not verify local AI authentication: {error}"))?;
            if unauthenticated.status() != reqwest::StatusCode::UNAUTHORIZED
                && unauthenticated.status() != reqwest::StatusCode::FORBIDDEN {
                return Err("The local AI server did not enforce its per-run API key.".into());
            }
            let authenticated = instance.client.post(protected_endpoint.as_str())
                .bearer_auth(&instance.api_key).json(&serde_json::json!({}))
                .timeout(Duration::from_secs(2)).send()
                .map_err(|error| format!("Could not authenticate the local AI server: {error}"))?;
            if (authenticated.status() == reqwest::StatusCode::UNAUTHORIZED
                || authenticated.status() == reqwest::StatusCode::FORBIDDEN
                || authenticated.status() == reqwest::StatusCode::NOT_FOUND)
                || !ai_server_port_owned_by(instance.port, instance.child.id())? {
                return Err("The local AI server identity check failed.".into());
            }
            if backend == AiComputeBackend::Cuda {
                let log = fs::read_to_string(&instance.log_path).unwrap_or_default();
                match gpu_offloaded_layers(&log) {
                    Some((layers, total)) if layers > 0 => emit_ai_progress(app, "runtime", format!("CUDA: verified {layers}/{total} GPU layers"), layers as u64, Some(total as u64)),
                    _ => return Err(format!("CUDA was selected but positive GPU layer offload was not confirmed. No CPU fallback was used. Log: {}", instance.log_path.display())),
                }
            }
            return Ok(instance);
        }
        thread::sleep(Duration::from_millis(200));
    }
    Err(format!("The local AI server did not become ready. Log: {}", instance.log_path.display()))
}

fn ai_group_response_format(target_ids: &[usize]) -> serde_json::Value {
    serde_json::json!({
        "type": "json_object",
        "schema": {
            "type": "array", "minItems": target_ids.len(), "maxItems": target_ids.len(),
            "items": {
                "type": "object", "properties": {
                    "id": {"type": "integer", "enum": target_ids},
                    "text": {"type": "string"}
                }, "required": ["id", "text"], "additionalProperties": false
            }
        }
    })
}

fn run_ai_chunk(
    app: &AppHandle,
    server: &AiServer,
    cli: &Path,
    model_path: &Path,
    prompt: &str,
    _backend: AiComputeBackend,
    _cuda_tag: &str,
    _gpu_layers: u32,
    single_cue: Option<(usize, &str, bool)>,
    target_ids: &[usize],
) -> Result<Vec<AiCueOutput>, String> {
    ensure_ai_not_cancelled(app)?;
    if !ai_server_port_owned_by(server.port, server.child.id())? {
        return Err("The local AI server is no longer owned by this application.".into());
    }
    let mut payload = serde_json::json!({
        "model": model_path.to_string_lossy(),
        "messages": [{"role":"user", "content":prompt}],
        "max_tokens": 2048, "temperature": 0.2, "top_k": 20, "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": false}, "stream": false
    });
    if single_cue.is_none() {
        // Constrain grouped output during generation, before strict ID/order validation.
        // Single-cue retries keep their existing plain-text format.
        payload["response_format"] = ai_group_response_format(target_ids);
    }
    let response = server.client.post(format!("{}/v1/chat/completions", server.endpoint))
        .bearer_auth(&server.api_key).json(&payload).send().map_err(|error| {
            if app.state::<AiState>().cancel_requested.load(Ordering::SeqCst) { "AI task cancelled.".to_string() }
            else { format!("Local AI request failed: {error}. Log: {}", server.log_path.display()) }
        })?;
    ensure_ai_not_cancelled(app)?;
    let status = response.status();
    let body = response.text().map_err(|error| format!("Could not read local AI response: {error}"))?;
    if !status.is_success() {
        return Err(format!("Local AI engine returned {status}: {}. Log: {}", body.chars().take(500).collect::<String>(), server.log_path.display()));
    }
    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|error| format!("Local AI response was invalid JSON: {error}"))?;
    let content = value["choices"][0]["message"]["content"].as_str()
        .ok_or("Local AI response did not contain subtitle text.")?;
    if let Some((id, source, is_translation)) = single_cue {
        let parsed = if is_translation { parse_translation_output(content, id, source) }
            else { parse_single_cleanup_output(content, id) };
        return parsed.map(|text| vec![AiCueOutput { id, text }]).map_err(|error| {
            let log = fs::read_to_string(&server.log_path).unwrap_or_default();
            let diagnostic = persist_ai_diagnostic(app, cli, model_path, content, "", &log, &error);
            diagnostic.map_or_else(|| error.clone(), |path| format!("{error} Diagnostic log: {}", path.display()))
        });
    }
    parse_ai_process_output(app, cli, model_path, content, "", "")
}

fn gpu_offloaded_layers(diagnostics: &str) -> Option<(usize, usize)> {
    // Read the model-loader INFO line captured from the persistent server.
    diagnostics.lines().find_map(|line| {
        let line = line.to_ascii_lowercase();
        let (_, counts) = line.split_once("offloaded ")?;
        let counts = counts.split_once(" layers to gpu")?.0;
        let (layers, total) = counts.split_once('/')?;
        Some((layers.trim().parse().ok()?, total.trim().parse().ok()?))
    })
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
    _context_before: &[(usize, &SubtitleCue)],
    _context_after: &[(usize, &SubtitleCue)],
    _document_samples: &[(usize, &SubtitleCue)],
    allow_empty_text: bool,
) -> Result<Vec<AiCueOutput>, String> {
    let is_target = |id: usize| targets.iter().any(|(target_id, _)| *target_id == id);
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

fn should_retry_single_ai_output(error: &str, target_count: usize) -> bool {
    target_count == 1 && error.starts_with("The model ")
}

fn corrective_single_ai_prompt(prompt: &str, is_translation: bool) -> String {
    let instruction = if is_translation {
        "Your previous response was not usable. Retry once. Return only the translated target subtitle text, preserving line breaks. Do not return reference or context text, the source text, JSON, Markdown, or an explanation."
    } else {
        "Your previous response was not usable. Retry once. Return only the cleaned target subtitle text, or exactly [[REMOVE]] only if it is clear non-dialogue noise. Do not return reference or context text, JSON, Markdown, or an explanation."
    };
    format!("{prompt}\n\n{instruction}")
}

fn process_local_ai_range(
    app: &AppHandle,
    cli: &Path,
    model_path: &Path,
    server: &AiServer,
    cues: &[SubtitleCue],
    document_samples: &[(usize, &SubtitleCue)],
    start: usize,
    end: usize,
    results: &mut BTreeMap<usize, String>,
    request: &AiRequest,
    backend: AiComputeBackend,
    cuda_tag: &str,
    gpu_layers: u32,
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
        } else if request.mode == "translate" && !needs_translation(cue) {
            // Numbers and punctuation do not need generation. This also avoids
            // a one-cue model echo exhausting the recursive retry path.
            results.insert(id, cue.text.join("\n"));
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
    let single_cue = if model_targets.len() == 1 {
        model_targets.first().map(|(id, cue)| (*id, cue.text.join("\n"), request.mode == "translate"))
    } else { None };
    let target_ids = model_targets.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    let execute_attempt = |attempt_prompt: &str| run_ai_chunk(app, server, cli, model_path, attempt_prompt, backend, cuda_tag, gpu_layers, single_cue.as_ref().map(|(id, text, is_translation)| (*id, text.as_str(), *is_translation)), &target_ids).and_then(|response| {
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
    let attempt = match execute_attempt(&prompt) {
        Err(error) if should_retry_single_ai_output(&error, model_targets.len()) => {
            let retry_prompt = corrective_single_ai_prompt(&prompt, request.mode == "translate");
            execute_attempt(&retry_prompt)
        }
        result => result,
    };
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
                server,
                cues,
                document_samples,
                start,
                midpoint,
                results,
                request,
                backend,
                cuda_tag,
                gpu_layers,
                chunk_index,
                total_chunks,
            )?;
            process_local_ai_range(
                app,
                cli,
                model_path,
                server,
                cues,
                document_samples,
                midpoint,
                end,
                results,
                request,
                backend,
                cuda_tag,
                gpu_layers,
                chunk_index,
                total_chunks,
            )
        }
        Err(error) => Err(error),
    }
}

fn run_local_ai(app: &AppHandle, request: &AiRequest) -> Result<AiResult, String> {
    let model = ai_model_spec(&request.model_id)?;
    let backend = request.compute_backend.unwrap_or(AiComputeBackend::Cuda);
    let measured_gpu_memory = if backend == AiComputeBackend::Cuda {
        Some(gpu_memory().ok_or("Could not measure total and free GPU memory. Choose CPU.")?)
    } else {
        None
    };
    if backend == AiComputeBackend::Cuda {
        let (eligible, reason) = gpu_eligibility(model, measured_gpu_memory);
        if !eligible {
            return Err(reason);
        }
    }
    let root = ai_storage_root(app)?;
    let model_path = ai_model_path(&root, model.id)
        .ok_or_else(|| format!("Download {} before running local AI.", model.name))?;
    let mut warm = take_warm_ai(app, model, backend)?;
    if warm.is_some() && !model_verification_cache_matches(&model_path, model.sha256) {
        // The loaded server belongs to the previous bytes. Drop it, then take
        // the normal cold path so the current file is hashed before loading.
        warm = None;
    }
    let measured_gpu_memory = if backend == AiComputeBackend::Cuda {
        Some(gpu_memory().ok_or("Could not remeasure total and free GPU memory before inference. Choose CPU.")?)
    } else {
        None
    };
    if backend == AiComputeBackend::Cuda {
        let (eligible, reason) = gpu_eligibility(model, measured_gpu_memory);
        if !eligible {
            return Err(reason);
        }
    }
    if warm.is_none() {
        if let Some(memory) = measured_gpu_memory {
            if memory.free_mib < model.minimum_free_gpu_mib {
                return Err(format!("There is not enough free GPU memory to start {}. Close GPU apps or choose CPU.", model.name));
            }
        }
    }
    if warm.is_none() {
        // Cold loads reuse a durable content verification only while the local
        // file fingerprint is unchanged. A warm server must match it too.
        verify_model_integrity_for_task(app, &model_path, model)?;
    }
    let (cli, cuda_tag) = require_llama_runtime(app, &root, backend)?;
    let gpu_layers = if backend == AiComputeBackend::Cuda {
        if warm.is_none() {
            let memory = gpu_memory().ok_or("Could not remeasure free GPU memory before model loading. Choose CPU.")?;
            if memory.free_mib < model.minimum_free_gpu_mib {
                return Err(format!("There is not enough free GPU memory to start {}. Close GPU apps or choose CPU.", model.name));
            }
        }
        cuda_layers_for_model(model, None)
    } else {
        0
    };
    let server = match warm {
        Some(server) => server,
        None => {
            emit_ai_progress(app, "runtime", "Loading the verified model into the local AI server", 0, None);
            start_ai_server(app, &cli, &model_path, backend, gpu_layers)?
        }
    };
    let result: Result<AiResult, String> = (|| {
    let cues = parse_srt_cues(&request.content)?;
    let document_samples = document_sample_cues(&cues);
    let mut results: BTreeMap<usize, String> = BTreeMap::new();
    let chunk_size = if request.mode == "translate" {
        AI_TRANSLATION_CHUNK_SIZE
    } else {
        AI_TARGET_CHUNK_SIZE
    };
    let total_chunks = cues.len().div_ceil(chunk_size);
    for (chunk_index, start) in (0..cues.len()).step_by(chunk_size).enumerate() {
        ensure_ai_not_cancelled(app)?;
        let end = (start + chunk_size).min(cues.len());
        let action = if request.mode == "translate" {
            "Translating"
        } else {
            "Cleaning"
        };
        emit_ai_progress(
            app,
            "processing",
            format!(
                "{action} contextual group {} of {} ({})",
                chunk_index + 1,
                total_chunks,
                if backend == AiComputeBackend::Cuda { format!("NVIDIA CUDA {cuda_tag}") } else { "explicit CPU mode".to_string() }
            ),
            chunk_index as u64,
            Some(total_chunks as u64),
        );
        process_local_ai_range(
            app,
            &cli,
            &model_path,
            &server,
            &cues,
            &document_samples,
            start,
            end,
            &mut results,
            request,
            backend,
            &cuda_tag,
            gpu_layers,
            chunk_index,
            total_chunks,
        )?;
    }
    let (output, cue_count) = render_ai_srt(&cues, &results)?;
    ensure_ai_not_cancelled(app)?;
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
    })();
    let transport_failed = result.as_ref().err().is_some_and(|error| {
        error.starts_with("Local AI request failed:")
            || error.starts_with("Local AI engine returned")
            || error.starts_with("Local AI response")
            || error.starts_with("Could not read local AI response:")
    });
    if !transport_failed { preserve_warm_ai(app, model, backend, server)?; }
    result
}

fn run_ai_blocking(app: AppHandle, request: AiRequest) -> Result<AiResult, String> {
    begin_ai_task(&app)?;
    let result = (|| {
        validate_choice(&request.mode, &["clean", "translate"], "AI task")?;
        run_local_ai(&app, &request)
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
        &["Auto", "CUDA", "CPU", "auto", "cuda", "cpu"],
        "pipeline compute mode",
    )?;
    validate_choice(
        &request.ocr_compute,
        &["Auto", "CUDA", "CPU", "auto", "cuda", "cpu", "mixed"],
        "OCR compute mode",
    )?;
    let video = validate_video(&request.video_path)?;
    let project = resolve_project_root()?;
    let results_root = resolve_results_root(&project)?;
    let reports_root = resolve_reports_root(&results_root)?;
    let script = project.join("run-pipeline.ps1");
    let session_token = new_ai_server_key()
        .map_err(|error| format!("Could not create pipeline workspace ownership: {error}"))?
        .chars().take(32).collect::<String>();
    let state = app.state::<PipelineState>();
    {
        let mut active = lock(&state.active_pid)?;
        if state.closing.load(Ordering::SeqCst) {
            return Err("The application is closing.".into());
        }
        if active.is_some() {
            return Err("Another subtitle extraction is already running.".to_string());
        }
        if lock(&app.state::<AiState>().active_pid)?.is_some() {
            return Err("Local AI is running. Finish or cancel it before video OCR.".into());
        }
        state.cancel_requested.store(false, Ordering::SeqCst);
        *active = Some(0);
    }
    if let Err(error) = release_warm_ai(&app) {
        clear_active_pid(&app, 0);
        return Err(error);
    }
    // A prior cleanup failure must not lose its recovery token on the next run.
    if let Err(error) = cleanup_pipeline_workspace(&app, None) {
        clear_active_pid(&app, 0);
        return Err(error);
    }
    *lock(&state.active_session)? = Some(PipelineSession {
        token: session_token.clone(), project: project.clone(),
    });

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
        .env("SUBHOOPER_SESSION_TOKEN", &session_token)
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

    // Hold the PID lock through spawning: close cannot observe the placeholder
    // and exit while an untracked pipeline is being created.
    let mut active = lock(&state.active_pid)?;
    if state.closing.load(Ordering::SeqCst) || state.cancel_requested.load(Ordering::SeqCst) {
        *active = None;
        *lock(&state.active_session)? = None;
        return Err("Processing was cancelled.".into());
    }
    let mut child = command.spawn().map_err(|error| {
        *active = None;
        if let Ok(mut session) = state.active_session.lock() {
            *session = None;
        }
        format!("Could not start the pipeline: {error}")
    })?;
    let pid = child.id();
    *active = Some(pid);
    drop(active);
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
    let mut status = child
        .wait()
        .map_err(|error| format!("Could not wait for the pipeline: {error}"));
    let mut terminated = status.is_ok();
    if !terminated && kill_process_tree(pid).is_ok() {
        terminated = true;
        status = child.wait().map_err(|error| format!("Could not reap the terminated pipeline: {error}"));
    }
    if !terminated {
        return Err(status.err().unwrap_or_else(|| "Could not confirm pipeline termination.".into()));
    }
    if let Some(handle) = stdout_thread {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_thread {
        let _ = handle.join();
    }
    if terminated {
        if let Err(error) = cleanup_pipeline_workspace(&app, Some(&session_token)) {
            let _ = app.emit("pipeline-output", format!("WARNING: {error}"));
        }
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
fn ai_catalog(
    app: AppHandle,
    compute_backend: Option<AiComputeBackend>,
) -> Result<Vec<AiModelInfo>, String> {
    startup_trace("ai-catalog-enter");
    let root = ai_storage_root(&app)?;
    let gpu_mib = gpu_memory();
    let models = AI_MODELS
        .iter()
        .copied()
        .map(|model| ai_model_info(&root, model, compute_backend, gpu_mib))
        .collect();
    startup_trace("ai-catalog-exit");
    Ok(models)
}

fn import_ai_model_blocking(app: AppHandle, model_id: String, selected: String) -> Result<AiModelInfo, String> {
    begin_ai_task(&app)?;
    let result = (|| {
        release_warm_ai(&app)?;
        let model = ai_model_spec(&model_id)?;
        let source = PathBuf::from(selected);
        let metadata = fs::symlink_metadata(&source).map_err(|error| format!("Could not inspect the selected GGUF: {error}"))?;
        if runtime_reparse_point(&metadata) || !metadata.is_file()
            || !source.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.eq_ignore_ascii_case(model.filename)) {
            return Err(format!("Select the exact {} file.", model.filename));
        }
        let root = ai_storage_root(&app)?;
        let destination = root.join("models").join(model.id).join(model.filename);
        if source == destination {
            verify_model_integrity_for_task(&app, &source, model)?;
        } else {
            let parent = destination.parent().ok_or("The AI model path is invalid.")?;
            fs::create_dir_all(parent).map_err(|error| format!("Could not create the AI model folder: {error}"))?;
            let partial = destination.with_extension("importing");
            let copied = (|| {
                let mut input = fs::File::open(&source).map_err(|error| format!("Could not open the selected GGUF: {error}"))?;
                let mut output = fs::File::create(&partial).map_err(|error| format!("Could not prepare the model import: {error}"))?;
                let mut hasher = Sha256::new();
                let mut buffer = [0_u8; 1024 * 1024];
                let mut received = 0_u64;
                loop {
                    ensure_ai_not_cancelled(&app)?;
                    let count = input.read(&mut buffer).map_err(|error| format!("Could not read the selected GGUF: {error}"))?;
                    if count == 0 { break; }
                    output.write_all(&buffer[..count]).map_err(|error| format!("Could not copy the selected GGUF: {error}"))?;
                    hasher.update(&buffer[..count]);
                    received += count as u64;
                    if received % (16 * 1024 * 1024) < count as u64 {
                        emit_ai_progress(&app, "model", format!("Verifying and importing {}", model.name), received, Some(metadata.len()));
                    }
                }
                output.sync_all().map_err(|error| format!("Could not finish the GGUF import: {error}"))?;
                ensure_ai_not_cancelled(&app)?;
                if !format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(model.sha256) {
                    return Err("The selected GGUF failed its pinned SHA-256 check.".into());
                }
                Ok(())
            })();
            if let Err(error) = copied { let _ = fs::remove_file(&partial); return Err(error); }
            if destination.exists() { fs::remove_file(&destination).map_err(|error| format!("Could not replace the old GGUF: {error}"))?; }
            fs::rename(&partial, &destination).map_err(|error| format!("Could not install the verified GGUF: {error}"))?;
            write_model_digest_sidecar(&destination, model.sha256)?;
            write_model_verification_cache(&destination, model.sha256)?;
        }
        Ok(ai_model_info(&root, model, None, gpu_memory()))
    })();
    end_ai_task(&app);
    result
}

#[tauri::command]
async fn import_ai_model(app: AppHandle, model_id: String, path: String) -> Result<AiModelInfo, String> {
    tauri::async_runtime::spawn_blocking(move || import_ai_model_blocking(app, model_id, path))
        .await.map_err(|error| format!("The model import stopped unexpectedly: {error}"))?
}

#[tauri::command]
fn release_ai_runtime(app: AppHandle) -> Result<(), String> {
    release_warm_ai(&app)
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
async fn download_ai_model(
    app: AppHandle,
    model_id: String,
    compute_backend: Option<AiComputeBackend>,
) -> Result<AiModelInfo, String> {
    let backend = compute_backend.unwrap_or(AiComputeBackend::Cuda);
    tauri::async_runtime::spawn_blocking(move || install_ai_model_blocking(app, model_id, backend))
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
    startup_trace("builder-before");
    let builder = tauri::Builder::default();
    startup_trace("builder-created");
    let builder = builder.plugin(tauri_plugin_dialog::init());
    startup_trace("dialog-plugin-added");
    let builder = builder.plugin(tauri_plugin_process::init());
    startup_trace("process-plugin-added");
    let builder = builder.plugin(tauri_plugin_updater::Builder::new().build());
    startup_trace("updater-plugin-added");
    let builder = builder.manage(PipelineState::default()).manage(AiState::default());
    startup_trace("state-added");
    let builder = builder.invoke_handler(tauri::generate_handler![
            start_pipeline,
            cancel_pipeline,
            save_export,
            component_status,
            install_components,
            ai_catalog,
            ocr_hardware_recommendation,
            import_ai_model,
            release_ai_runtime,
            read_subtitle_file,
            download_ai_model,
            run_ai_cleaning,
            cancel_ai,
        ]);
    startup_trace("commands-added");
    let builder = builder.setup(|_| {
        startup_trace("setup-enter");
        startup_trace("setup-exit");
        Ok(())
    });
    let builder = builder.on_page_load(|_, _| startup_trace("page-load"));
    let builder = builder.on_window_event(|window, event| {
            static WINDOW_EVENTS_TRACED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if WINDOW_EVENTS_TRACED.fetch_add(1, Ordering::Relaxed) < 32 {
                startup_trace("window-event");
            }
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.app_handle().state::<PipelineState>();
                state.closing.store(true, Ordering::SeqCst);
                state.cancel_requested.store(true, Ordering::SeqCst);
                let ai_state = window.app_handle().state::<AiState>();
                ai_state.closing.store(true, Ordering::SeqCst);
                ai_state.cancel_requested.store(true, Ordering::SeqCst);
                let mut pipeline_terminated = false;
                if let Ok(active) = state.active_pid.lock() {
                    pipeline_terminated = true;
                    if let Some(pid) = *active {
                        if pid > 0 {
                            pipeline_terminated = kill_process_tree(pid).is_ok();
                        }
                    }
                };
                if pipeline_terminated {
                    if let Err(error) = cleanup_pipeline_workspace(window.app_handle(), None) {
                        eprintln!("WARNING: {error}");
                    }
                }
                if let Ok(active) = ai_state.active_pid.lock() {
                    if let Some(pid) = *active {
                        if pid > 0 {
                            let _ = kill_process_tree(pid);
                        }
                    }
                };
                let _ = release_warm_ai(window.app_handle());
            }
        });
    startup_trace("context-before");
    let context = tauri::generate_context!();
    startup_trace("context-created");
    builder.run(context)
        .expect("Could not start the SubHooper GUI");
}

pub fn startup_trace(phase: &str) {
    let message = format!(
        "SubHooper 0.4.3 startup pid={} thread={:?} time_ms={} phase={phase}\n",
        std::process::id(),
        std::thread::current().id(),
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_millis())
    );
    let path = std::env::var_os("SUBHOOPER_STARTUP_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::var_os("LOCALAPPDATA")
            .map(|root| PathBuf::from(root).join("SubHooper").join("reports"))
            .unwrap_or_else(std::env::temp_dir).join("startup-0.4.3.log"));
    if let Some(parent) = path.parent() { let _ = fs::create_dir_all(parent); }
    if let Ok(mut log) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = log.write_all(message.as_bytes());
    }
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
    fn stage_parser_maps_native_pipeline_markers() {
        let stage = stage_for_line("1/2 Native engine: test").unwrap();
        assert_eq!(stage.key, "ocr");
        assert_eq!(stage.label, "Scanning subtitles with the native OCR engine");
        let progress = stage_for_line("OCRProgress=67").unwrap();
        assert_eq!(progress.key, "ocr");
        assert_eq!(progress.percent, Some(67));
        assert!(stage_for_line("OCRProgress=100").is_none());
        assert!(stage_for_line("ordinary output").is_none());
    }

    #[test]
    fn gpu_offload_parser_reads_layer_telemetry() {
        assert_eq!(gpu_offloaded_layers("load: loaded CUDA backend\noffloaded 61/65 layers to GPU"), Some((61, 65)));
        assert_eq!(gpu_offloaded_layers("0.01.123.456 I load_tensors: offloaded 40/41 layers to GPU"), Some((40, 41)));
        assert_eq!(gpu_offloaded_layers("offloaded 0/65 layers to GPU"), Some((0, 65)));
        assert_eq!(gpu_offloaded_layers("no GPU layer log"), None);
    }

    #[test]
    fn cuda_layer_request_and_capacity_gate() {
        assert_eq!(cuda_layers_for_model(AI_MODELS[0], Some(6 * 1024)), 99);
        let memory = |total_mib, free_mib| Some(GpuMemory { total_mib, free_mib });
        assert!(!gpu_eligibility(AI_MODELS[0], memory(2048, 1800)).0);
        assert!(gpu_eligibility(AI_MODELS[0], memory(4096, 3900)).0);
        assert!(!gpu_eligibility(AI_MODELS[1], memory(6144, 6000)).0);
        assert!(gpu_eligibility(AI_MODELS[1], memory(8192, 7600)).0);
        assert!(!gpu_eligibility(AI_MODELS[2], memory(10240, 9800)).0);
        assert!(gpu_eligibility(AI_MODELS[2], memory(12282, 11800)).0);
        assert!(!gpu_eligibility(AI_MODELS[2], None).0);
        assert!(gpu_eligibility(AI_MODELS[2], memory(12000, 9500)).1.contains("needs more GPU memory"));
    }

    #[test]
    fn gpu_memory_parser_reads_total_and_free_together() {
        assert_eq!(
            parse_gpu_memory_report("12282, 11840\n"),
            Some(GpuMemory { total_mib: 12282, free_mib: 11840 }),
        );
        assert_eq!(parse_gpu_memory_report("N/A, 11840\n"), None);
        assert_eq!(parse_gpu_memory_report("1000, 1024\n"), None);
    }

    #[test]
    fn ocr_hardware_recommendation_matches_two_visible_modes() {
        assert_eq!(recommended_ocr_mode(false, false), "cpu");
        assert_eq!(recommended_ocr_mode(true, false), "mixed");
        assert_eq!(recommended_ocr_mode(false, true), "mixed");
        assert_eq!(recommended_ocr_mode(true, true), "mixed");
    }

    #[test]
    fn grouped_ai_schema_bounds_cue_count_and_restricts_target_ids() {
        let format = ai_group_response_format(&[181, 182, 184]);
        let schema = &format["schema"];
        assert_eq!(format["type"], "json_object");
        assert_eq!(schema["minItems"], 3);
        assert_eq!(schema["maxItems"], 3);
        assert_eq!(schema["items"]["properties"]["id"]["enum"], serde_json::json!([181, 182, 184]));
        assert_eq!(schema["items"]["required"], serde_json::json!(["id", "text"]));
        assert_eq!(schema["items"]["additionalProperties"], false);
    }

    #[test]
    fn edited_srt_is_lf_normalized_and_terminated() {
        let value =
            prepare_srt_for_save("1\r\n00:00:01,000 --> 00:00:02,000\r\nText").expect("valid SRT");
        assert_eq!(value, "1\n00:00:01,000 --> 00:00:02,000\nText\n");
    }

    #[test]
    fn srt_export_keeps_original_line_endings_without_ai() {
        let source = "1\r\n00:00:01,000 --> 00:00:02,000\r\nOCR text\r\n";
        assert_eq!(export_content(source, "srt").unwrap(), source);
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
    fn ai_output_parser_repairs_only_a_missing_array_terminator() {
        let parsed =
            clean_model_output("[{\"id\":1,\"text\":\"First\"},{\"id\":2,\"text\":\"Second\"}")
                .expect("safely repairable model output");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].id, 2);
        assert!(clean_model_output("[{\"id\":1,\"text\":\"partial\"").is_err());
    }

    #[test]
    fn translation_retry_accepts_single_object_but_rejects_partial_group() {
        let transcript = "User:\nInput: [{\"id\":1,\"text\":\"HELLO THERE\"}]\n\nAssistant:\n{\"id\":1,\"text\":\"MERHABA İSTANBUL\"}";
        let parsed = clean_model_output(assistant_output(transcript)).expect("single cue response");
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["HELLO THERE".into()],
        };
        assert!(select_ai_target_response(&parsed, &[(1, &cue)], &[], &[], &[], false).is_ok());
        assert!(select_ai_target_response(&parsed, &[(1, &cue), (2, &cue)], &[], &[], &[], false).is_err());
    }

    #[test]
    fn translation_accepts_plain_text_and_logged_raw_newline_json() {
        let plain = "User:\nSubtitle:\nHello\n\nAssistant:\nMerhaba";
        assert_eq!(parse_translation_output(plain, 194, "Hello").unwrap(), "Merhaba");
        let logged = "User:\nSubtitle:\nHello\n\nAssistant:\n{\"id\":194,\"text\":\"Birinci satır\nİkinci satır\"}";
        assert_eq!(parse_translation_output(logged, 194, "Hello").unwrap(), "Birinci satır\nİkinci satır");
        assert!(parse_translation_output(logged, 195, "Hello").is_err());
        assert!(parse_translation_output("Assistant:\n", 194, "Hello").is_err());
        let group = "[{\"id\":1,\"text\":\"First\nline\"},{\"id\":2,\"text\":\"Second\"}]";
        let parsed = clean_model_output(group).expect("raw newline in a grouped JSON string");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].text, "First\nline");
    }

    #[test]
    fn qwen_string_fragment_repair_keeps_exact_cue_ids() {
        let logged = r#"[{"id":159,"text":"Örnek ilk satır."+"\nÖrnek ikinci satır."},{"id":160,"text":"Örnek sonraki altyazı."}]"#;
        let parsed = clean_model_output(logged).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].text, "Örnek ilk satır.\nÖrnek ikinci satır.");
        assert_eq!(parsed[0].id, 159);
        assert_eq!(parsed[1].id, 160);
        assert!(join_json_string_fragments(r#"[{"id":1,"text":"normal"}]"#).is_none());
    }

    #[test]
    fn qwen_merged_cue_objects_repair_only_complete_boundaries() {
        let logged = r#"[{"id":23,"text":"","id":24,"text":"The sample is ready.","id":25,"text":"Done"}]"#;
        let parsed = clean_model_output(logged).expect("logged Qwen response");
        assert_eq!(parsed.iter().map(|cue| cue.id).collect::<Vec<_>>(), [23, 24, 25]);
        assert_eq!(parsed[1].text, "The sample is ready.");
        let quoted = r#"[{"id":23,"text":"The literal ,\"id\": marker"}]"#;
        assert_eq!(clean_model_output(quoted).unwrap()[0].text, "The literal ,\"id\": marker");
        assert!(clean_model_output(r#"[{"id":23,"text":"","id":24}]"#).is_err());
    }

    #[test]
    fn single_cleanup_accepts_text_and_explicit_removal_but_rejects_empty_array() {
        assert_eq!(parse_single_cleanup_output("Merhaba, Gulnara!", 12).unwrap(), "Merhaba, Gulnara!");
        assert_eq!(parse_single_cleanup_output("[[REMOVE]]", 12).unwrap(), "");
        assert_eq!(parse_single_cleanup_output("{\"id\":12,\"text\":\"Merhaba\"}", 12).unwrap(), "Merhaba");
        assert!(parse_single_cleanup_output("[]", 12).is_err());
        assert!(parse_single_cleanup_output("{\"id\":11,\"text\":\"Merhaba\"}", 12).is_err());
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
    fn single_cue_retry_is_limited_to_model_output_errors() {
        assert!(should_retry_single_ai_output(
            "The model returned invalid subtitle data. Diagnostic log: test",
            1,
        ));
        assert!(!should_retry_single_ai_output(
            "Local AI request failed: connection refused.",
            1,
        ));
        assert!(!should_retry_single_ai_output(
            "AI task cancelled.",
            1,
        ));
        assert!(!should_retry_single_ai_output(
            "The model returned invalid subtitle data.",
            2,
        ));
        let translate = corrective_single_ai_prompt("Context: [\"read only\"]", true);
        assert!(translate.contains("only the translated target subtitle text"));
        assert!(translate.contains("Context: [\"read only\"]"));
        let clean = corrective_single_ai_prompt("Target subtitle: text", false);
        assert!(clean.contains("cleaned target subtitle text, or exactly [[REMOVE]]"));
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
    fn local_ai_requires_every_target_once_in_order_and_ignores_extras() {
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
        let with_extra = select_ai_target_response(
            &unknown, &targets, &[], &context_after, &[], false,
        ).expect("valid target responses may contain unsolicited objects");
        assert_eq!(with_extra.iter().map(|cue| cue.id).collect::<Vec<_>>(), [13, 14]);
        let wrong_only = (205..=212).map(|id| AiCueOutput {
            id,
            text: format!("Unrequested cue {id}"),
        }).collect::<Vec<_>>();
        assert!(select_ai_target_response(
            &wrong_only, &targets, &[], &context_after, &[], false,
        ).is_err(), "unrequested IDs must never be remapped to targets");
        assert!(
            select_ai_target_response(&empty, &targets, &[], &context_after, &[], false).is_err()
        );
        let cleaned = select_ai_target_response(&empty, &targets, &[], &context_after, &[], true)
            .expect("cleanup may explicitly drop noise");
        assert!(cleaned[1].text.is_empty());
    }

    #[test]
    fn translation_prompt_requires_a_target_language() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["Hello".into()],
        };
        let before = SubtitleCue { text: vec!["Earlier context".into()], ..cue.clone() };
        let after = SubtitleCue { text: vec!["Later context".into()], ..cue.clone() };
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
            &[(8, &before)],
            &[(10, &after)],
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
        assert!(prompt.contains("Translate this subtitle into Turkish"));
        assert!(prompt.contains("Source: English"));
        assert!(prompt.contains("idiomatic Turkish"));
        assert!(prompt.contains("Earlier context"));
        assert!(prompt.contains("Later context"));
        assert!(prompt.contains("translate only the target subtitle and never return context cues"));
        assert!(!prompt.contains("Previous translation"));
        assert!(prompt.contains("Keep the character name Ada."));
        assert!(!prompt.contains("target_ids"));
        assert!(!prompt.contains("document_samples"));
        assert!(prompt.contains("Target subtitle:\nHello"));
        assert!(prompt.contains("Earlier context"));
        assert!(!prompt.contains("\"id\":8"));
        assert!(!prompt.contains("/no_think"));
        let grouped = ai_prompt(
            &[(1, &cue), (2, &cue)],
            &[], &[], &[], &[],
            "translate", Some("English"), Some("Turkish"), None, None, None,
        ).unwrap();
        assert!(grouped.contains("Translate each input subtitle into Turkish"));
        let grouped_context = ai_prompt(
            &[(1, &cue)], &[(8, &before)], &[(10, &after)], &[], &[],
            "translate", None, Some("Turkish"), None, None, None,
        ).unwrap();
        assert!(grouped_context.contains("Earlier context"));
        assert!(grouped_context.contains("Later context"));
        assert!(!grouped_context.contains("\"id\":8"));
        assert!(!grouped_context.contains("\"id\":10"));
        let grouped_with_context = ai_prompt(
            &[(1, &cue), (2, &cue)], &[(8, &before)], &[(10, &after)], &[], &[],
            "translate", None, Some("Turkish"), None, None, None,
        ).unwrap();
        assert!(grouped_with_context.contains("\"id\":1"));
        assert!(grouped_with_context.contains("\"id\":2"));
        assert!(!grouped_with_context.contains("\"id\":8"));
        assert!(!grouped_with_context.contains("\"id\":10"));
        assert!(grouped.contains("\"id\":1,"));
        assert!(grouped.contains("\"id\":2,"));
    }

    #[test]
    fn numeric_subtitle_does_not_need_model_translation() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["169".into()],
        };
        assert!(!needs_translation(&cue));
        let dialogue = SubtitleCue { text: vec!["Hello, 169".into()], ..cue };
        assert!(needs_translation(&dialogue));
    }

    #[test]
    fn cleanup_prompt_normalizes_language_and_marks_only_noise_for_removal() {
        let cue = SubtitleCue {
            start: "00:00:01,000".into(),
            end: "00:00:02,000".into(),
            text: vec!["Meaningful dialogue".into()],
        };
        let before = SubtitleCue { text: vec!["Earlier context".into()], ..cue.clone() };
        let after = SubtitleCue { text: vec!["Later context".into()], ..cue.clone() };
        let prompt = ai_prompt(
            &[(1, &cue)],
            &[(8, &before)],
            &[(10, &after)],
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
        assert!(prompt.contains("[[REMOVE]]"));
        assert!(prompt.contains("Language samples: [\"Meaningful dialogue\"]"));
        assert!(prompt.contains("read-only context"));
        assert!(prompt.contains("Earlier context"));
        assert!(prompt.contains("Later context"));
        assert!(prompt.contains("Target subtitle:\nMeaningful dialogue"));
        assert!(!prompt.contains("\"id\":1"));
        assert!(!prompt.contains("\"id\":8"));
        assert!(!prompt.contains("\"id\":10"));
        assert!(!prompt.contains("document_samples"));
        let grouped = ai_prompt(
            &[(1, &cue), (2, &cue)], &[(8, &before)], &[(10, &after)], &[(40, &cue)], &[],
            "clean", None, None, Some("Turkish"), Some("balanced"), None,
        ).unwrap();
        assert!(grouped.contains("Input: [{\"id\":1,"));
        assert!(grouped.contains("Language samples: []"));
        assert!(grouped.contains("Earlier context"));
        assert!(grouped.contains("Later context"));
        assert!(grouped.contains("\"id\":2"));
        assert!(!grouped.contains("\"id\":8"));
        assert!(!grouped.contains("\"id\":10"));
        assert!(!grouped.contains("target_ids"));
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
    fn local_catalog_uses_pinned_official_qwen_ggufs() {
        let expected = [
            ("qwen3-4b-q4km", "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"),
            ("qwen3-8b-q4km", "Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"),
            ("qwen3-14b-q4km", "Qwen/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf", "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"),
        ];
        assert_eq!(AI_MODELS.len(), expected.len());
        for (model, (id, repository, filename, digest)) in AI_MODELS.iter().zip(expected) {
            assert_eq!((model.id, model.repository, model.filename, model.sha256), (id, repository, filename, digest));
            assert!(valid_sha256(model.sha256));
        }
        assert!(ai_model_spec("bonsai-8b").is_err());
    }

    #[test]
    fn local_model_path_only_accepts_the_selected_official_filename() {
        let root = std::env::temp_dir().join(format!("subhooper-model-path-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let model_dir = root.join("models").join("qwen3-4b-q4km");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("old-bonsai-model.gguf"), b"not the selected model").unwrap();
        assert!(ai_model_path(&root, "qwen3-4b-q4km").is_none());
        fs::write(model_dir.join("Qwen3-4B-Q4_K_M.gguf"), b"selected model").unwrap();
        assert_eq!(ai_model_path(&root, "qwen3-4b-q4km").as_deref(), Some(model_dir.join("Qwen3-4B-Q4_K_M.gguf").as_path()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_reports_canonical_file_without_adopting_or_hashing_it() {
        let root = std::env::temp_dir().join(format!("subhooper-catalog-metadata-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let model = AI_MODELS[0];
        let path = root.join("models").join(model.id).join(model.filename);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"unverified local bytes").unwrap();
        let info = ai_model_info(&root, model, Some(AiComputeBackend::Cpu), None);
        assert!(info.model_cached);
        assert!(!info.installed);
        assert!(!model_digest_sidecar(&path).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ai_launch_refuses_closing_or_cancelled_state() {
        let state = AiState::default();
        state.closing.store(true, Ordering::SeqCst);
        assert!(register_ai_launch::<(), _>(&state, || panic!("must not launch after close")).is_err());
        assert_eq!(*state.active_pid.lock().unwrap(), None);
        state.closing.store(false, Ordering::SeqCst);
        state.cancel_requested.store(true, Ordering::SeqCst);
        assert!(register_ai_launch::<(), _>(&state, || panic!("must not launch after cancel")).is_err());
    }

    #[test]
    fn close_cannot_observe_placeholder_during_ai_spawn() {
        let state = Arc::new(AiState::default());
        *state.active_pid.lock().unwrap() = Some(0);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_state = Arc::clone(&state);
        let worker = thread::spawn(move || register_ai_launch(&worker_state, || {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(((), 123))
        }));
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        state.closing.store(true, Ordering::SeqCst);
        state.cancel_requested.store(true, Ordering::SeqCst);
        assert!(state.active_pid.try_lock().is_err());
        release_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();
        assert_eq!(*state.active_pid.lock().unwrap(), Some(123));
    }

    #[test]
    fn download_limit_and_cancel_stop_before_unverified_bytes_are_installed() {
        let mut output = Vec::new();
        let digest = copy_download_limited(&mut std::io::Cursor::new(b"verified"), &mut output, 8, |_| Ok(())).unwrap();
        assert_eq!(output, b"verified");
        assert_eq!(digest, format!("{:x}", Sha256::digest(b"verified")));
        let mut oversized = Vec::new();
        assert!(copy_download_limited(&mut std::io::Cursor::new(b"too much"), &mut oversized, 4, |_| Ok(())).is_err());
        assert!(oversized.is_empty());
        let mut cancelled = Vec::new();
        assert!(copy_download_limited(&mut std::io::Cursor::new(b"verified"), &mut cancelled, 8,
            |_| Err("AI processing was cancelled.".into())).is_err());
        assert!(cancelled.is_empty());
    }

    #[test]
    fn cached_model_adoption_propagates_cancellation_instead_of_requesting_download() {
        let path = PathBuf::from("cached.gguf");
        let cancelled = adopt_cached_model_with(
            Some(path.clone()),
            |_| Err("AI processing was cancelled.".to_string()),
            || true,
        );
        assert_eq!(cancelled, Err("AI processing was cancelled.".to_string()));

        let invalid = adopt_cached_model_with(Some(path), |_| Err("integrity mismatch".to_string()), || false);
        assert_eq!(invalid, Ok(false));
    }

    #[test]
    fn model_digest_sidecar_is_rechecked_and_invalidated_on_tampering() {
        let root = std::env::temp_dir().join(format!(
            "subhooper-model-digest-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let model_path = root.join("model.gguf");
        fs::write(&model_path, b"model weights").unwrap();
        let model = AiModelSpec {
            sha256: "a2d42c4aa884e21216cbb8da4c7ba2fcf9b6033b2331666e666145c24caf7a38",
            ..AI_MODELS[0]
        };
        write_model_digest_sidecar(&model_path, model.sha256).unwrap();
        assert!(model_sidecar_matches(&model_path, model.sha256));
        verify_model_integrity(&model_path, model).unwrap();
        assert!(model_verification_cache_path(&model_path).exists());
        fs::write(&model_path, b"changed weights").unwrap();
        assert!(verify_model_integrity(&model_path, model).is_err());
        assert!(!model_sidecar_matches(&model_path, model.sha256));
        assert!(!model_verification_cache_path(&model_path).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_verification_cache_survives_invocations_and_skips_unchanged_hash() {
        let root = std::env::temp_dir().join(format!("subhooper-model-cache-fast-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("model.gguf");
        fs::write(&path, b"model weights").unwrap();
        let model = AiModelSpec { sha256: "a2d42c4aa884e21216cbb8da4c7ba2fcf9b6033b2331666e666145c24caf7a38", ..AI_MODELS[0] };
        let mut hashes = 0;
        verify_model_integrity_with(&path, model, |_| { hashes += 1; Ok(model.sha256.into()) }).unwrap();
        assert_eq!(hashes, 1);
        verify_model_integrity_with(&path, model, |_| panic!("unchanged model must use its persistent cache")).unwrap();
        assert_eq!(hashes, 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changed_same_size_model_and_corrupt_or_missing_cache_force_rehash() {
        let root = std::env::temp_dir().join(format!("subhooper-model-cache-change-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("model.gguf");
        fs::write(&path, b"model weights").unwrap();
        let model = AiModelSpec { sha256: "a2d42c4aa884e21216cbb8da4c7ba2fcf9b6033b2331666e666145c24caf7a38", ..AI_MODELS[0] };
        verify_model_integrity_with(&path, model, |_| Ok(model.sha256.into())).unwrap();
        let previous_modified = fs::metadata(&path).unwrap().modified().unwrap();
        fs::write(&path, b"other weights").unwrap();
        // Force a distinct timestamp even on filesystems with coarse clocks.
        fs::OpenOptions::new().write(true).open(&path).unwrap()
            .set_times(fs::FileTimes::new().set_modified(previous_modified + Duration::from_secs(2))).unwrap();
        let mut hashes = 0;
        assert!(verify_model_integrity_with(&path, model, |path| { hashes += 1; sha256_file(path) }).is_err());
        assert_eq!(hashes, 1);
        fs::write(&path, b"model weights").unwrap();
        verify_model_integrity_with(&path, model, |path| { hashes += 1; sha256_file(path) }).unwrap();
        assert_eq!(hashes, 2);
        fs::write(model_verification_cache_path(&path), b"bad json").unwrap();
        verify_model_integrity_with(&path, model, |path| { hashes += 1; sha256_file(path) }).unwrap();
        assert_eq!(hashes, 3);
        fs::remove_file(model_verification_cache_path(&path)).unwrap();
        verify_model_integrity_with(&path, model, |path| { hashes += 1; sha256_file(path) }).unwrap();
        assert_eq!(hashes, 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn digest_mismatch_removes_both_model_verification_records() {
        let root = std::env::temp_dir().join(format!("subhooper-model-cache-mismatch-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("model.gguf");
        fs::write(&path, b"model weights").unwrap();
        let model = AiModelSpec { sha256: "a2d42c4aa884e21216cbb8da4c7ba2fcf9b6033b2331666e666145c24caf7a38", ..AI_MODELS[0] };
        verify_model_integrity_with(&path, model, |_| Ok(model.sha256.into())).unwrap();
        fs::write(&path, b"bad content!").unwrap();
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(verify_model_integrity_with(&path, model, |_| Ok(wrong_hash.into())).is_err());
        assert!(!model_digest_sidecar(&path).exists());
        assert!(!model_verification_cache_path(&path).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn runtime_manifest_detects_modified_executable_bytes() {
        let root = std::env::temp_dir().join(format!(
            "subhooper-ai-runtime-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("llama-server.exe"), b"verified runtime executable").unwrap();
        assert!(!is_llama_runtime(&root));
        write_runtime_integrity_manifest(
            &root,
            AiComputeBackend::Cpu,
            None,
            BTreeMap::from([("llama-cpu.zip".to_string(), "0".repeat(64))]),
            None,
        )
        .unwrap();
        assert!(is_llama_runtime(&root));
        fs::write(root.join("llama-server.exe"), b"modified runtime executable").unwrap();
        assert!(!is_llama_runtime(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn runtime_catalog_uses_manifest_path_without_scanning_other_directories() {
        let root = std::env::temp_dir().join(format!(
            "subhooper-ai-catalog-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::write(root.join("bin").join("llama-server.exe"), b"verified executable").unwrap();
        write_runtime_integrity_manifest(
            &root,
            AiComputeBackend::Cpu,
            None,
            BTreeMap::from([("llama-cpu.zip".to_string(), "0".repeat(64))]),
            None,
        ).unwrap();
        fs::create_dir_all(root.join("unrelated")).unwrap();
        assert_eq!(verify_llama_runtime_files(&root, false, None).unwrap(), root.join("bin").join("llama-server.exe"));
        fs::write(root.join("unrelated").join("unexpected.bin"), b"unexpected").unwrap();
        assert!(verify_llama_runtime_files(&root, true, None).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn runtime_tree_rejects_symlink_cycle_without_recursing() {
        let root = std::env::temp_dir().join(format!(
            "subhooper-ai-cycle-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(root.join("nested")).unwrap();
        std::os::unix::fs::symlink(&root, root.join("nested").join("cycle")).unwrap();
        assert!(find_file(&root, "llama-server.exe").is_none());
        assert!(collect_runtime_digests(&root, &root, None).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_download_url_uses_pinned_revision_and_one_separator() {
        let model = AI_MODELS[0];
        let normal = model_download_url(model.repository, model.revision, model.filename).unwrap();
        let leading = model_download_url(model.repository, model.revision, &format!("/{}", model.filename)).unwrap();
        for url in [normal, leading] {
            assert!(url.contains(&format!("/resolve/{}/{}", model.revision, model.filename)));
            assert!(!url.contains("//Qwen3-4B-Q4_K_M.gguf"));
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
