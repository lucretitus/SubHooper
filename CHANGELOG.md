# Changelog

## 0.4.3

### Features

- Local ONNX OCR with selectable video regions and CPU or GPU + CPU processing.
- Experimental AMD/Intel DirectML OCR, including DirectX 12-capable integrated graphics, with an isolated runtime and hardware execution checks.
- Local Qwen3 4B, 8B, and 14B models for subtitle cleanup and translation.
- Light, Balanced, and Strong cleanup, plus translation from original or cleaned subtitles.
- SRT, TTML, text, and Markdown export; original subtitles remain available.

### Fixes

- Check the selected GPU + CPU runtime before extraction, open component verification when it is unavailable, and retain validation errors in pipeline reports.
- Honor the selected SRT, TTML, text, or Markdown format when exporting original subtitles.
- Remove recognized legacy VideoSubFinder 6.10 and RapidVideOCR Python 3.14 environments after native component setup validates successfully; retain unrecognized or linked paths.
- Restore the original CUDA inference and single-crop retry path; remove slow cuDNN DEFAULT recovery.
- Pin and validate NVIDIA CUDA dependencies, replacing incompatible managed environments during component setup.
- Record the NVIDIA runtime package versions in OCR reports.
- Normalize mixed Windows log encodings before forwarding live OCR progress.
- Prefer high-performance DirectML hardware, overlap bounded GPU detection with CPU preparation, and stop on explicit GPU failures.
- Allow component verification after CPU setup so outdated or missing GPU runtimes can be repaired.
- Constrain grouped local AI output to valid JSON and the requested decision count before validating cue IDs and order.
- Install NumPy before Python checks in both GitHub workflows.
- Refresh expired ModelScope download redirects once while retaining asset hash and size verification.
- Read source videos directly and clean owned temporary workspaces after processing, cancellation, and application close.
- Correct NVIDIA memory classification and check available memory before loading an AI model.
- Reuse verified, unchanged model files instead of hashing them on every action.
- Preserve target subtitle IDs and order when AI returns extra or incomplete output; retry invalid single-cue responses once.
- Report preparation progress correctly and make model verification cancellable.
- Prevent AI processes from starting or being retained after application close.
- Keep real processing errors visible and align the selection overlay with the displayed video frame.
- Select an available loopback port for development startup.

### Security and packaging

- Bound component, OCR model, and AI downloads; remove incomplete files on failure.
- Limit stalled network operations and preserve signed updates and pinned download verification.
- Update the TLS dependency to rustls 0.23.45.
- Restrict video asset access to files explicitly selected or dropped into the app.
- Pin build actions, validate release tags, and bundle only required runtime scripts and notices.
