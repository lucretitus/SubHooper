# Security

Report suspected vulnerabilities privately through GitHub's private vulnerability reporting feature. Avoid attaching videos, subtitle text, or personal diagnostic logs to public reports.

## Downloads and execution

Pinned Python, OpenCV, OCR model, and Qwen3 downloads are checked with SHA-256. The llama.cpp installer verifies archives against digests from the official pinned GitHub release and records file digests for later integrity checks. Downloads have size limits. Update packages require the existing Tauri updater signature.

Local runtimes and verification records are stored under the current Windows account. These checks detect ordinary corruption and file changes; they do not protect against an attacker who already controls that account or its files. Model verification records use file fingerprints to avoid rehashing unchanged large files.

## Local processing

OCR and local AI do not send video or subtitle content to a remote service. AI listens on loopback with a random per-run key and process ownership checks. The application checks GPU offload when GPU inference is selected. Temporary cleanup is limited to owned workspaces.

The video asset protocol starts with no filesystem access. File dialogs and native drag-and-drop grant access to selected files. UI text is rendered without inserting subtitle content as HTML.

## Maintenance

The application includes third-party libraries and downloaded runtimes with their own security policies. Keep Windows, WebView2, and graphics drivers updated. Dependency and source review reduce risk but do not guarantee the absence of vulnerabilities.
