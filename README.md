# SubHooper

SubHooper is a Windows desktop application for extracting hardcoded subtitles
from video, reviewing the generated SRT, and optionally cleaning or translating
subtitle text with a local Qwen3 model or a personal DeepL API account.

This repository contains the **0.3.6 beta** source. End users install the signed
NSIS package from GitHub Releases; they do not need Node.js, Rust, or a separate
source folder.

## Installation

1. Download `SubHooper_0.3.6_x64-setup.exe` from the latest GitHub Release.
2. Run the installer and approve the Windows administrator prompt. SubHooper is
   installed for all users under `C:\Program Files\SubHooper`.
3. Open **Settings > Video extraction components** and review the component
   notice.
4. Approve **Download and install components**. SubHooper downloads verified
   copies of VideoSubFinder 6.10, a private Python 3.14.7 runtime, and pinned OCR
   packages into `%LOCALAPPDATA%\SubHooper`.

VideoSubFinder and the OCR runtime are not embedded in the SubHooper installer.
AI cleaning remains usable without the video-extraction components.

## Data and network behavior

- Videos, frames, OCR output, reports, and local AI prompts stay on the PC.
- The component installer contacts SourceForge, Python.org, and Python package
  repositories only after explicit approval.
- Local Qwen3 model downloads use official Hugging Face repositories selected in
  the application.
- DeepL translation is optional and requires a personal API key. Only subtitle
  text, nearby subtitle context, and optional terminology guidance are sent after
  a separate confirmation. API Free is locked to DeepL's free endpoint.
- Update checks contact this project's public GitHub Releases endpoint.

## Developer build

The easiest release build is the included GitHub Actions workflow. A local
Windows build is also available through `Build-Installer.cmd`; Node.js 22, the
stable Rust MSVC toolchain, Microsoft C++ Build Tools, and WebView2 are developer
requirements only.

Before the first release, run `Generate-Updater-Key.cmd`, store the private key
securely, add the documented repository secrets, and run
`Configure-GitHub.cmd`. Full steps are in `docs/GITHUB_RELEASE.md`.

## License

SubHooper source code is licensed under the MIT License. Downloaded components
retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

