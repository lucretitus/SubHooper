# SubHooper

SubHooper is an open-source Windows desktop application that turns hardcoded
video subtitles into editable SRT files through a single, clean interface. It
automates subtitle-frame extraction, OCR, review, local AI-assisted cleanup,
optional translation, and export.

> **Beta status:** SubHooper 0.3.6 is an early public beta. Important subtitle
> output should be reviewed before production or screening use.

## Download

### [Download the latest Windows installer](https://github.com/lucretitus/SubHooper/releases/latest)

Open the latest release and download the Windows setup file ending in
`_x64-setup.exe`. The source-code archives generated automatically by GitHub are
not the application installer.

SubHooper installs under `C:\Program Files\SubHooper` and creates a normal
Windows application entry. Node.js, Rust, a separate Python installation, and a
copy of the source repository are not required for normal use.

Windows may display a SmartScreen warning because the beta installer does not
currently carry a commercial Authenticode certificate. The release updater uses
its own cryptographic signature, but that signature does not replace Windows
publisher signing.

## What SubHooper does

- Extracts hardcoded subtitle frames from video.
- Runs OCR and produces an editable SRT file.
- Preserves the original OCR result for comparison.
- Supports CPU and compatible NVIDIA GPU processing with fallback behavior.
- Imports an existing SRT directly into the AI Cleaning workspace.
- Offers multiple cleanup strengths for conservative or stronger subtitle
  repair.
- Uses optional local Qwen3 models for subtitle cleaning without uploading the
  video or SRT.
- Offers optional DeepL translation using a personal DeepL API account.
- Exports cleaned or translated SRT files while keeping reports and results
  between application updates.

## Typical workflow

1. Open a video in SubHooper.
2. Select the subtitle area and processing options.
3. Start extraction and OCR.
4. Review the original SRT in the AI Cleaning panel.
5. Optionally clean the subtitles with a local Qwen3 model.
6. Optionally translate either the original or cleaned result.
7. Export the selected SRT.

## First-run components

Video extraction and OCR components are not embedded in the installer. On first
use, open **Settings > Video extraction components**, review the notice, and
approve the component installation. SubHooper then prepares its managed runtime
under `%LOCALAPPDATA%\SubHooper`.

This process does not require a separate manual installation of VideoSubFinder,
RapidVideOCR, Python, Node.js, or Rust. AI Cleaning can also be used independently
by importing an existing SRT without installing the video-extraction components.

## Core projects

SubHooper provides the unified interface, workflow automation, component
management, result review, AI cleanup, translation controls, and export layer.
Its core subtitle-extraction pipeline works in the background with:

- [VideoSubFinder](https://github.com/SWHL/VideoSubFinder) for detecting and
  extracting hardcoded subtitle frames.
- [RapidVideOCR](https://github.com/SWHL/RapidVideOCR) for converting extracted
  subtitle images into timed subtitle text.

VideoSubFinder and RapidVideOCR are independent third-party projects maintained
by their respective authors. They are downloaded only after explicit approval
and retain their own licenses. They are not presented as part of the SubHooper
source code. Additional component and license information is available in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## AI cleaning

Local AI cleaning is optional. Qwen3 models run through a locally managed
`llama.cpp` runtime. Qwen3 8B is the recommended quality option for systems with
enough memory; smaller models remain available for lower-resource computers.

Cleanup strength controls how aggressively OCR noise, unrelated symbols,
foreign-language fragments, broken lines, and obvious recognition errors are
handled. AI output is not guaranteed to be correct and should remain subject to
human review.

## Translation

DeepL translation is optional and requires a personal DeepL API key, including
for DeepL API Free. SubHooper does not provide a shared key, create subscriptions,
or change account spending limits.

Before an online translation starts, the application identifies that subtitle
text will be sent to an external service. When both versions are available, the
original or AI-cleaned SRT can be selected as the translation source.

## Privacy and network use

- Video files and extracted frames remain on the computer.
- OCR processing and local Qwen3 cleaning remain on the computer.
- Component downloads occur only after approval and contact the relevant
  official distribution services.
- Local AI model installation downloads the selected model and `llama.cpp`.
- DeepL is the only built-in online translation provider. When selected, the
  subtitle text and limited context required for translation are sent to DeepL.
- Update checks contact this repository's GitHub Releases endpoint.

SubHooper does not include analytics, advertising, or a SubHooper-operated cloud
service.

## Files and updates

Application data is stored under:

```text
%LOCALAPPDATA%\SubHooper
```

This location contains managed components, AI models, reports, results, and
working data. Installing an application update does not intentionally remove
these files.

New versions are distributed through GitHub Releases. The application can check
for a signed update from **Settings > Check for updates**.

## System support

- 64-bit Windows
- Windows 11 is the currently tested platform
- Internet access for first-run component downloads, model downloads, optional
  DeepL translation, and update checks
- An NVIDIA GPU is optional; supported work can fall back to CPU processing
- Local Qwen3 model requirements vary according to model size

## Source code and contributions

The source code is available in this repository for transparency, auditing, bug
reports, and continued development. Normal users should install SubHooper from
the latest GitHub Release rather than building the application from source.

Technical contribution notes are kept separately in
[CONTRIBUTING.md](CONTRIBUTING.md) so that the main page remains focused on
installation and use.

## Development credit

SubHooper was developed with assistance from OpenAI Codex running GPT-5.6 Sol
during the beta development process. Generated and modified code remains subject
to the project's testing, review, and licensing requirements.

## License

SubHooper source code is licensed under the [MIT License](LICENSE). Downloaded
third-party components retain their own licenses and terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
