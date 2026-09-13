# Windows beta acceptance — 0.3.7

Use a clean Windows 10/11 x64 test account without Node.js, Rust, Python,
VideoSubFinder, RapidVideOCR, or a previous portable SubHooper folder.

- [ ] The NSIS setup EXE installs to `C:\Program Files\SubHooper` after one
      administrator prompt.
- [ ] SubHooper starts from the Start menu without Node.js or Rust.
- [ ] AI Cleaning opens and can import an SRT before OCR components are installed.
- [ ] Video extraction requests component setup instead of failing obscurely.
- [ ] Component setup displays licenses/hosts and requires explicit approval.
- [ ] VideoSubFinder and OCR runtime both show **Ready** after setup.
- [ ] Component setup recovers when one SourceForge endpoint returns HTML or a
      transient redirect response, without accepting a mismatched SHA-256.
- [ ] Installed extraction can read `C:\Program Files\SubHooper\app\VERSION.txt`.
- [ ] A short known video completes and its SRT opens in AI Cleaning automatically.
- [ ] Cancel works during VideoSubFinder, OCR, model download, and local AI.
- [ ] Qwen3 4B and 8B download, verify, clean, and export separate SRT files.
- [ ] Strong cleanup removes obvious symbol/numeric OCR-noise cues without changing
      timestamps of retained cues.
- [ ] DeepL is tested only with a test account and non-confidential subtitles.
- [ ] Results and reports are created under `%LOCALAPPDATA%\SubHooper`.
- [ ] A signed test update installs over the current Program Files version and
      preserves results, reports, models, and OCR components.
- [ ] Uninstall removes the application while user-created exports remain intact.

Record Windows version, GPU/CPU, installer filename, and any failed item in the
GitHub pre-release notes. Do not call the beta stable until this checklist passes.
