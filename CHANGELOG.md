# Changelog

## 0.3.7 beta

- Fixed the Windows installer resource list so `app/VERSION.txt` is available
  to the installed extraction pipeline.
- Added a safe pipeline fallback when version metadata is unexpectedly missing.
- Made VideoSubFinder downloads resilient to SourceForge redirect pages by
  using `curl.exe` when available, retrying failed or unverified responses, and
  falling back across official SourceForge endpoints.
- Kept strict SHA-256 verification for VideoSubFinder and the private Python
  runtime; unverified downloads are never installed.

## 0.3.6 beta

- Added a normal per-machine Windows installer.
- Added signed GitHub Releases update support.
- Added an opt-in component manager for VideoSubFinder and the OCR runtime.
- Moved mutable results, reports, models, downloads, and runtimes to
  `%LOCALAPPDATA%\SubHooper`.
- Preserved direct handoff from extracted SRT to AI Cleaning.
- Added local Qwen3 4B and recommended Qwen3 8B cleanup models.
- Added Light, Balanced, and Strong cleanup levels.
- Added original-versus-cleaned translation source selection.
- Kept DeepL API Free and Pro as explicit personal-key options with online data
  transfer confirmation.
