# Third-party notices

SubHooper 0.3.6 beta does not bundle the following runtime components. The
application downloads them only after explicit approval and stores them outside
the installed application directory. Each component remains an independent work
under its own license and terms.

| Component | Pinned source/status | License or terms |
| --- | --- | --- |
| VideoSubFinder 6.10 x64 | SourceForge archive; SHA-256 `3c0cc03793ec9753a6a4ee8a91c1d226c20b80aab901718f7c97d4fcb3580c0e` | GPL-2.0 |
| Microsoft Visual C++ 2015–2022 x64 runtime | Microsoft stable download; Microsoft Authenticode signature required | Microsoft license terms |
| Python 3.14.7 x64 | Python.org installer; SHA-256 `9d9eb2709ef81bf5cd30db3c2096bdbc4ea10087c22e62f27d356b36f6ae9649` | Python Software Foundation License |
| RapidVideOCR 3.1.1 | Installed from the Python package index | Apache-2.0 |
| RapidOCR 3.9.2 | Installed from the Python package index | Apache-2.0; model notices may also apply |
| ONNX Runtime 1.29.0 | Installed from the Python package index | MIT |
| Qwen3 GGUF models | Optional official Hugging Face download initiated in AI Cleaning | Apache-2.0 |
| llama.cpp | Official Windows release downloaded and verified when a local model is installed | MIT |
| DeepL API | Optional remote service; no key, SDK, or DeepL code is bundled | Personal DeepL account terms and privacy policy |

Upstream sources:

- https://sourceforge.net/projects/videosubfinder/
- https://github.com/SWHL/VideoSubFinder
- https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist
- https://www.python.org/downloads/release/python-3147/
- https://github.com/SWHL/RapidVideOCR
- https://github.com/RapidAI/RapidOCR
- https://github.com/microsoft/onnxruntime
- https://github.com/QwenLM/Qwen3
- https://github.com/ggml-org/llama.cpp
- https://developers.deepl.com/

## Compiled application dependencies

The Windows application is built with Tauri and Rust ecosystem crates. The
frontend includes React, ReactDOM, and production assets produced by Vite. Their
license texts and notices can be obtained from their package registries and
source repositories. SubHooper source code is provided under `LICENSE`.

This notice is informational and is not legal advice.
