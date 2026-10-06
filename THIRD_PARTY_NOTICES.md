# Third-party notices

SubHooper 0.4.3 does not bundle the following runtime components.
They are installed after user approval under `%LOCALAPPDATA%\SubHooper` and
remain independent works under their own licenses and terms.

| Component | Source/status | License or terms |
| --- | --- | --- |
| Python 3.13.13 x64 | Python.org installer; SHA-256 `3c9c81d80f91c002ced86d645422d81432c68c7d9b6b0e974768ca2e449a4d00` | Python Software Foundation License |
| OpenCV, NumPy | Pinned Python package index downloads | Apache-2.0 (OpenCV); BSD-3-Clause (NumPy) |
| ONNX Runtime CPU 1.22.1 / CUDA 1.22.0 / DirectML 1.22.0 | Pinned Python package index downloads; isolated environments | MIT; NVIDIA dependency terms; bundled DirectML retains Microsoft's license |
| PP-OCRv6 small ONNX detector, recognizer and dictionary | Versioned RapidAI ModelScope assets with pinned SHA-256 in `app/engine/download_models.py` | Upstream model licenses and notices apply |
| Qwen3 4B / 8B / 14B Q4_K_M GGUF | Optional official Hugging Face downloads at fixed revisions and SHA-256 checks | Apache-2.0 model terms; inspect model cards |
| Upstream llama.cpp b10941 | Optional pinned release tag; archive digest supplied by official release API, installed files rechecked before execution | MIT and upstream third-party notices |

Upstream sources:

- https://www.python.org/downloads/release/python-31313/
- https://github.com/opencv/opencv
- https://github.com/numpy/numpy
- https://github.com/microsoft/onnxruntime
- https://github.com/microsoft/DirectML
- https://www.modelscope.cn/models/RapidAI/RapidOCR
- https://huggingface.co/Qwen/Qwen3-4B-GGUF
- https://huggingface.co/Qwen/Qwen3-8B-GGUF
- https://huggingface.co/Qwen/Qwen3-14B-GGUF
- https://github.com/ggml-org/llama.cpp

## Compiled application dependencies

The Windows application is built with Tauri and Rust ecosystem crates. The
frontend includes React, ReactDOM, and production assets produced by Vite. Their
license texts and notices can be obtained from their package registries and
source repositories. SubHooper source code is provided under `LICENSE`.

This notice is informational and is not legal advice.
