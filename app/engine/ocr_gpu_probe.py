import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    import onnxruntime as ort

    preload = getattr(ort, 'preload_dlls', None)
    if callable(preload):
        preload(directory='')
    providers = list(ort.get_available_providers())
    import rapidocr
    model_path = Path(rapidocr.__file__).parent / 'models' / 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'
    if not model_path.is_file():
        raise RuntimeError(f'GPU probe model was not found: {model_path}')
    session = ort.InferenceSession(
        str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    session_providers = list(session.get_providers())
    versions = {
        'rapid_videocr': metadata.version('rapid_videocr'),
        'rapidocr': metadata.version('rapidocr'),
        'onnxruntime_gpu': metadata.version('onnxruntime-gpu'),
    }
    result = {
        'python': sys.version,
        'executable': sys.executable,
        **versions,
        'providers': providers,
        'session_providers': session_providers,
    }
    print(json.dumps(result, ensure_ascii=False))
    if sys.version_info[:3] != (3, 14, 7):
        raise RuntimeError(f'Python 3.14.7 is required: {sys.version}')
    if versions['rapid_videocr'] != '3.1.1' or versions['rapidocr'] != '3.9.2':
        raise RuntimeError(f'OCR package versions do not match: {versions}')
    if versions['onnxruntime_gpu'] not in {'1.29.0', '1.26.0'}:
        raise RuntimeError(f'ONNX Runtime GPU version does not match: {versions}')
    if 'CUDAExecutionProvider' not in providers:
        raise RuntimeError(f'CUDAExecutionProvider was not found: {providers}')
    if not session_providers or session_providers[0] != 'CUDAExecutionProvider':
        raise RuntimeError(f'CUDA oturumu etkin degil: {session_providers}')


if __name__ == '__main__':
    main()
