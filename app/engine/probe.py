"""Validate the managed direct ONNX runtime and its immutable OCR assets."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import struct
import sys

from download_models import FILES


# NVIDIA dependencies are pinned to avoid incompatible package version drift.
CUDA_PACKAGES = {
    'nvidia-cuda-runtime-cu12': '12.9.79',
    'nvidia-cudnn-cu12': '9.26.0.51',
    'nvidia-cublas-cu12': '12.9.1.4',
    'nvidia-cuda-nvrtc-cu12': '12.9.86',
    'nvidia-cufft-cu12': '11.4.1.4',
    'nvidia-curand-cu12': '10.3.10.19',
    'nvidia-nvjitlink-cu12': '12.9.86',
}


def verify_cuda_packages():
    versions = {}
    for name, expected in CUDA_PACKAGES.items():
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError as error:
            raise RuntimeError(
                f'Missing CUDA package: {name}. Open Settings > Components and run setup.') from error
        if versions[name] != expected:
            raise RuntimeError(
                f'CUDA package mismatch: {name} {versions[name]}; expected {expected}. '
                'Open Settings > Components and run setup.')
    return versions


def verify(models_dir=None):
    import cv2
    import onnxruntime as ort

    if sys.prefix == sys.base_prefix:
        raise RuntimeError('A project-local virtual environment is required.')
    if sys.version_info[:3] != (3, 13, 13) or struct.calcsize('P') != 8:
        raise RuntimeError('Python 3.13.13 x64 is required.')
    expected = {'numpy': '2.2.6', 'opencv-python-headless': '4.14.0.94'}
    providers = ort.get_available_providers()
    package = ('onnxruntime-directml' if 'DmlExecutionProvider' in providers
               else ('onnxruntime-gpu' if 'CUDAExecutionProvider' in providers else 'onnxruntime'))
    version = md.version(package)
    if (package, version) not in (('onnxruntime', '1.22.1'), ('onnxruntime-gpu', '1.22.0'),
                                   ('onnxruntime-directml', '1.22.0')):
        raise RuntimeError(f'Unsupported ONNX Runtime: {package} {version}')
    cuda_versions = verify_cuda_packages() if package == 'onnxruntime-gpu' else {}
    versions = {name: md.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f'OCR package versions do not match: {versions}')
    if models_dir is not None:
        root = Path(models_dir)
        for filename, (_, digest) in FILES.items():
            path = root / filename
            if not path.is_file():
                raise RuntimeError(f'Missing OCR model: {filename}')
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(block)
            if hasher.hexdigest() != digest:
                raise RuntimeError(f'OCR model checksum mismatch: {filename}')
    if package == 'onnxruntime-directml' and models_dir is not None:
        # Confirm that these models actually execute on a hardware adapter.
        # Provider advertisement alone does not prove usable DirectX support.
        import numpy as np
        from native_ocr import DirectOnnxOCR
        ocr = DirectOnnxOCR(models_dir, 'dml', ort)
        for session, shape in ((ocr.detector, (1, 3, 64, 64)),
                               (ocr.recognizer, (1, 3, 48, 320))):
            session.run(None, {session.get_inputs()[0].name: np.zeros(shape, np.float32)})
    return {'python': sys.version, 'executable': sys.executable, 'opencv': cv2.__version__,
            package: version, 'providers': providers, **versions, **cuda_versions}


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1] if len(sys.argv) > 1 else None)))
