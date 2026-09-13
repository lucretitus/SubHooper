import importlib.metadata as md
import json
import struct
import sys

EXPECTED = {'rapid_videocr': '3.1.1', 'rapidocr': '3.9.2', 'onnxruntime': '1.29.0'}

def verify():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('A project-local virtual environment is required.')
    if sys.version_info[:3] != (3, 14, 7) or struct.calcsize('P') != 8:
        raise RuntimeError('Python 3.14.7 x64 is required.')
    versions = {name: md.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f'OCR package versions do not match: {versions}')
    return {'python': sys.version, 'executable': sys.executable, **versions}

if __name__ == '__main__':
    print(json.dumps(verify()))
