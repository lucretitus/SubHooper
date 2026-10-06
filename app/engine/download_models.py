"""Download the version-pinned PP-OCRv6 small models with SHA-256 checks."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import uuid
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


FILES = {
    "PP-OCRv6_det_small.onnx": (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    ),
    "PP-OCRv6_rec_small.onnx": (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    ),
    "ppocrv6_dict.txt": (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt",
        # The upstream v3.9.2 manifest lists this URL but no dictionary digest.
        # This SHA-256 was computed from the version-pinned official response.
        "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
    ),
}

MAX_BYTES = {
    "PP-OCRv6_det_small.onnx": 128 * 1024 * 1024,
    "PP-OCRv6_rec_small.onnx": 256 * 1024 * 1024,
    "ppocrv6_dict.txt": 8 * 1024 * 1024,
}


def _open_model_response(url):
    headers = {"User-Agent": "SubHooper/0.4.3", "Cache-Control": "no-cache"}
    try:
        return urlopen(Request(url, headers=headers), timeout=60)
    except HTTPError as error:
        if error.code != 403 or urlsplit(url).hostname not in {'www.modelscope.cn', 'modelscope.cn'}:
            raise
        error.close()
        # ModelScope can cache an expired signed redirect. Refresh the same
        # versioned asset once; its SHA-256 and size limits remain mandatory.
        separator = '&' if '?' in url else '?'
        refreshed = f'{url}{separator}download=true&subhooper_request={uuid.uuid4().hex}'
        return urlopen(Request(refreshed, headers=headers), timeout=60)


def fetch(url: str, destination: Path, expected: str, max_bytes: int) -> None:
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output, _open_model_response(url) as response:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else 0
            if total < 0 or total > max_bytes:
                raise RuntimeError(f"Download size exceeds the limit for {destination.name}.")
            received = 0
            next_update = 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if received + len(chunk) > max_bytes:
                    raise RuntimeError(f"Download size exceeds the limit for {destination.name}.")
                digest.update(chunk)
                output.write(chunk)
                received += len(chunk)
                if received >= next_update:
                    if total:
                        print(f"OCR model {destination.name}: {min(100, received * 100 // total)}% ({received / 1048576:.1f} MB)", flush=True)
                    else:
                        print(f"OCR model {destination.name}: {received / 1048576:.1f} MB received", flush=True)
                    next_update = received + 1024 * 1024
        actual = digest.hexdigest()
        if actual.lower() != expected:
            raise RuntimeError(f"SHA-256 mismatch for {destination.name}: {actual}")
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: download_models.py MODEL_DIRECTORY", file=sys.stderr)
        return 2
    directory = Path(sys.argv[1]).expanduser().resolve()
    for filename, (url, expected) in FILES.items():
        target = directory / filename
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"Verified existing {filename}")
            continue
        print(f"Downloading and verifying {filename}...", flush=True)
        fetch(url, target, expected, MAX_BYTES[filename])
    print(f"Models verified in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
