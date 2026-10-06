import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

import download_models


class FakeResponse:
    def __init__(self, payload, content_length=None):
        self.stream = io.BytesIO(payload)
        self.headers = {}
        if content_length is not None:
            self.headers['Content-Length'] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.stream.close()

    def read(self, size=-1):
        return self.stream.read(size)


class DownloadModelTests(unittest.TestCase):
    def _fetch(self, payload, content_length, expected, max_bytes):
        response = FakeResponse(payload, content_length)
        return patch('download_models.urlopen', return_value=response)

    def test_rejects_oversize_declared_length_before_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'model.onnx'
            destination.write_bytes(b'existing')
            with self._fetch(b'', 5, 'unused', 4) as mocked:
                with self.assertRaisesRegex(RuntimeError, 'size exceeds'):
                    download_models.fetch('https://example.test/model', destination, 'unused', 4)
                mocked.assert_called_once()
            self.assertEqual(destination.read_bytes(), b'existing')
            self.assertEqual(list(Path(folder).iterdir()), [destination])

    def test_rejects_oversize_stream_without_content_length(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'model.onnx'
            with self._fetch(b'12345', None, 'unused', 4):
                with self.assertRaisesRegex(RuntimeError, 'size exceeds'):
                    download_models.fetch('https://example.test/model', destination, 'unused', 4)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_wrong_hash_cleans_partial_and_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'model.onnx'
            destination.write_bytes(b'previous verified file')
            with self._fetch(b'bad payload', None, 'wrong digest', 64):
                with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                    download_models.fetch('https://example.test/model', destination, 'wrong digest', 64)
            self.assertEqual(destination.read_bytes(), b'previous verified file')
            self.assertEqual(list(Path(folder).iterdir()), [destination])

    def test_valid_download_replaces_destination_after_verification(self):
        import hashlib

        payload = b'verified model'
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'model.onnx'
            destination.write_bytes(b'old file')
            with self._fetch(payload, len(payload), digest, len(payload)):
                download_models.fetch('https://example.test/model', destination, digest, len(payload))
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(list(Path(folder).iterdir()), [destination])


if __name__ == '__main__':
    unittest.main()
