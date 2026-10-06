import json
from pathlib import Path
import tomllib
import unittest
import sys
import hashlib
import io
import tempfile
from urllib.error import HTTPError
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from probe import CUDA_PACKAGES, verify_cuda_packages
from download_models import FILES, fetch


ROOT = Path(__file__).resolve().parents[2]


class ReleaseConfigurationTests(unittest.TestCase):
    def test_expired_model_redirect_retries_same_pinned_asset_and_verifies_bytes(self):
        url = FILES['PP-OCRv6_det_small.onnx'][0]
        content = b'verified test model'
        response = io.BytesIO(content)
        response.headers = {'Content-Length': str(len(content))}
        expired = HTTPError(url, 403, 'expired redirect', {}, io.BytesIO())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.onnx'
            with patch('download_models.urlopen', side_effect=[expired, response]) as opened:
                fetch(url, destination, hashlib.sha256(content).hexdigest(), 1024)
            self.assertEqual(destination.read_bytes(), content)
            self.assertTrue(opened.call_args_list[1].args[0].full_url.startswith(url + '?download=true&'))

    def test_repeated_model_http_failure_is_bounded_and_removes_partial_file(self):
        url = FILES['PP-OCRv6_det_small.onnx'][0]
        failures = [HTTPError(url, 403, 'forbidden', {}, io.BytesIO()) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.onnx'
            with patch('download_models.urlopen', side_effect=failures) as opened:
                with self.assertRaises(HTTPError):
                    fetch(url, destination, '0' * 64, 1024)
            self.assertEqual(opened.call_count, 2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cuda_versions_match_installer_and_successful_archive(self):
        installer = (ROOT / 'app' / 'install-components.ps1').read_text(encoding='utf-8')
        for name, version in CUDA_PACKAGES.items():
            self.assertIn(f"'{name}=={version}'", installer)
        self.assertEqual(CUDA_PACKAGES['nvidia-cudnn-cu12'], '9.26.0.51')
        self.assertEqual(CUDA_PACKAGES['nvidia-cuda-runtime-cu12'], '12.9.79')

    def test_cuda_probe_accepts_pinned_environment(self):
        with patch('probe.md.version', side_effect=CUDA_PACKAGES.__getitem__):
            self.assertEqual(verify_cuda_packages(), CUDA_PACKAGES)

    def test_cuda_probe_rejects_old_installed_cudnn(self):
        versions = dict(CUDA_PACKAGES, **{'nvidia-cudnn-cu12': '9.27.0.0'})
        with patch('probe.md.version', side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, 'CUDA package mismatch.*Components'):
                verify_cuda_packages()

    def test_cuda_probe_rejects_missing_dependency(self):
        from importlib.metadata import PackageNotFoundError
        with patch('probe.md.version', side_effect=PackageNotFoundError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'Missing CUDA package.*Components'):
                verify_cuda_packages()

    def test_application_versions_are_consistent(self):
        expected = (ROOT / 'app' / 'VERSION.txt').read_text(encoding='ascii').strip()
        package = json.loads((ROOT / 'app' / 'gui' / 'package.json').read_text(encoding='utf-8'))
        tauri = json.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'tauri.conf.json').read_text(encoding='utf-8'))
        cargo = tomllib.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'Cargo.toml').read_text(encoding='utf-8'))

        self.assertEqual(expected, '0.4.3')
        self.assertEqual(package['version'], expected)
        self.assertEqual(tauri['version'], expected)
        self.assertEqual(cargo['package']['version'], expected)

    def test_installer_bundles_pipeline_version_metadata(self):
        tauri = json.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'tauri.conf.json').read_text(encoding='utf-8'))
        resources = tauri['bundle']['resources']
        self.assertEqual(resources.get('../../VERSION.txt'), 'app/VERSION.txt')
        self.assertTrue((ROOT / 'app' / 'VERSION.txt').is_file())

    def test_bundle_resources_are_explicit_and_cover_the_engine(self):
        config_path = ROOT / 'app' / 'gui' / 'src-tauri' / 'tauri.conf.json'
        tauri = json.loads(config_path.read_text(encoding='utf-8'))
        resources = tauri['bundle']['resources']
        engine = ROOT / 'app' / 'engine'
        bundled_engine = {destination.removeprefix('app/engine/')
                          for destination in resources.values()
                          if destination.startswith('app/engine/')}
        self.assertEqual(bundled_engine, {path.name for path in engine.glob('*.py')})
        for source in resources:
            self.assertNotIn('*', source)
            self.assertTrue((config_path.parent / source).is_file(), source)
        self.assertIn('LICENSE', resources.values())
        self.assertIn('THIRD_PARTY_NOTICES.md', resources.values())

    def test_media_scope_requires_a_selected_or_dropped_file(self):
        tauri = json.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'tauri.conf.json').read_text(encoding='utf-8'))
        self.assertTrue(tauri['app']['security']['assetProtocol']['enable'])
        self.assertEqual(tauri['app']['security']['assetProtocol']['scope'], [])
        self.assertNotIn('connect-src *', tauri['app']['security']['csp'])

    def test_component_downloads_keep_pinned_verification(self):
        installer = (ROOT / 'app' / 'install-components.ps1').read_text(encoding='utf-8')
        self.assertIn('3c9c81d80f91c002ced86d645422d81432c68c7d9b6b0e974768ca2e449a4d00', installer)
        models = (ROOT / 'app' / 'engine' / 'download_models.py').read_text(encoding='utf-8')
        self.assertIn('090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f', models)
        self.assertIn('curl.exe', installer)
        self.assertIn('official sources', installer)


if __name__ == '__main__':
    unittest.main()
