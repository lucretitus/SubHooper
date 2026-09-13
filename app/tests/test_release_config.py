import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReleaseConfigurationTests(unittest.TestCase):
    def test_application_versions_are_consistent(self):
        expected = (ROOT / 'app' / 'VERSION.txt').read_text(encoding='ascii').strip()
        package = json.loads((ROOT / 'app' / 'gui' / 'package.json').read_text(encoding='utf-8'))
        tauri = json.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'tauri.conf.json').read_text(encoding='utf-8'))
        cargo = tomllib.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'Cargo.toml').read_text(encoding='utf-8'))

        self.assertEqual(expected, '0.3.7')
        self.assertEqual(package['version'], expected)
        self.assertEqual(tauri['version'], expected)
        self.assertEqual(cargo['package']['version'], expected)

    def test_installer_bundles_pipeline_version_metadata(self):
        tauri = json.loads(
            (ROOT / 'app' / 'gui' / 'src-tauri' / 'tauri.conf.json').read_text(encoding='utf-8'))
        resources = tauri['bundle']['resources']
        self.assertEqual(resources.get('../../VERSION.txt'), 'app/VERSION.txt')
        self.assertTrue((ROOT / 'app' / 'VERSION.txt').is_file())

    def test_component_downloads_keep_pinned_verification(self):
        installer = (ROOT / 'app' / 'install-components.ps1').read_text(encoding='utf-8')
        self.assertIn(
            '3c0cc03793ec9753a6a4ee8a91c1d226c20b80aab901718f7c97d4fcb3580c0e',
            installer)
        self.assertIn(
            '9d9eb2709ef81bf5cd30db3c2096bdbc4ea10087c22e62f27d356b36f6ae9649',
            installer)
        self.assertIn('curl.exe', installer)
        self.assertIn('official sources', installer)


if __name__ == '__main__':
    unittest.main()
