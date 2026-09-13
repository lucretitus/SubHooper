import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from config import build_vsf_command, get_region_profile, validate_region_offsets
from ocr_worker import compact_text, detector_format_suspicious, fuse_text, parse_srt, timestamp_from_image
from pipeline import read_package_version, resolve_results_root
from runtime import Workspace, cleanup_stale_workspaces, normalize_srt, run_process, select_compute_mode


class RuntimeTests(unittest.TestCase):
    def test_package_version_reads_bundled_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'VERSION.txt').write_text('0.3.7\n', encoding='ascii')
            self.assertEqual(read_package_version(root), '0.3.7')

    def test_package_version_has_safe_missing_file_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('os.environ', {}, clear=True):
                self.assertEqual(read_package_version(Path(folder)), 'unknown')

    def test_results_root_can_live_outside_application(self):
        with tempfile.TemporaryDirectory() as folder:
            configured = Path(folder) / 'results'
            with patch.dict('os.environ', {'SUBTITLE_RESULTS_ROOT': str(configured)}):
                self.assertEqual(resolve_results_root(Path(folder) / 'app'), configured.resolve())

    def test_ocr_fusion_preserves_detector_spacing(self):
        selected, reason = fuse_text("I'll do what I want.", 'Illdowhat Iwant.', 0.99)
        self.assertEqual(selected, "I'll do what I want.")
        self.assertEqual(reason, 'detector_equivalent')

    def test_ocr_fusion_repairs_leading_punctuation(self):
        selected, reason = fuse_text(
            'We gotta get it first.\n. Think later.',
            'We gotta get it first. Think later.', 0.983)
        self.assertEqual(selected, 'We gotta get it first. Think later.')
        self.assertEqual(reason, 'recognizer_format_repair')
        self.assertTrue(detector_format_suspicious('. Think later.'))

    def test_timestamp_and_unicode_compaction(self):
        image = Path('0_02_39_800__0_02_44_119_1005709441805007619201080.jpeg')
        self.assertEqual(timestamp_from_image(image), '00:02:39,800 --> 00:02:44,119')
        self.assertEqual(compact_text('Good café!'), 'goodcafé')

    def test_worker_srt_parser_keeps_empty_cues(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'detector.srt'
            path.write_text(
                '1\n00:00:01,000 --> 00:00:02,000\nText\n\n'
                '2\n00:00:03,000 --> 00:00:04,000\n\n', encoding='utf-8')
            cues = parse_srt(path)
            self.assertEqual(cues['00:00:01,000 --> 00:00:02,000'], 'Text')
            self.assertEqual(cues['00:00:03,000 --> 00:00:04,000'], '')

    def test_workspace_cleanup_checks_owner(self):
        work = Workspace()
        (work.path / '.owner').write_text('wrong')
        with self.assertRaises(RuntimeError):
            work.cleanup()
        (work.path / '.owner').write_text(work.token)
        work.cleanup()
        self.assertFalse(work.path.exists())

    def test_normalize_srt_drops_empty_and_merges_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.srt'
            path.write_text(
                '1\n00:00:01,000 --> 00:00:02,000\nSame text\n\n'
                '2\n00:00:02,300 --> 00:00:03,000\nSame   text\n\n'
                '3\n00:00:04,000 --> 00:00:05,000\n\n', encoding='utf-8')
            stats = normalize_srt(path)
            self.assertEqual(stats['subtitle_count'], 1)
            self.assertEqual(stats['empty_cues_dropped'], 1)
            self.assertEqual(stats['duplicate_cues_merged'], 1)

    def test_region_profiles_and_custom_offsets(self):
        self.assertLess(float(get_region_profile('bottom')['top']), float(get_region_profile('full')['top']))
        offsets = validate_region_offsets(0.88, 0.12, 0.15, 0.91)
        command = build_vsf_command('vsf.exe', 'input.mov', 'output', 'custom', 'cpu', offsets)
        self.assertEqual(command[command.index('-te') + 1], '0.88')
        self.assertEqual(command[command.index('-be') + 1], '0.12')

    def test_vsf_cpu_and_cuda_commands(self):
        cpu = build_vsf_command('vsf.exe', 'input.mov', 'output', 'bottom', 'cpu')
        cuda = build_vsf_command('vsf.exe', 'input.mov', 'output', 'bottom', 'cuda')
        self.assertEqual(cpu[cpu.index('-use_cuda_gpu') + 1], '0')
        self.assertIn('-uc', cuda)

    def test_compute_selection(self):
        class Result:
            returncode = 0
            stdout = 'NVIDIA Test GPU\n'

        import runtime
        original = runtime.shutil.which
        try:
            runtime.shutil.which = lambda name: 'nvidia-smi' if name == 'nvidia-smi' else None
            auto = select_compute_mode('auto', runner=lambda *args, **kwargs: Result())
        finally:
            runtime.shutil.which = original
        self.assertEqual(auto['selected'], 'cuda')

    def test_process_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaises(RuntimeError):
                run_process([sys.executable, '-c', 'raise SystemExit(7)'], path, path / 'fail.log', 10)


if __name__ == '__main__':
    unittest.main()
