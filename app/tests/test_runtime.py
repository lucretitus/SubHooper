import sys
import io
from contextlib import redirect_stdout
from subprocess import TimeoutExpired
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from config import get_region_profile, validate_region_offsets
from pipeline import read_package_version, resolve_results_root
from runtime import (Workspace, cleanup_session_workspaces,
                     _terminate_process_tree, normalize_srt, run_process,
                     select_compute_mode)
from runtime import _remove_owned_workspace


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

    def test_workspace_cleanup_checks_owner(self):
        work = Workspace()
        (work.path / '.owner').write_text('wrong')
        with self.assertRaises(RuntimeError):
            work.cleanup()
        (work.path / '.owner').write_text(work.token)
        work.cleanup()
        self.assertFalse(work.path.exists())

    def test_session_cleanup_removes_only_matching_owned_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            session_token = 'a' * 32
            with patch('runtime.tempfile.gettempdir', return_value=folder), \
                 patch.dict('os.environ', {'SUBHOOPER_SESSION_TOKEN': session_token}):
                matching = Workspace()
            with patch('runtime.tempfile.gettempdir', return_value=folder), \
                 patch.dict('os.environ', {'SUBHOOPER_SESSION_TOKEN': 'b' * 32}):
                unrelated = Workspace()
            self.assertEqual(cleanup_session_workspaces(session_token, folder), 1)
            self.assertFalse(matching.path.exists())
            self.assertTrue(unrelated.path.exists())
            unrelated.cleanup()

    def test_session_cleanup_respects_keep_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            token = 'c' * 32
            with patch('runtime.tempfile.gettempdir', return_value=folder), \
                 patch.dict('os.environ', {'SUBHOOPER_SESSION_TOKEN': token}):
                kept = Workspace()
            (kept.path / '.keep-temp').write_text('true', encoding='ascii')
            self.assertEqual(cleanup_session_workspaces(token, folder), 0)
            self.assertTrue(kept.path.exists())
            kept.cleanup()

    def test_session_cleanup_skips_workspace_root_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'base'
            root.mkdir()
            token = 'e' * 32
            target = Path(folder) / 'outside' / 'subtitle-poc-target'
            target.parent.mkdir()
            target.mkdir()
            (target / '.owner').write_text('f' * 32, encoding='ascii')
            (target / '.session').write_text(token, encoding='ascii')
            link = root / 'subtitle-poc-link'
            try:
                link.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest('Directory symlinks are unavailable on this platform.')
            self.assertEqual(cleanup_session_workspaces(token, root), 0)
            self.assertTrue(target.exists())

    def test_workspace_cleanup_refuses_root_symlink_without_deleting_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'subtitle-poc-target'
            target.mkdir()
            (target / '.owner').write_text('d' * 32, encoding='ascii')
            link = root / 'subtitle-poc-link'
            try:
                link.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest('Directory symlinks are unavailable on this platform.')
            with self.assertRaises(RuntimeError):
                _remove_owned_workspace(link, root, 'd' * 32)
            self.assertTrue(target.exists())

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
        self.assertEqual(offsets['top'], '0.88')
        self.assertEqual(offsets['bottom'], '0.12')

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

    def test_native_progress_is_forwarded_and_diagnostics_stay_in_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            display = io.StringIO()
            with redirect_stdout(display):
                run_process([sys.executable, '-c', "print('OCRProgress=42'); print('private diagnostic')"],
                            root, root / 'progress.log', 10, progress_prefix='OCRProgress=')
            self.assertEqual(display.getvalue(), 'OCRProgress=42\n')
            self.assertIn('private diagnostic', (root / 'progress.log').read_text())

    def test_progress_after_utf16_native_diagnostic_is_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            display = io.StringIO()
            script = ("import sys; "
                      "sys.stdout.buffer.write('native diagnostic\\n'.encode('utf-16-le')); "
                      "sys.stdout.buffer.write(b'OCRProgress=42\\n'); "
                      "sys.stdout.buffer.flush()")
            with redirect_stdout(display):
                run_process([sys.executable, '-c', script], root,
                            root / 'progress.log', 10, progress_prefix='OCRProgress=')
            self.assertEqual(display.getvalue(), 'OCRProgress=42\n')
            log = (root / 'progress.log').read_text(encoding='utf-8')
            self.assertIn('native diagnostic', log)
            self.assertNotIn('\x00', log)

    def test_utf16_and_ansi_progress_is_forwarded_without_control_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            display = io.StringIO()
            script = ("import sys; "
                      "sys.stdout.buffer.write('\\x1b[0;93mOCRProgress=67\\x1b[m\\n'"
                      ".encode('utf-16-le')); sys.stdout.buffer.flush()")
            with redirect_stdout(display):
                run_process([sys.executable, '-c', script], root,
                            root / 'progress.log', 10, progress_prefix='OCRProgress=')
            self.assertEqual(display.getvalue(), 'OCRProgress=67\n')
            log = (root / 'progress.log').read_text(encoding='utf-8')
            self.assertNotIn('\x00', log)
            # The command header can contain escaped ANSI text, not raw ESC.
            self.assertNotIn('\x1b', log)

    def test_strict_session_cleanup_reports_failure_and_preserves_ownership(self):
        session = 'a' * 32
        with tempfile.TemporaryDirectory() as folder, \
             patch('runtime.tempfile.gettempdir', return_value=folder), \
             patch.dict('os.environ', {'SUBHOOPER_SESSION_TOKEN': session}):
            workspace = Workspace()
            with patch('runtime.shutil.rmtree', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    cleanup_session_workspaces(session, base=folder, strict=True)
            self.assertEqual((workspace.path / '.session').read_text(), session)
            self.assertEqual((workspace.path / '.owner').read_text(), workspace.token)
            self.assertEqual(cleanup_session_workspaces(session, base=folder, strict=True), 1)

    def test_windows_kill_fallback_stops_direct_process_on_taskkill_failure(self):
        class FakeProcess:
            pid = 123

            def __init__(self):
                self.alive = True
                self.kill_calls = 0

            def poll(self):
                return None if self.alive else -9

            def kill(self):
                self.kill_calls += 1
                self.alive = False

            def wait(self, timeout=None):
                if self.alive:
                    raise TimeoutExpired('process', timeout)
                return -9

        process = FakeProcess()
        with patch('runtime.subprocess.CREATE_NO_WINDOW', 0, create=True), \
             patch('runtime.subprocess.run', return_value=type('Result', (), {'returncode': 1})()):
            _terminate_process_tree(process, io.StringIO(), windows=True)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(process.alive)


if __name__ == '__main__':
    unittest.main()
