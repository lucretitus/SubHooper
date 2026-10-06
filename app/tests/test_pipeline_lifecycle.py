import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

import pipeline


class PipelineLifecycleTests(unittest.TestCase):
    def run_pipeline(self, failure=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / 'source.mp4'
            source_bytes = b'video source must be read in place'
            video.write_bytes(source_bytes)
            original_stat = video.stat()
            models = root / 'models'
            models.mkdir()
            results = root / 'results'
            workspaces = []
            native_inputs = []

            from runtime import Workspace

            def make_workspace():
                # Keep this integration test's owned temp directory isolated.
                with patch('runtime.tempfile.gettempdir', return_value=folder):
                    workspace = Workspace()
                workspaces.append(workspace)
                return workspace

            def fake_run_process(command, cwd, log_path, timeout, **kwargs):
                native_inputs.append(Path(command[2]))
                if failure == 'error':
                    raise RuntimeError('simulated OCR failure')
                if failure == 'cancel':
                    raise KeyboardInterrupt()
                output = Path(command[3])
                output.mkdir()
                (output / 'result.srt').write_text(
                    '1\n00:00:01,000 --> 00:00:02,000\nSubtitle\n', encoding='utf-8')
                (output / 'ocr-metrics.json').write_text(json.dumps({
                    'sampled_frames': 1, 'model_session_providers': {},
                    'compute_effective': 'CPU', 'ocr_stage_seconds': {},
                }), encoding='utf-8')
                return 0.1

            args = [str(video), '--models-dir', str(models), '--compute', 'cpu']
            with patch.object(sys, 'argv', ['pipeline.py', *args]), \
                 patch.dict(os.environ, {'SUBTITLE_RESULTS_ROOT': str(results)}), \
                 patch.object(pipeline, 'Workspace', side_effect=make_workspace), \
                 patch.object(pipeline, 'verify', return_value={}), \
                 patch.object(pipeline, 'select_compute_mode', return_value={
                     'requested': 'cpu', 'selected': 'cpu', 'ocr': 'CPU'}), \
                 patch.object(pipeline, 'run_process', side_effect=fake_run_process), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = pipeline.main()

            self.assertEqual(video.read_bytes(), source_bytes)
            self.assertEqual((video.stat().st_size, video.stat().st_mtime_ns),
                             (original_stat.st_size, original_stat.st_mtime_ns))
            self.assertEqual(native_inputs, [video.resolve()])
            self.assertEqual(len(workspaces), 1)
            self.assertFalse(workspaces[0].path.exists())
            result_dir = next(path for path in results.iterdir() if path.is_dir())
            self.assertTrue((result_dir / 'report.json').is_file())
            self.assertTrue((result_dir / 'summary.txt').is_file())
            self.assertTrue((result_dir / 'subhooper-result-bundle.zip').is_file())
            report = json.loads((result_dir / 'report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['temp_cleaned'])
            return code, report

    def test_native_engine_reads_original_and_success_cleans_only_temp(self):
        code, report = self.run_pipeline()
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'COMPLETE')

    def test_native_failure_cleans_temp_and_preserves_failure_report(self):
        code, report = self.run_pipeline('error')
        self.assertEqual(code, 1)
        self.assertEqual(report['error'], 'simulated OCR failure')

    def test_cancellation_cleans_temp_and_preserves_results(self):
        code, report = self.run_pipeline('cancel')
        self.assertEqual(code, 130)
        self.assertEqual(report['error'], 'Processing was cancelled.')


if __name__ == '__main__':
    unittest.main()
