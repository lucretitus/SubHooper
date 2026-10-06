import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from native_ocr import (DirectOnnxOCR, _cuda_primary_all,
                        _has_subtitle_shaped_change,
                        _text_confidence_threshold, _timestamp, extract_video)


class FakeCapture:
    def __init__(self, frames, fps=2, positions_ms=None):
        self.frames = frames
        self.position = 0
        self.fps = fps
        self.positions_ms = positions_ms
        self.released = False
        self.grab_count = 0
        self.retrieve_count = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 1:
            return self.fps
        if prop == 2:
            return len(self.frames)
        if prop == 3:
            if self.positions_ms is not None:
                return self.positions_ms[self.position - 1]
            return (self.position - 1) * 1000 / self.fps
        return 0

    def grab(self):
        if self.position >= len(self.frames):
            return False
        self.position += 1
        self.grab_count += 1
        return True

    def retrieve(self):
        self.retrieve_count += 1
        return True, self.frames[self.position - 1]

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_POS_MSEC = 3

    def __init__(self, frames, fps=2, positions_ms=None):
        self.capture = FakeCapture(frames, fps, positions_ms)
        self.thread_count = None

    def setNumThreads(self, count):
        self.thread_count = count

    def VideoCapture(self, _path):
        return self.capture


class NativeOcrTests(unittest.TestCase):
    def test_mixed_direct_sessions_keep_detector_and_recognizer_on_cuda(self):
        class CV2:
            pass

        class Session:
            def __init__(self, _path, sess_options=None, providers=None):
                self.providers = providers
                self.fallback_disabled = False

            def disable_fallback(self):
                self.fallback_disabled = True

            def get_providers(self):
                return self.providers

            def get_inputs(self):
                return [SimpleNamespace(name='input', shape=['batch', 3, 48, 'width'])]

        ort = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(), InferenceSession=Session)
        with tempfile.TemporaryDirectory() as directory:
            for name in ('PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx'):
                (Path(directory) / name).touch()
            (Path(directory) / 'ppocrv6_dict.txt').write_text('A\n', encoding='utf-8')
            with patch.dict(sys.modules, {'cv2': CV2}):
                engine = DirectOnnxOCR(directory, 'mixed', ort)
        self.assertEqual(engine.detector.get_providers()[0], 'CUDAExecutionProvider')
        self.assertEqual(engine.recognizer.get_providers()[0], 'CUDAExecutionProvider')
        self.assertTrue(engine.detector.fallback_disabled)
        self.assertTrue(engine.recognizer.fallback_disabled)
        self.assertEqual(engine.recognizer_batch_size, 8)
        self.assertEqual(engine.recognizer_provider_calibration_status, 'disabled')

    def test_cuda_batch_failure_retries_singles_without_cpu_session_switch(self):
        class CV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def resize(image, size):
                return np.full((size[1], size[0], 3), image[0, 0, 0], dtype=np.uint8)

            @staticmethod
            def cvtColor(image, _code):
                return image

        class Session:
            def __init__(self, path, sess_options=None, providers=None):
                self.model = Path(path).name
                self.providers = [entry[0] if isinstance(entry, tuple) else entry
                                  for entry in providers]
                self.fallback_disabled = False

            def disable_fallback(self):
                self.fallback_disabled = True

            def get_providers(self):
                return self.providers

            def get_inputs(self):
                return [SimpleNamespace(name='input', shape=['batch', 3, 48, 'width'])]

            def run(self, _outputs, feed):
                tensor = feed['input']
                if len(tensor) > 1:
                    if not self.fallback_disabled:
                        self.providers = ['CPUExecutionProvider']
                    raise RuntimeError('CUDNN_FE HEURISTIC_QUERY_FAILED')
                symbol = 1 if tensor[0, 0, 0, 0] < 0 else 2
                scores = np.zeros((1, 1, 3), dtype=np.float32)
                scores[0, 0, symbol] = 1
                return [scores]

        ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(),
                              InferenceSession=Session)
        with tempfile.TemporaryDirectory() as directory:
            for name in ('PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx'):
                (Path(directory) / name).touch()
            (Path(directory) / 'ppocrv6_dict.txt').write_text('A\nB\n', encoding='utf-8')
            with patch.dict(sys.modules, {'cv2': CV2}):
                engine = DirectOnnxOCR(directory, 'mixed', ort)
                crops = [np.zeros((48, 32, 3), dtype=np.uint8),
                         np.full((48, 32, 3), 255, dtype=np.uint8)]
                result = engine._recognize_many(crops)
        self.assertEqual(result, [('A', 1.0), ('B', 1.0)])
        self.assertEqual(engine.recognizer_batch_size, 1)
        self.assertEqual(engine.stage_timings['recognizer_batch_fallbacks'], 1)
        self.assertEqual(engine.recognizer.get_providers()[0], 'CUDAExecutionProvider')

    def test_auto_selects_cpu_recognizer_after_matching_live_crops(self):
        class CV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def resize(image, size):
                return np.full((size[1], size[0], 3), image[0, 0, 0], dtype=np.uint8)

            @staticmethod
            def cvtColor(image, _code):
                return image

        class Session:
            def __init__(self, path, sess_options=None, providers=None):
                self.model = Path(path).name
                self.providers = providers

            def disable_fallback(self):
                pass

            def get_providers(self):
                return self.providers

            def get_inputs(self):
                return [SimpleNamespace(name='input', shape=['batch', 3, 48, 'width'])]

            def run(self, _outputs, feed):
                tensor = feed['input']
                if self.providers[0] == 'CUDAExecutionProvider':
                    time.sleep(.004)
                rows = np.zeros((len(tensor), 1, 3), dtype=np.float32)
                rows[:, :, 1] = 1
                return [rows]

        ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(),
                              InferenceSession=Session)
        with tempfile.TemporaryDirectory() as directory:
            for name in ('PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx'):
                (Path(directory) / name).touch()
            (Path(directory) / 'ppocrv6_dict.txt').write_text('A\nB\n', encoding='utf-8')
            with patch.dict(sys.modules, {'cv2': CV2}):
                engine = DirectOnnxOCR(directory, 'auto', ort)
            self.assertEqual(engine.detector.get_providers()[0], 'CUDAExecutionProvider')
            crops = [np.zeros((48, 32, 3), dtype=np.uint8) for _ in range(16)]
            self.assertEqual([text for text, _ in engine._recognize_many(crops)], ['A'] * 16)
            self.assertEqual(engine.recognizer.get_providers()[0], 'CPUExecutionProvider')
            self.assertEqual(engine.recognizer_provider_calibration_status, 'cpu_selected')
            self.assertEqual(engine.recognizer_batch_size, 1)

    def test_auto_calibration_mismatch_keeps_cuda_recognizer(self):
        engine = DirectOnnxOCR.__new__(DirectOnnxOCR)
        engine.recognizer = 'gpu'
        engine._cpu_recognizer_candidate = 'cpu'
        engine.recognizer_batch_size = 8
        engine.recognizer_provider_calibration_status = 'pending'
        engine.recognizer_provider_calibration_seconds = 0
        engine._recognize_many_impl = lambda crops, **kwargs: [('A' if engine.recognizer == 'gpu' else 'B', .99)] * len(crops)
        self.assertEqual(engine._recognize_many([object()] * 8), [('A', .99)] * 8)
        self.assertEqual(engine.recognizer_provider_calibration_status, 'output_mismatch')
        self.assertEqual(engine.recognizer, 'gpu')
        self.assertEqual(engine.recognizer_batch_size, 8)

    def test_direct_recognizer_ctc_removes_repeated_symbols_and_blank(self):
        class CV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def resize(_image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def cvtColor(image, _code):
                return image

        class Session:
            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name='input')]

            @staticmethod
            def run(_outputs, feed):
                assert feed['input'].shape == (1, 3, 48, 32)
                return [np.array([[[0, .99, 0], [0, .98, 0],
                                   [.99, 0, 0], [0, 0, .99]]], dtype=np.float32)]

        ocr = DirectOnnxOCR.__new__(DirectOnnxOCR)
        ocr.cv2, ocr.np = CV2, np
        ocr.stage_timings = {
            'recognizer_preprocess_seconds': 0.0,
            'recognizer_forward_seconds': 0.0,
            'recognizer_postprocess_seconds': 0.0,
            'recognizer_runs': 0,
        }
        ocr.recognizer = Session()
        ocr.characters = ['', 'A', 'B']
        text, confidence = ocr._recognize(np.zeros((48, 32, 3), dtype=np.uint8))
        self.assertEqual(text, 'AB')
        self.assertGreater(confidence, .95)
        self.assertEqual(ocr.stage_timings['recognizer_runs'], 1)
        self.assertGreaterEqual(ocr.stage_timings['recognizer_preprocess_seconds'], 0)
        self.assertGreaterEqual(ocr.stage_timings['recognizer_forward_seconds'], 0)
        self.assertGreaterEqual(ocr.stage_timings['recognizer_postprocess_seconds'], 0)

    def test_ctc_confidence_gather_preserves_ties_and_repeated_short_text(self):
        ocr = DirectOnnxOCR.__new__(DirectOnnxOCR)
        ocr.np = np
        ocr.characters = ['', 'A', 'B']
        scores = np.array([
            [.10, .45, .45],  # A and B tie; argmax chooses A.
            [.10, .80, .10],  # Repeated A is collapsed.
            [.90, .05, .05],  # Blank separates the next character.
            [.10, .10, .80],
        ], dtype=np.float32)
        text, confidence = ocr._decode_recognition(scores)
        self.assertEqual(text, 'AB')
        self.assertAlmostEqual(confidence, (.45 + .80) / 2, places=6)

    def test_direct_recognizer_converts_logits_to_probability_confidence(self):
        class CV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def resize(_image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def cvtColor(image, _code):
                return image

        class Session:
            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name='input')]

            @staticmethod
            def run(_outputs, _feed):
                return [np.array([[[0, 2, 0]]], dtype=np.float32)]

        ocr = DirectOnnxOCR.__new__(DirectOnnxOCR)
        ocr.cv2, ocr.np = CV2, np
        ocr.stage_timings = {
            'recognizer_preprocess_seconds': 0.0,
            'recognizer_forward_seconds': 0.0,
            'recognizer_postprocess_seconds': 0.0,
            'recognizer_runs': 0,
        }
        ocr.recognizer = Session()
        ocr.characters = ['', '中', '文']
        text, confidence = ocr._recognize(np.zeros((48, 32, 3), dtype=np.uint8))
        self.assertEqual(text, '中')
        self.assertAlmostEqual(confidence, np.exp(2) / (np.exp(2) + 2), places=5)

    def test_cuda_recognizer_batches_similar_widths_and_restores_crop_order(self):
        class CV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def resize(image, size):
                return np.full((size[1], size[0], 3), image[0, 0, 0], dtype=np.uint8)

            @staticmethod
            def cvtColor(image, _code):
                return image

        class Session:
            def __init__(self):
                self.shapes = []
                self.inputs = []

            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name='input')]

            def run(self, _outputs, feed):
                tensor = feed['input']
                self.shapes.append(tensor.shape)
                self.inputs.append(tensor.copy())
                rows = []
                for image in tensor:
                    symbol = 1 if image[0, 0, 0] < 0 else 2
                    row = np.zeros((1, 3), dtype=np.float32)
                    row[0, symbol] = 1
                    rows.append(row)
                return [np.stack(rows)]

        ocr = DirectOnnxOCR.__new__(DirectOnnxOCR)
        ocr.cv2, ocr.np = CV2, np
        ocr.stage_timings = {
            'recognizer_preprocess_seconds': 0.0,
            'recognizer_forward_seconds': 0.0,
            'recognizer_postprocess_seconds': 0.0,
            'recognizer_runs': 0,
            'recognizer_images': 0,
            'recognizer_batch_histogram': {},
        }
        ocr.recognizer = Session()
        ocr.recognizer_batch_size = 8
        ocr.characters = ['', 'A', 'B']
        crops = [np.full((48, 160, 3), 255, dtype=np.uint8),
                 np.zeros((48, 32, 3), dtype=np.uint8),
                 np.full((48, 36, 3), 255, dtype=np.uint8)]
        self.assertEqual([text for text, _ in ocr._recognize_many(crops)], ['B', 'A', 'B'])
        self.assertEqual(ocr.recognizer.shapes, [(2, 3, 48, 40), (1, 3, 48, 160)])
        np.testing.assert_array_equal(ocr.recognizer.inputs[0][0, :, :, :32], -1)
        np.testing.assert_array_equal(ocr.recognizer.inputs[0][0, :, :, 32:], -1)
        np.testing.assert_array_equal(ocr.recognizer.inputs[0][1], 1)
        self.assertEqual(ocr.stage_timings['recognizer_images'], 3)
        self.assertEqual(ocr.stage_timings['recognizer_runs'], 2)
        self.assertEqual(ocr.stage_timings['recognizer_batch_histogram'], {'2': 1, '1': 1})

    def test_recognize_frames_preserves_frame_and_line_order(self):
        class BatchedDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}
                self.recognizer_batch_size = 8
                self.received = None

            def _detect_crops(self, frame):
                marker = int(frame[0, 0])
                if marker == 1:
                    return 100, [(60, 20, np.array([1])),
                                 (10, 5, np.array([2]))]
                return 100, [(40, 7, np.array([3]))]

            def _recognize_many(self, crops):
                self.received = [int(crop[0]) for crop in crops]
                return [('later', .99), ('first', .99), ('next-frame', .99)]

        ocr = BatchedDirect()
        frames = [np.full((1, 1), marker, dtype=np.uint8) for marker in (1, 2)]
        self.assertEqual(ocr.recognize_frames(frames),
                         [['first', 'later'], ['next-frame']])
        self.assertEqual(ocr.received, [1, 2, 3])
        self.assertEqual(ocr.stage_timings['recognizer_frame_batches'], 1)
        self.assertEqual(ocr.stage_timings['recognizer_frames'], 2)
        self.assertEqual(ocr.stage_timings['recognizer_cross_frame_batches'], 1)

    def test_recognize_detected_frames_preserves_cross_frame_order(self):
        class DetectedDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}
                self.recognizer_batch_size = 8
                self.received = None

            def _recognize_many(self, crops):
                self.received = [int(crop[0]) for crop in crops]
                return [(f'cue-{int(crop[0])}', .99) for crop in crops]

        ocr = DetectedDirect()
        frame_data = [
            (100, [(60, 20, np.array([1])), (10, 5, np.array([2]))]),
            (100, [(40, 7, np.array([3]))]),
        ]
        self.assertEqual(ocr.recognize_detected_frames(frame_data),
                         [['cue-2', 'cue-1'], ['cue-3']])
        self.assertEqual(ocr.received, [1, 2, 3])
        self.assertEqual(ocr.stage_timings['recognizer_frame_batches'], 1)
        self.assertEqual(ocr.stage_timings['recognizer_frames'], 2)
        self.assertEqual(ocr.stage_timings['recognizer_cross_frame_batches'], 1)

    def test_cuda_extraction_flushes_cross_frame_recognition_in_order(self):
        class BatchedDirect(DirectOnnxOCR):
            def __init__(self):
                self.detector = object()
                self.recognizer = object()
                self.stage_timings = {}
                self.recognizer_batch_size = 8
                self.groups = []

            def __call__(self, _crop):
                raise AssertionError('frames should flush as a recognition batch')

            def recognize_frames(self, frames):
                self.groups.append(len(frames))
                names = ('Alpha', 'Beta', 'Gamma', 'Delta')
                return [[names[index]] for index in range(len(frames))]

        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(8)]
        cv2 = FakeCV2(frames, fps=2)
        ocr = BatchedDirect()
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video('input.mp4', temp, (0, 1, 0, 1),
                                    compute='cuda', cv2_module=cv2, ocr=ocr)
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertEqual(ocr.groups, [4])
        self.assertEqual(metrics['ocr_count'], 4)
        self.assertEqual(metrics['recognizer_frame_batch_size'], 4)
        self.assertIn('Alpha', srt)
        self.assertIn('Beta', srt)
        self.assertIn('Gamma', srt)
        self.assertIn('Delta', srt)

    def test_cuda_final_batch_honors_cancel_during_recognition(self):
        cancelled = [False]

        class BatchedDirect(DirectOnnxOCR):
            def __init__(self):
                self.detector = object()
                self.recognizer = object()
                self.recognizer_batch_size = 8
                self.stage_timings = {}

            def recognize_frames(self, frames):
                cancelled[0] = True
                return [['Subtitle'] for _ in frames]

        cv2 = FakeCV2([np.zeros((8, 8), dtype=np.uint8)], fps=2)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(InterruptedError):
                extract_video('input.mp4', temp, (0, 1, 0, 1), compute='cuda',
                              cv2_module=cv2, ocr=BatchedDirect(),
                              cancel_check=lambda: cancelled[0])
            self.assertFalse((Path(temp) / 'result.srt').exists())

    def test_recognizer_retries_single_crops_when_batch_unsupported(self):
        class CV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def resize(_image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def cvtColor(image, _code):
                return image

        class Session:
            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name='input')]

            @staticmethod
            def get_providers():
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']

            @staticmethod
            def run(_outputs, feed):
                if feed['input'].shape[0] > 1:
                    raise ValueError('export accepts only batch=1')
                return [np.array([[[0, 1]]], dtype=np.float32)]

        ocr = DirectOnnxOCR.__new__(DirectOnnxOCR)
        ocr.cv2, ocr.np = CV2, np
        ocr.stage_timings = dict.fromkeys((
            'recognizer_preprocess_seconds', 'recognizer_forward_seconds',
            'recognizer_postprocess_seconds', 'recognizer_runs',
            'recognizer_images', 'recognizer_batch_fallbacks'), 0)
        ocr.recognizer = Session()
        ocr.recognizer_batch_size = 8
        ocr.characters = ['', 'A']
        crops = [np.zeros((48, 32, 3), dtype=np.uint8)] * 2
        self.assertEqual(ocr._recognize_many(crops), [('A', 1.0)] * 2)
        self.assertEqual(ocr.recognizer_batch_size, 1)
        self.assertEqual(ocr.stage_timings['recognizer_batch_fallbacks'], 1)

    def test_recognizer_calibration_uses_baseline_when_outputs_differ(self):
        class CalibratedDirect(DirectOnnxOCR):
            def __init__(self):
                self.recognizer_batch_size = 8
                self.recognizer_calibration_status = 'pending'
                self.recognizer_calibration_seconds = 0.0
                self.calls = []

            def _recognize_many_impl(self, crops, batch_size=None, calibration=False):
                self.calls.append((batch_size, calibration))
                if batch_size == 16:
                    return [('candidate text', .99) for _ in crops]
                return [('baseline text', .99) for _ in crops]

        ocr = CalibratedDirect()
        crops = [np.zeros((48, 32, 3), dtype=np.uint8) for _ in range(16)]
        self.assertEqual(ocr._recognize_many(crops), [('baseline text', .99)] * 16)
        self.assertEqual(ocr.recognizer_batch_size, 8)
        self.assertEqual(ocr.recognizer_calibration_status, 'output_mismatch')
        self.assertEqual(ocr.calls, [(8, False), (16, True)])

    def test_recognizer_calibration_selects_faster_equal_candidate(self):
        class CalibratedDirect(DirectOnnxOCR):
            def __init__(self):
                self.recognizer_batch_size = 8
                self.recognizer_calibration_status = 'pending'
                self.recognizer_calibration_seconds = 0.0

            def _recognize_many_impl(self, crops, batch_size=None, calibration=False):
                return [('same output', .99) for _ in crops]

        ocr = CalibratedDirect()
        crops = [np.zeros((48, 32, 3), dtype=np.uint8) for _ in range(16)]
        # perf_counter marks wrapper start, baseline start/end, candidate
        # start/end, and total end. The candidate pass measures faster.
        ticks = iter((0.0, 0.1, 0.6, 0.7, 0.9, 1.0))
        with patch('native_ocr.time.perf_counter', side_effect=lambda: next(ticks)):
            results = ocr._recognize_many(crops)
        self.assertEqual(results, [('same output', .99)] * 16)
        self.assertEqual(ocr.recognizer_batch_size, 16)
        self.assertEqual(ocr.recognizer_calibration_status, 'larger_batch_selected')

    def test_recognizer_calibration_waits_for_bounded_representative_group(self):
        class CalibratedDirect(DirectOnnxOCR):
            def __init__(self):
                self.recognizer_batch_size = 8
                self.recognizer_calibration_status = 'pending'
                self.calls = 0

            def _recognize_many_impl(self, crops, batch_size=None, calibration=False):
                self.calls += 1
                return [('text', .99) for _ in crops]

        ocr = CalibratedDirect()
        self.assertEqual(ocr._recognize_many([np.zeros((2, 2))]), [('text', .99)])
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(ocr.recognizer_calibration_status, 'pending')

    def test_short_subtitle_confidence_threshold_preserves_cjk_and_digits(self):
        self.assertEqual(_text_confidence_threshold('中'), .78)
        self.assertEqual(_text_confidence_threshold('12'), .78)
        self.assertEqual(_text_confidence_threshold('hello'), .45)

    def test_timestamp_keeps_milliseconds_and_hours(self):
        self.assertEqual(_timestamp(3661.25), '01:01:01,250')

    def test_cuda_validation_requires_used_primary_sessions(self):
        providers = {
            'detector': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'recognizer': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'classifier': [],
        }
        no_classifier = SimpleNamespace()
        self.assertTrue(_cuda_primary_all(providers, no_classifier, 1))
        self.assertFalse(_cuda_primary_all(providers, no_classifier, 0))
        classifier_enabled = SimpleNamespace(
            cfg=SimpleNamespace(Global=SimpleNamespace(use_cls=True)))
        self.assertFalse(_cuda_primary_all(providers, classifier_enabled, 1))
        providers['classifier'] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.assertTrue(_cuda_primary_all(providers, classifier_enabled, 1))

    def test_samples_changed_region_and_writes_srt_and_metrics(self):
        frames = [np.zeros((8, 10), dtype=np.uint8) for _ in range(6)]
        frames[2:] = [np.full((8, 10), 40 + i, dtype=np.uint8) for i in range(4)]
        cv2 = FakeCV2(frames, fps=4)
        calls = []

        def ocr(crop):
            calls.append(crop.copy())
            return SimpleNamespace(txts=['A subtitle'])

        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video('input.mp4', temp, (0, 1, 0, 1),
                                    cv2_module=cv2, ocr=ocr)
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
            saved_metrics = json.loads((Path(temp) / 'ocr-metrics.json').read_text())

        self.assertEqual(len(calls), 2)  # initial sample and scene-shift boundary
        self.assertEqual(metrics['stable_samples_skipped'], 1)
        self.assertEqual(metrics['change_gate_checks'], 2)
        self.assertEqual(metrics['change_gate_ocr_triggers'], 1)
        self.assertEqual(metrics['forced_ocr_checks'], 1)
        self.assertEqual(metrics['forced_only_ocr_calls'], 0)
        self.assertEqual(metrics['ocr_transition_backdates'], 0)
        self.assertEqual(metrics['sample_count'], 3)
        self.assertEqual(cv2.capture.grab_count, 6)
        self.assertEqual(cv2.capture.retrieve_count, 3)
        self.assertEqual(saved_metrics['cue_count'], 1)
        self.assertIn('video_decode_seconds', metrics)
        self.assertIn('video_sampling_seconds', metrics)
        self.assertIn('ocr_total_seconds', metrics)
        self.assertIn('ocr_unattributed_seconds', metrics)
        self.assertEqual(metrics['ocr_stage_seconds']['detector_runs'], 0)
        self.assertEqual(metrics['ocr_stage_seconds']['recognizer_runs'], 0)
        self.assertEqual(metrics['ocr_stage_seconds']['recognizer_images'], 0)
        self.assertEqual(metrics['recognizer_batch_histogram'], {})
        self.assertGreaterEqual(metrics['video_decode_seconds'], 0)
        self.assertGreaterEqual(metrics['video_sampling_seconds'], 0)
        self.assertIn('00:00:00,000 --> 00:00:01,500', srt)
        self.assertIn('A subtitle', srt)
        self.assertTrue(cv2.capture.released)

    def test_cancel_releases_capture(self):
        cv2 = FakeCV2([np.zeros((2, 2), dtype=np.uint8)])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(InterruptedError):
                extract_video('input.mp4', temp, (0, 1, 0, 1),
                              cv2_module=cv2, ocr=lambda _: None,
                              cancel_check=lambda: True)
        self.assertTrue(cv2.capture.released)

    def test_variable_frame_timestamps_are_used(self):
        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(6)]
        frames[2] = np.full((8, 8), 40, dtype=np.uint8)
        cv2 = FakeCV2(frames, fps=4,
                      positions_ms=[0, 250, 600, 900, 2100, 2350])
        texts = iter(('First', 'Second', 'Second'))
        with tempfile.TemporaryDirectory() as temp:
            extract_video('input.mp4', temp, (0, 1, 0, 1), cv2_module=cv2,
                          ocr=lambda _: SimpleNamespace(txts=[next(texts)]))
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertIn('00:00:00,000 --> 00:00:00,600', srt)
        self.assertIn('00:00:00,600 --> 00:00:02,600', srt)

    def test_periodic_ocr_checks_stable_regions_for_disappearance(self):
        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(12)]
        cv2 = FakeCV2(frames, fps=2)
        responses = iter(('Visible subtitle', '', '', '', '', ''))
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video(
                'input.mp4', temp, (0, 1, 0, 1), cv2_module=cv2,
                ocr=lambda _: SimpleNamespace(txts=[next(responses)]))
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertEqual(metrics['ocr_count'], 6)
        self.assertEqual(metrics['forced_ocr_checks'], 6)
        self.assertIn('00:00:00,000 --> 00:00:00,750', srt)
        self.assertEqual(metrics['ocr_transition_backdates'], 1)
        self.assertGreater(metrics['forced_only_ocr_calls'], 0)

    def test_change_gate_accepts_compact_subtitle_band_and_rejects_scene_change(self):
        previous = np.full((100, 300), 80, dtype=np.uint8)
        subtitle = previous.copy()
        subtitle[58:70, 60:240] = 230
        scene_cut = previous.copy()
        scene_cut[:, :150] = 120  # Broad but moderate motion is not text-shaped.
        hard_cut = np.full_like(previous, 220)
        self.assertTrue(_has_subtitle_shaped_change(previous, subtitle, np))
        self.assertFalse(_has_subtitle_shaped_change(previous, scene_cut, np))
        self.assertTrue(_has_subtitle_shaped_change(previous, hard_cut, np))

    def test_change_gate_finds_subtitle_transition_over_moving_background(self):
        gradient = np.tile(np.linspace(30, 150, 300, dtype=np.uint8), (100, 1))
        previous = gradient.copy()
        moving_only = np.roll(previous, 4, axis=1)
        subtitle = moving_only.copy()
        for letter in range(5):
            x = 60 + letter * 24
            subtitle[58:70, x:x + 2] = 240
            subtitle[58:60, x:x + 14] = 240
            subtitle[63:65, x:x + 14] = 240
            subtitle[68:70, x:x + 14] = 240
        self.assertFalse(_has_subtitle_shaped_change(previous, moving_only, np))
        self.assertTrue(_has_subtitle_shaped_change(moving_only, subtitle, np))

    def test_short_glyph_transition_triggers_ocr_for_a_single_sample(self):
        frames = [np.full((120, 320), 80, dtype=np.uint8) for _ in range(6)]
        cue = frames[2].copy()
        x = 155
        cue[90:104, x:x + 2] = 230
        cue[90:92, x:x + 10] = 230
        cue[96:98, x:x + 10] = 230
        cue[102:104, x:x + 10] = 230
        frames[2] = cue
        cv2 = FakeCV2(frames, fps=4)
        responses = iter(('', '1', ''))
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video(
                'input.mp4', temp, (0, 1, 0, 1), cv2_module=cv2,
                ocr=lambda _: SimpleNamespace(txts=[next(responses)]))
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertEqual(metrics['change_gate_ocr_triggers'], 2)
        self.assertIn('00:00:00,500 --> 00:00:01,000', srt)

    def test_unsampled_final_pts_extends_video_duration(self):
        frames = [np.full((8, 8), 80, dtype=np.uint8) for _ in range(8)]
        positions_ms = [0, 100, 250, 500, 750, 1000, 1300, 2400]
        cv2 = FakeCV2(frames, fps=4, positions_ms=positions_ms)
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video(
                'input.mp4', temp, (0, 1, 0, 1), cv2_module=cv2,
                ocr=lambda _: SimpleNamespace(txts=['Subtitle']))
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertEqual(metrics['last_decoded_timestamp_seconds'], 2.4)
        self.assertEqual(metrics['duration_seconds'], 2.65)
        self.assertIn('--> 00:00:02,650', srt)

    def test_sampled_pts_is_read_after_retrieve(self):
        class RetrievePositionCapture(FakeCapture):
            def __init__(self, frames, fps, positions_ms):
                super().__init__(frames, fps, positions_ms)
                self.pts = 0

            def get(self, prop):
                if prop == FakeCV2.CAP_PROP_POS_MSEC:
                    return self.pts
                return super().get(prop)

            def retrieve(self):
                ok, frame = super().retrieve()
                self.pts = self.positions_ms[self.position - 1]
                return ok, frame

        class RetrievePositionCV2(FakeCV2):
            def __init__(self, frames, fps, positions_ms):
                self.capture = RetrievePositionCapture(frames, fps, positions_ms)

        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(6)]
        frames[2] = np.full((8, 8), 40, dtype=np.uint8)
        positions_ms = [0, 250, 600, 900, 2100, 2350]
        cv2 = RetrievePositionCV2(frames, fps=4, positions_ms=positions_ms)
        texts = iter(('First', 'Second', 'Second'))
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video(
                'input.mp4', temp, (0, 1, 0, 1), cv2_module=cv2,
                ocr=lambda _: SimpleNamespace(txts=[next(texts)]))
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertIn('00:00:00,000 --> 00:00:00,600', srt)
        self.assertEqual(metrics['last_decoded_timestamp_seconds'], 2.1)
        self.assertEqual(metrics['duration_seconds'], 2.35)

    def test_cuda_direct_path_prefetches_in_order_with_bounded_queue(self):
        class OverlapCapture(FakeCapture):
            def __init__(self, frames, fps, positions_ms):
                super().__init__(frames, fps, positions_ms)
                self.buffer = np.zeros_like(frames[0])
                self.ocr_started = None

            def retrieve(self):
                if self.position >= 2 and not self.ocr_started.wait(timeout=1):
                    raise RuntimeError('decoder did not overlap active OCR')
                self.retrieve_count += 1
                self.buffer[...] = self.frames[self.position - 1]
                return True, self.buffer

        class OverlapCV2(FakeCV2):
            def __init__(self, frames, fps, positions_ms):
                self.capture = OverlapCapture(frames, fps, positions_ms)

        class SlowDirect(DirectOnnxOCR):
            def __init__(self):
                self.seen = []
                self.ocr_started = threading.Event()

            def __call__(self, crop):
                self.ocr_started.set()
                time.sleep(0.02)
                value = int(crop[0, 0])
                self.seen.append(value)
                return [f'subtitle-{value}']

        frames = [np.full((8, 8), value, dtype=np.uint8)
                  for value in (0, 60, 120)]
        ocr = SlowDirect()
        cv2 = OverlapCV2(frames, fps=2, positions_ms=[0, 800, 1500])
        cv2.capture.ocr_started = ocr.ocr_started
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video(
                'input.mp4', temp, (0, 1, 0, 1), compute='cuda',
                cv2_module=cv2, ocr=ocr)
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertEqual(ocr.seen, [0, 60, 120])
        self.assertIn('00:00:00,000 --> 00:00:00,800', srt)
        self.assertIn('00:00:00,800 --> 00:00:01,500', srt)
        self.assertTrue(metrics['video_prefetch_enabled'])
        self.assertEqual(metrics['video_prefetch_queue_capacity'], 3)
        self.assertGreater(metrics['video_prefetch_max_depth'], 0)
        self.assertLessEqual(metrics['video_prefetch_max_depth'], 3)
        self.assertTrue(cv2.capture.released)

    def test_cpu_direct_prefetch_preserves_srt_and_releases_capture(self):
        class Direct(DirectOnnxOCR):
            def __init__(self):
                pass

            def __call__(self, crop):
                return [f'text-{int(crop[0, 0])}']

        class Sequential:
            def __call__(self, crop):
                return [f'text-{int(crop[0, 0])}']

        frames = [np.full((8, 8), value, dtype=np.uint8)
                  for value in (0, 60, 120)]
        with tempfile.TemporaryDirectory() as temp:
            cv2_cpu = FakeCV2(frames, fps=2, positions_ms=[0, 800, 1500])
            cpu = extract_video('input.mp4', Path(temp) / 'cpu',
                                (0, 1, 0, 1), compute='cpu',
                                cv2_module=cv2_cpu, ocr=Direct())
            cv2_sequence = FakeCV2(frames, fps=2, positions_ms=[0, 800, 1500])
            sequential = extract_video('input.mp4', Path(temp) / 'sequence',
                                       (0, 1, 0, 1), compute='cpu',
                                       cv2_module=cv2_sequence, ocr=Sequential())
            self.assertEqual((Path(temp) / 'cpu' / 'result.srt').read_bytes(),
                             (Path(temp) / 'sequence' / 'result.srt').read_bytes())
        self.assertTrue(cpu['video_prefetch_enabled'])
        self.assertFalse(sequential['video_prefetch_enabled'])
        self.assertTrue(cv2_cpu.capture.released)

    def test_directml_sessions_require_primary_provider_and_sequential_options(self):
        class Options:
            enable_mem_pattern = True
            execution_mode = None

        class Session:
            def __init__(self, _path, sess_options, providers):
                self.options = sess_options
                self.providers = providers

            def get_providers(self):
                return [value[0] if isinstance(value, tuple) else value for value in self.providers]

            def disable_fallback(self):
                self.fallback_disabled = True

            def get_inputs(self):
                return [SimpleNamespace(shape=['N', 3, 48, 320])]

        ort = SimpleNamespace(SessionOptions=Options, InferenceSession=Session,
                              ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0))
        with tempfile.TemporaryDirectory() as temp:
            for name in ('PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx',
                         'ppocrv6_dict.txt'):
                (Path(temp) / name).write_bytes(b'example')
            with patch.dict(sys.modules, {'cv2': SimpleNamespace()}):
                direct = DirectOnnxOCR(temp, 'dml', ort)
        self.assertEqual(direct.detector.get_providers()[0], 'DmlExecutionProvider')
        self.assertFalse(direct.detector.options.enable_mem_pattern)
        self.assertEqual(direct.detector.options.execution_mode, 0)
        self.assertTrue(direct.detector.fallback_disabled)
        self.assertTrue(direct.recognizer.fallback_disabled)
        self.assertEqual(direct.detector.providers[0][1], {
            'performance_preference': 'high_performance', 'device_filter': 'gpu'})

    def test_cuda_prefetch_propagates_decoder_error_and_releases_capture(self):
        class BrokenCapture(FakeCapture):
            def grab(self):
                raise ValueError('decoder failed')

        class BrokenCV2(FakeCV2):
            def __init__(self):
                self.capture = BrokenCapture([])

        class Direct(DirectOnnxOCR):
            def __init__(self):
                pass

            def __call__(self, _crop):
                return []

        cv2 = BrokenCV2()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'decoder failed'):
                extract_video('input.mp4', temp, (0, 1, 0, 1), compute='cuda',
                              cv2_module=cv2, ocr=Direct())
        self.assertTrue(cv2.capture.released)

    def test_cuda_prefetch_cancel_stops_worker_and_releases_capture(self):
        class SlowDirect(DirectOnnxOCR):
            def __init__(self):
                pass

            def __call__(self, _crop):
                time.sleep(0.02)
                return ['subtitle']

        frames = [np.full((8, 8), value, dtype=np.uint8)
                  for value in (0, 60, 120, 180, 240)]
        cv2 = FakeCV2(frames, fps=2)
        polls = [0]

        def cancel():
            polls[0] += 1
            return polls[0] >= 3

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(InterruptedError):
                extract_video('input.mp4', temp, (0, 1, 0, 1), compute='cuda',
                              cv2_module=cv2, ocr=SlowDirect(),
                              cancel_check=cancel)
        self.assertTrue(cv2.capture.released)
        self.assertFalse(any(thread.name == 'subtitle-video-prefetch'
                             for thread in threading.enumerate()))

    def test_mixed_detector_batches_cuda_recognition_and_keeps_order(self):
        worker_names = []
        recognizer_names = []

        class MixedDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}
                class Session:
                    @staticmethod
                    def get_providers():
                        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.detector = self.recognizer = Session()

            def _detect_crops(self, frame):
                worker_names.append(threading.current_thread().name)
                marker = len(worker_names)
                time.sleep(.003)
                crop = np.full_like(frame, marker)
                return frame.shape[0], [(0, 0, crop)]

            def _recognize_many(self, crops):
                recognizer_names.append(threading.current_thread().name)
                return [(f'text-{int(crop[0, 0])}', .99) for crop in crops]

        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(7)]
        cv2 = FakeCV2(frames, fps=2)
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video('input.mp4', temp, (0, 1, 0, 1),
                                    compute='mixed', cv2_module=cv2,
                                    ocr=MixedDirect())
            srt = (Path(temp) / 'result.srt').read_text(encoding='utf-8')
        self.assertEqual(len(worker_names), 4)
        self.assertTrue(all(name.startswith('subtitle-mixed-detector')
                            for name in worker_names))
        self.assertTrue(all(name == threading.current_thread().name
                            for name in recognizer_names))
        self.assertEqual(metrics['mixed_detector_overlap_enabled'], True)
        self.assertEqual(metrics['mixed_detector_pending_capacity'], 4)
        self.assertEqual(metrics['mixed_detector_submitted'], 4)
        self.assertEqual(metrics['mixed_detector_max_pending'], 4)
        self.assertTrue(metrics['mixed_cpu_decode_gate_overlap_enabled'])
        self.assertTrue(metrics['mixed_gpu_detector_worker_enabled'])
        self.assertEqual(metrics['model_session_providers']['detector'][0],
                         'CUDAExecutionProvider')
        self.assertEqual(metrics['model_session_providers']['recognizer'][0],
                         'CUDAExecutionProvider')
        self.assertEqual(metrics['compute_effective'], 'CUDA_CPU_OVERLAP')
        self.assertTrue(metrics['cuda_primary_all'])
        self.assertGreater(metrics['mixed_detector_elapsed_seconds'], 0)
        self.assertGreaterEqual(metrics['mixed_detector_wait_seconds'], 0)
        self.assertGreaterEqual(metrics['opencv_threads'], 2)
        self.assertLessEqual(metrics['opencv_threads'], 4)
        self.assertEqual(cv2.thread_count, metrics['opencv_threads'])
        self.assertEqual(metrics['ocr_stage_seconds']['recognizer_frames'],
                         metrics['ocr_count'])
        self.assertEqual(metrics['ocr_stage_seconds']['recognizer_frame_batches'], 1)
        self.assertEqual(metrics['ocr_stage_seconds']['recognizer_cross_frame_batches'], 1)
        self.assertIn('text-1', srt)
        self.assertIn('text-2', srt)
        self.assertLess(srt.index('text-1'), srt.index('text-2'))
        self.assertTrue(cv2.capture.released)

    def test_mixed_detector_pending_work_is_bounded_and_cancelled_cleanly(self):
        class MixedDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}
                self.detected = 0

            def _detect_crops(self, frame):
                self.detected += 1
                time.sleep(.005)
                return frame.shape[0], []

            def _recognize_many(self, _crops):
                return []

        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(6)]
        cv2 = FakeCV2(frames, fps=2)
        ocr = MixedDirect()
        with tempfile.TemporaryDirectory() as temp:
            metrics = extract_video('input.mp4', temp, (0, 1, 0, 1),
                                    compute='mixed', cv2_module=cv2, ocr=ocr)
        self.assertEqual(metrics['mixed_detector_submitted'], 3)
        self.assertLessEqual(metrics['mixed_detector_max_pending'], 4)
        self.assertLessEqual(ocr.detected, 3)
        self.assertTrue(cv2.capture.released)
        self.assertFalse(any(thread.name.startswith('subtitle-mixed-detector')
                             for thread in threading.enumerate()))

    def test_mixed_and_cpu_direct_paths_keep_srt_parity(self):
        class SameDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}

            def _detect_crops(self, frame):
                return frame.shape[0], [(0, 0, frame.copy())]

            def _recognize_many(self, crops):
                return [(f'text-{int(crop[0, 0])}', .99) for crop in crops]

        frames = [np.full((8, 8), value, dtype=np.uint8)
                  for value in (0, 0, 60, 60, 120, 120)]
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for mode in ('cpu', 'mixed', 'dml'):
                output = Path(temp) / mode
                extract_video(
                    'input.mp4', output, (0, 1, 0, 1), compute=mode,
                    cv2_module=FakeCV2(frames, fps=2), ocr=SameDirect())
                outputs.append((output / 'result.srt').read_bytes())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_mixed_detector_error_propagates_and_cancellation_shuts_down(self):
        class BrokenDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}

            def _detect_crops(self, _frame):
                raise ValueError('mixed detector failed')

            def _recognize_many(self, _crops):
                return []

        cv2 = FakeCV2([np.zeros((8, 8), dtype=np.uint8)], fps=2)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'mixed detector failed'):
                extract_video('input.mp4', temp, (0, 1, 0, 1), compute='mixed',
                              cv2_module=cv2, ocr=BrokenDirect())
        self.assertTrue(cv2.capture.released)

        started = threading.Event()

        class SlowDirect(DirectOnnxOCR):
            def __init__(self):
                self.stage_timings = {}

            def _detect_crops(self, _frame):
                started.set()
                time.sleep(.03)
                return 8, []

            def _recognize_many(self, _crops):
                return []

        cancel_cv2 = FakeCV2([np.zeros((8, 8), dtype=np.uint8)] * 4, fps=2)
        checks = [0]

        def cancel_after_submit():
            checks[0] += 1
            return checks[0] >= 2

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(InterruptedError):
                extract_video('input.mp4', temp, (0, 1, 0, 1), compute='mixed',
                              cv2_module=cancel_cv2, ocr=SlowDirect(),
                              cancel_check=cancel_after_submit)
        self.assertTrue(started.is_set())
        self.assertTrue(cancel_cv2.capture.released)
        self.assertFalse(any(thread.name.startswith('subtitle-mixed-detector')
                             for thread in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
