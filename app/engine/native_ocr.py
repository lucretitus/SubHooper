"""Fast first-stage subtitle OCR directly from sampled video frames."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import queue
import threading
import time
import unicodedata
from pathlib import Path

SAMPLE_FPS = 2.0
# Sample cheaply at a bounded resolution so ordinary camera/background motion
# does not cause a full detector + recognizer pass for every frame.
CHANGE_GATE_WIDTH = 320
CHANGE_GATE_HEIGHT = 160
CHANGE_GATE_PIXEL_THRESHOLD = 32
FORCED_OCR_INTERVAL = 1.0
CUDA_RECOGNIZER_FRAME_BATCH = 4
# Fall back to frame-local recognition if a crowded scene would allocate too
# many normalized float crop tensors at once.
CUDA_RECOGNIZER_CROP_LIMIT = 32
CUDA_RECOGNIZER_CALIBRATION_CROPS = 16
CUDA_RECOGNIZER_MAX_BATCH_SIZE = 16
PROVIDER_CALIBRATION_MIN_CROPS = 8


def _compact(text: str) -> str:
    folded = unicodedata.normalize('NFKC', text.casefold())
    return ''.join(char for char in folded if char.isalnum())


def _timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def _srt(cues: list[tuple[float, float, str]]) -> str:
    return ''.join(
        f'{index}\n{_timestamp(start)} --> {_timestamp(max(start + 0.1, end))}\n{text}\n\n'
        for index, (start, end, text) in enumerate(cues, 1)
    )


def _recognized_text(result) -> str:
    texts = getattr(result, 'txts', None)
    if texts is None and isinstance(result, (tuple, list)):
        texts = result
    if not texts:
        return ''
    if isinstance(texts, str):
        return texts.strip()
    return '\n'.join(str(value).strip() for value in texts if str(value).strip()).strip()


def _text_confidence_threshold(text: str) -> float:
    """Require stronger recognition evidence for very short OCR fragments.

    One or two character subtitles remain eligible (common for CJK and numeric
    dialogue), but need more confidence than a longer phrase to pass.
    """
    visible = sum(char.isalnum() for char in text)
    return 0.78 if visible <= 2 else 0.45


def _gray_thumbnail(frame, np):
    """Return a small grayscale view for inexpensive temporal change checks."""
    height, width = frame.shape[:2]
    step = max(1, math.ceil(max(height / CHANGE_GATE_HEIGHT,
                                width / CHANGE_GATE_WIDTH)))
    sampled = frame[::step, ::step]
    if sampled.ndim == 2:
        return sampled.astype(np.uint8, copy=False)
    # Mean channels are adequate for change detection and avoid a cv2 call in
    # the sampling loop. Inference still receives the original color crop.
    return sampled.astype(np.uint16).mean(axis=2).astype(np.uint8)


def _has_subtitle_shaped_change(previous, current, np):
    """Detect localized, line-like changes while ignoring broad scene motion.

    Subtitle transitions produce a sparse band of changed pixels with a
    horizontal footprint. Camera motion and cuts usually change a much larger
    fraction of the ROI or lack that compact line geometry. Periodic OCR remains
    the safety net for small text and changes this inexpensive gate misses.
    """
    if previous.shape != current.shape:
        return True
    signed_delta = current.astype(np.int16) - previous.astype(np.int16)
    delta = np.abs(signed_delta)
    changed = delta > CHANGE_GATE_PIXEL_THRESHOLD
    height, width = changed.shape
    if height < 2 or width < 8:
        return bool(changed.any())

    # Compare each row's motion energy with the frame's typical level. Using
    # delta magnitude (not only a binary mask) can still reveal subtitle strokes
    # when background motion already changed most pixels.
    row_counts = delta.sum(axis=1)
    row_activity = np.convolve(row_counts, np.ones(3, dtype=np.int32), mode='same')
    background_activity = float(np.median(row_activity))
    active_rows = row_activity >= background_activity + max(
        width * 6, background_activity * 0.06)
    edges = np.diff(np.pad(active_rows.astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    total_changed = int(changed.sum())
    if (total_changed > width * height * 0.35
            and abs(float(signed_delta.mean())) > 28):
        # A broad same-direction luminance shift is likely a hard cut or flash.
        # Panning textures usually contain mixed signed changes and continue to
        # the local band check below.
        return True
    for start, end in zip(starts, ends):
        band_height = end - start
        if band_height < 2 or band_height > max(5, int(height * 0.22)):
            continue
        band = changed[max(0, start - 1):min(height, end + 1)]
        column_counts = delta[max(0, start - 1):min(height, end + 1)].sum(
            axis=0).astype(np.float32)
        outside_height = max(1, height - band.shape[0])
        outside_counts = delta.sum(axis=0) - column_counts
        expected_motion = outside_counts * (band.shape[0] / outside_height)
        excess_columns = column_counts - expected_motion
        active_columns = excess_columns >= max(32, int(band.shape[0] * 8))
        columns = np.flatnonzero(active_columns)
        if len(columns) < 2:
            continue
        span = int(columns[-1] - columns[0] + 1)
        band_changed = int(band.sum())
        regular_text_span = span >= width * 0.18
        narrow_glyph_span = (
            span >= max(4, int(width * 0.025))
            and band_changed >= max(8, int(band.shape[0] * span * 0.008)))
        if ((regular_text_span or narrow_glyph_span) and span <= width * 0.96
                and band_changed <= width * max(5, int(height * 0.22))):
            return True
    return False


def _sampled_frames(capture, cv2_module, fps, region, cancel_check, stats,
                    copy_crops=False):
    """Yield ordered sampled crops while recording decode and ROI work."""
    top, bottom, left, right = region
    stride = max(1, round(fps / SAMPLE_FPS))
    frame_index = 0
    last_timestamp = -1.0
    stats.update({
        'video_decode_seconds': 0.0,
        'video_sampling_seconds': 0.0,
        'frame_index': 0,
        'sample_count': 0,
        'last_timestamp': -1.0,
        'last_decoded_timestamp': -1.0,
    })

    def record_decoded_pts(milliseconds):
        timestamp = milliseconds / 1000
        if (milliseconds >= 0 and timestamp > stats['last_decoded_timestamp']
                and timestamp <= 7 * 24 * 60 * 60):
            stats['last_decoded_timestamp'] = timestamp

    while True:
        if cancel_check and cancel_check():
            raise InterruptedError('Processing was cancelled.')
        current_index = frame_index
        decode_started = time.perf_counter()
        if callable(getattr(capture, 'grab', None)) and callable(
                getattr(capture, 'retrieve', None)):
            if not capture.grab():
                stats['video_decode_seconds'] += time.perf_counter() - decode_started
                break
            frame_index += 1
            stats['frame_index'] = frame_index
            if current_index % stride:
                frame_time_ms = float(capture.get(cv2_module.CAP_PROP_POS_MSEC) or 0)
                record_decoded_pts(frame_time_ms)
                stats['video_decode_seconds'] += time.perf_counter() - decode_started
                continue
            ok, frame = capture.retrieve()
            # Sampled frames need the post-retrieve position: some backends
            # only advance CAP_PROP_POS_MSEC once retrieve() has delivered it.
            frame_time_ms = float(capture.get(cv2_module.CAP_PROP_POS_MSEC) or 0)
            record_decoded_pts(frame_time_ms)
        else:
            ok, frame = capture.read()
            if ok:
                frame_index += 1
                stats['frame_index'] = frame_index
                frame_time_ms = float(capture.get(cv2_module.CAP_PROP_POS_MSEC) or 0)
                record_decoded_pts(frame_time_ms)
            if ok and current_index % stride:
                stats['video_decode_seconds'] += time.perf_counter() - decode_started
                continue
        stats['video_decode_seconds'] += time.perf_counter() - decode_started
        if not ok:
            break
        stats['sample_count'] += 1
        sampling_started = time.perf_counter()
        height, width = frame.shape[:2]
        y1, y2 = int(height * top), int(height * bottom)
        x1, x2 = int(width * left), int(width * right)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            stats['video_sampling_seconds'] += time.perf_counter() - sampling_started
            continue
        nominal_timestamp = current_index / fps
        frame_timestamp = frame_time_ms / 1000
        if (frame_time_ms >= 0 and frame_timestamp > last_timestamp
                and frame_timestamp <= 7 * 24 * 60 * 60):
            timestamp = frame_timestamp
        else:
            timestamp = max(nominal_timestamp, last_timestamp + 1 / fps)
        last_timestamp = timestamp
        stats['last_timestamp'] = timestamp
        stats['frame_index'] = frame_index
        stats['video_sampling_seconds'] += time.perf_counter() - sampling_started
        # OpenCV adapters may reuse the same backing frame buffer on retrieve.
        # The prefetch queue must own stable pixels until OCR consumes them.
        yield crop.copy() if copy_crops else crop, current_index, timestamp
    stats['frame_index'] = frame_index
    stats['last_timestamp'] = last_timestamp


def _prefetched_frames(capture, cv2_module, fps, region, cancel_check, stats,
                       queue_capacity=3):
    """Overlap one decoder with OCR using a small, ordered queue."""
    samples = queue.Queue(maxsize=queue_capacity)
    stopped = threading.Event()
    end = object()
    stats['prefetch_enabled'] = True
    stats['prefetch_queue_capacity'] = queue_capacity
    stats['prefetch_max_depth'] = 0

    def put(message):
        while not stopped.is_set():
            try:
                samples.put(message, timeout=0.1)
                stats['prefetch_max_depth'] = max(
                    stats['prefetch_max_depth'], samples.qsize())
                return True
            except queue.Full:
                continue
        return False

    def decode():
        try:
            for sample in _sampled_frames(
                    capture, cv2_module, fps, region, stopped.is_set, stats,
                    copy_crops=True):
                if not put(('sample', sample)):
                    break
            put(('end', end))
        except Exception as exc:
            put(('error', exc))
        finally:
            capture.release()

    worker = threading.Thread(target=decode, name='subtitle-video-prefetch', daemon=True)
    worker.start()
    try:
        while True:
            if cancel_check and cancel_check():
                raise InterruptedError('Processing was cancelled.')
            try:
                kind, value = samples.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if kind == 'end':
                break
            if kind == 'error':
                raise value
            yield value
    finally:
        stopped.set()
        # Drain pending crops so a producer blocked by the bounded queue can
        # observe cancellation and release its capture deterministically.
        while worker.is_alive():
            try:
                samples.get(timeout=0.05)
            except queue.Empty:
                pass
            worker.join(timeout=0.05)


class DirectOnnxOCR:
    """Small PP-OCRv6 DB detector and CTC recognizer using ONNX Runtime directly."""

    def __init__(self, model_dir, compute, ort):
        import cv2
        import numpy as np

        self.cv2, self.np = cv2, np
        self.stage_timings = {
            'detector_preprocess_seconds': 0.0,
            'detector_forward_seconds': 0.0,
            'detector_postprocess_seconds': 0.0,
            'recognizer_preprocess_seconds': 0.0,
            'recognizer_forward_seconds': 0.0,
            'recognizer_postprocess_seconds': 0.0,
            'detector_runs': 0,
            'recognizer_runs': 0,
            'recognizer_images': 0,
            # Counts successful ONNX calls by the actual input batch size.
            # Keep this separate from the configured batch size because width
            # grouping and final partial groups routinely produce smaller runs.
            'recognizer_batch_histogram': {},
            'recognizer_batch_fallbacks': 0,
            'recognizer_frame_batches': 0,
            'recognizer_frames': 0,
            'recognizer_cross_frame_batches': 0,
        }
        model_dir = Path(model_dir)
        det = model_dir / 'PP-OCRv6_det_small.onnx'
        rec = model_dir / 'PP-OCRv6_rec_small.onnx'
        dictionary = model_dir / 'ppocrv6_dict.txt'
        for path in (det, rec, dictionary):
            if not path.is_file():
                raise RuntimeError(f'OCR asset is missing: {path}')
        self.characters = [''] + dictionary.read_text(encoding='utf-8-sig').splitlines() + [' ']
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        provider = {'cuda': 'CUDAExecutionProvider',
                    'auto': 'CUDAExecutionProvider',
                    'mixed': 'CUDAExecutionProvider',
                    'dml': 'DmlExecutionProvider'}.get(compute)
        if compute == 'dml':
            # DirectML requires sequential execution and no memory pattern.
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [provider, 'CPUExecutionProvider'] if provider else ['CPUExecutionProvider']
        if compute == 'dml':
            # Prefer the faster hardware adapter on hybrid laptops; exclude WARP/NPU.
            providers = [('DmlExecutionProvider', {
                'performance_preference': 'high_performance', 'device_filter': 'gpu'}),
                'CPUExecutionProvider']
        self.detector = ort.InferenceSession(str(det), sess_options=options, providers=providers)
        # Mixed mode keeps both ONNX sessions on CUDA. Its CPU work is bounded
        # decode, sampling, ROI preparation and DB/CTC postprocessing, which
        # overlaps the single ordered GPU inference lane.
        recognizer_providers = providers
        self.recognizer = ort.InferenceSession(
            str(rec), sess_options=options, providers=recognizer_providers)
        # Some ONNX exports fix the first dimension at one. Keep those on the
        # original path instead of assuming a dynamic batch dimension.
        batch_dim = self.recognizer.get_inputs()[0].shape[0]
        self.recognizer_batch_size = 8 if compute in ('cuda', 'auto', 'mixed') and not isinstance(batch_dim, int) else 1
        self.recognizer_calibration_status = (
            'pending' if self.recognizer_batch_size == 8 else 'disabled')
        self.recognizer_calibration_seconds = 0.0
        expected = (provider, provider)
        if provider and any(session.get_providers()[0] != required
                            for session, required in zip((self.detector, self.recognizer), expected)):
            raise RuntimeError(f'OCR sessions did not select requested providers: {expected}.')
        if provider in ('CUDAExecutionProvider', 'DmlExecutionProvider'):
            # ORT can replace a CUDA session with CPU after a run error. An
            # explicit GPU request must fail at that error, not after scanning
            # the entire video on CPU.
            for label, session in (('detector', self.detector),
                                   ('recognizer', self.recognizer)):
                disable_fallback = getattr(session, 'disable_fallback', None)
                if not callable(disable_fallback):
                    raise RuntimeError(f'The OCR {label} session cannot disable GPU fallback.')
                disable_fallback()
        self.recognizer_provider_calibration_status = 'pending' if compute == 'auto' else 'disabled'
        self.recognizer_provider_calibration_seconds = 0.0
        self.recognizer_provider_baseline_seconds = None
        self.recognizer_provider_candidate_seconds = None
        self._cpu_recognizer_candidate = None
        if compute == 'auto':
            # CPU is a candidate only; every selected GPU session is checked
            # above before the video is opened. Calibration is bounded to one
            # live group and the GPU result is retained for that group.
            self._cpu_recognizer_candidate = ort.InferenceSession(
                str(rec), sess_options=options, providers=['CPUExecutionProvider'])
            if self._cpu_recognizer_candidate.get_providers()[0] != 'CPUExecutionProvider':
                raise RuntimeError('CPU recognizer candidate did not select CPUExecutionProvider.')

    def __call__(self, bgr):
        return self.recognize_frames([bgr])[0]

    def recognize_frames(self, frames):
        """Detect each frame normally, then batch their ordered text crops."""
        if not frames:
            return []
        frame_data = [self._detect_crops(frame) for frame in frames]
        crop_count = sum(len(frame_crops) for _, frame_crops in frame_data)
        combined = crop_count <= CUDA_RECOGNIZER_CROP_LIMIT
        if combined:
            flattened = [crop for _, frame_crops in frame_data
                         for _, _, crop in frame_crops]
            recognized = iter(self._recognize_many(flattened))
        outputs = []
        for height, frame_crops in frame_data:
            if not combined:
                recognized = iter(self._recognize_many(
                    [crop for _, _, crop in frame_crops]))
            lines = []
            for y1, x1, _ in frame_crops:
                text, confidence = next(recognized)
                if text and confidence >= _text_confidence_threshold(text):
                    lines.append((y1, x1, text))
            lines.sort(key=lambda line: (round(line[0] / max(1, height * 0.06)), line[1]))
            outputs.append([line[2] for line in lines])
        self.stage_timings['recognizer_frame_batches'] = self.stage_timings.get(
            'recognizer_frame_batches', 0) + 1
        self.stage_timings['recognizer_frames'] = self.stage_timings.get(
            'recognizer_frames', 0) + len(frames)
        if len(frames) > 1 and combined:
            self.stage_timings['recognizer_cross_frame_batches'] = self.stage_timings.get(
                'recognizer_cross_frame_batches', 0) + 1
        return outputs

    def recognize_detected_frame(self, frame_data):
        """Recognize one frame whose detector pass has already completed."""
        return self.recognize_detected_frames([frame_data])[0]

    def recognize_detected_frames(self, frame_data_list):
        """Recognize ordered detector results from several queued frames."""
        if not frame_data_list:
            return []
        crop_count = sum(len(frame_crops) for _, frame_crops in frame_data_list)
        combined = crop_count <= CUDA_RECOGNIZER_CROP_LIMIT
        if combined:
            flattened = [crop for _, frame_crops in frame_data_list
                         for _, _, crop in frame_crops]
            recognized = iter(self._recognize_many(flattened))
        outputs = []
        for height, frame_crops in frame_data_list:
            if not combined:
                recognized = iter(self._recognize_many(
                    [crop for _, _, crop in frame_crops]))
            lines = []
            for y1, x1, _ in frame_crops:
                text, confidence = next(recognized)
                if text and confidence >= _text_confidence_threshold(text):
                    lines.append((y1, x1, text))
            lines.sort(key=lambda line: (round(line[0] / max(1, height * 0.06)), line[1]))
            outputs.append([line[2] for line in lines])
        self.stage_timings['recognizer_frame_batches'] = self.stage_timings.get(
            'recognizer_frame_batches', 0) + 1
        self.stage_timings['recognizer_frames'] = self.stage_timings.get(
            'recognizer_frames', 0) + len(frame_data_list)
        if len(frame_data_list) > 1 and combined:
            self.stage_timings['recognizer_cross_frame_batches'] = self.stage_timings.get(
                'recognizer_cross_frame_batches', 0) + 1
        return outputs

    def _detect_crops(self, bgr):
        cv2, np = self.cv2, self.np
        height, width = bgr.shape[:2]
        stage_started = time.perf_counter()
        # PP-OCR detection normalization: RGB, [0,1], mean/std 0.5.
        scale = min(1.0, 960 / max(height, width))
        dh = max(32, round(height * scale / 32) * 32)
        dw = max(32, round(width * scale / 32) * 32)
        resized = cv2.resize(bgr, (dw, dh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = ((rgb.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        self.stage_timings['detector_preprocess_seconds'] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        probability = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})[0]
        self.stage_timings['detector_forward_seconds'] += time.perf_counter() - stage_started
        self.stage_timings['detector_runs'] += 1
        stage_started = time.perf_counter()
        mask = np.squeeze(probability)
        if mask.ndim != 2:
            raise RuntimeError(f'Unexpected detector output shape: {probability.shape}')
        binary = (mask > 0.3).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        crops = []
        for contour in contours:
            if cv2.contourArea(contour) < 8:
                continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            # Expand the detector polygon around its center to retain character edges.
            center = box.mean(axis=0)
            box = (box - center) * 1.6 + center
            xs = np.clip(box[:, 0] * width / mask.shape[1], 0, width - 1)
            ys = np.clip(box[:, 1] * height / mask.shape[0], 0, height - 1)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            score_map = mask[max(0, int(box[:, 1].min())):min(mask.shape[0], int(box[:, 1].max()) + 1),
                             max(0, int(box[:, 0].min())):min(mask.shape[1], int(box[:, 0].max()) + 1)]
            if score_map.size == 0 or float(score_map.mean()) < 0.25:
                continue
            crops.append((y1, x1, bgr[y1:y2, x1:x2]))
        self.stage_timings['detector_postprocess_seconds'] += time.perf_counter() - stage_started
        return height, crops

    def _recognize(self, bgr):
        return self._recognize_many([bgr])[0]

    def _recognize_many(self, crops):
        """Recognize crops and test a larger batch once on representative input.

        Calibration repeats at most one bounded, real crop group. The larger
        batch is retained only if it returns exactly the same decoded text and
        confidence values and is faster on that group. The baseline result is
        used otherwise, so calibration cannot change subtitle output.
        """
        if not crops:
            return []
        if (getattr(self, 'recognizer_provider_calibration_status', 'disabled') == 'pending'
                and len(crops) >= PROVIDER_CALIBRATION_MIN_CROPS):
            return self._calibrate_recognizer_provider(crops)
        batch_size = getattr(self, 'recognizer_batch_size', 1)
        if (getattr(self, 'recognizer_calibration_status', 'disabled') != 'pending'
                or batch_size != 8 or len(crops) < CUDA_RECOGNIZER_CALIBRATION_CROPS):
            return self._recognize_many_impl(crops)

        self.recognizer_calibration_status = 'running'
        started = time.perf_counter()
        baseline_started = time.perf_counter()
        baseline = self._recognize_many_impl(crops, batch_size=8)
        baseline_seconds = time.perf_counter() - baseline_started
        # A baseline batch may have revealed that the export rejects batching.
        if self.recognizer_batch_size != 8:
            self.recognizer_calibration_status = 'baseline_fallback'
            self.recognizer_calibration_seconds += time.perf_counter() - started
            return baseline

        candidate_started = time.perf_counter()
        try:
            candidate = self._recognize_many_impl(
                crops, batch_size=CUDA_RECOGNIZER_MAX_BATCH_SIZE,
                calibration=True)
        except Exception:
            self.recognizer_calibration_status = 'candidate_failed'
            self.recognizer_calibration_seconds += time.perf_counter() - started
            return baseline
        candidate_seconds = time.perf_counter() - candidate_started
        same_results = candidate == baseline
        # The first representative group is run twice. Require a large enough
        # measured gain that one later eligible group can repay that bounded
        # startup cost; longer videos then accrue additional savings.
        if same_results and candidate_seconds < baseline_seconds * 0.5:
            self.recognizer_batch_size = CUDA_RECOGNIZER_MAX_BATCH_SIZE
            self.recognizer_calibration_status = 'larger_batch_selected'
            result = candidate
        else:
            self.recognizer_batch_size = 8
            self.recognizer_calibration_status = (
                'output_mismatch' if not same_results
                else 'larger_batch_gain_too_small')
            result = baseline
        self.recognizer_calibration_seconds += time.perf_counter() - started
        return result

    def _calibrate_recognizer_provider(self, crops):
        """Compare real bounded crops once, retaining the baseline output.

        A decoded mismatch keeps CUDA for the run. CPU is selected only when
        its measured gain is large enough to repay this pass on a longer clip.
        """
        self.recognizer_provider_calibration_status = 'running'
        started = time.perf_counter()
        gpu_started = time.perf_counter()
        baseline = self._recognize_many_impl(crops, batch_size=self.recognizer_batch_size)
        gpu_seconds = time.perf_counter() - gpu_started
        self.recognizer_provider_baseline_seconds = round(gpu_seconds, 3)
        candidate_session = self._cpu_recognizer_candidate
        original_session = self.recognizer
        cpu_started = time.perf_counter()
        try:
            self.recognizer = candidate_session
            candidate = self._recognize_many_impl(crops, batch_size=1, calibration=True)
            cpu_seconds = time.perf_counter() - cpu_started
            self.recognizer_provider_candidate_seconds = round(cpu_seconds, 3)
        except Exception:
            self.recognizer_provider_calibration_status = 'candidate_failed'
            return baseline
        finally:
            self.recognizer = original_session
            self.recognizer_provider_calibration_seconds += time.perf_counter() - started
            self._cpu_recognizer_candidate = None
        # Tiny floating point differences in scores do not affect SRT when
        # both paths make the same text inclusion decision.
        same_output = all(
            candidate_text == baseline_text
            and (candidate_score >= _text_confidence_threshold(candidate_text))
            == (baseline_score >= _text_confidence_threshold(baseline_text))
            for (candidate_text, candidate_score), (baseline_text, baseline_score)
            in zip(candidate, baseline)) and len(candidate) == len(baseline)
        if not same_output:
            self.recognizer_provider_calibration_status = 'output_mismatch'
        elif cpu_seconds < gpu_seconds * 0.8:
            self.recognizer = candidate_session
            self.recognizer_batch_size = 1
            self.recognizer_calibration_status = 'disabled_cpu_selected'
            self.recognizer_provider_calibration_status = 'cpu_selected'
        else:
            self.recognizer_provider_calibration_status = 'cuda_selected'
        return baseline

    def _recognize_many_impl(self, crops, batch_size=None, calibration=False):
        if not crops:
            return []
        cv2, np = self.cv2, self.np
        crop_widths = []
        for index, bgr in enumerate(crops):
            h, w = bgr.shape[:2]
            target_w = min(2048, max(32, int(np.ceil((48 * w / h) / 8) * 8)))
            crop_widths.append((index, target_w, bgr))
        results = [None] * len(crops)
        if batch_size is None:
            batch_size = getattr(self, 'recognizer_batch_size', 1)
        # Similar widths are padded only on the right. Grouping within one
        # frame bounds memory use and keeps the original reading order. Keep
        # only crop references and widths here; materialize each float tensor
        # directly into its final batch instead of retaining a second full set
        # of normalized crop tensors alongside the padded batch.
        pending = sorted(crop_widths, key=lambda entry: entry[1])
        while pending:
            group = [pending.pop(0)]
            if batch_size > 1:
                while (pending and len(group) < batch_size
                       and pending[0][1] <= group[0][1] * 1.25):
                    group.append(pending.pop(0))
            max_width = group[-1][1]
            tensor = np.full((len(group), 3, 48, max_width), -1, dtype=np.float32)
            stage_started = time.perf_counter()
            for row, (_, width, bgr) in enumerate(group):
                resized = cv2.resize(bgr, (width, 48))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor[row, :, :, :width] = (
                    (rgb.astype(np.float32) / 255 - 0.5) / 0.5
                ).transpose(2, 0, 1)
            self.stage_timings['recognizer_preprocess_seconds'] += (
                time.perf_counter() - stage_started)
            stage_started = time.perf_counter()
            try:
                scores = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: tensor})[0]
            except Exception as error:
                if len(group) == 1:
                    raise
                if calibration:
                    raise
                if self.recognizer.get_providers()[0] != 'CUDAExecutionProvider':
                    raise RuntimeError('The OCR recognizer lost CUDA after a batch failure.') from error
                # A dynamic first dimension does not guarantee that every
                # operator in an export supports batching. This also covers
                # a cuDNN algorithm lookup failure for a particular batch.
                # Retry unchanged single crops on CUDA and retain that mode.
                pending = sorted(group + pending, key=lambda entry: entry[1])
                batch_size = self.recognizer_batch_size = 1
                self.stage_timings['recognizer_batch_fallbacks'] = self.stage_timings.get('recognizer_batch_fallbacks', 0) + 1
                self.stage_timings['recognizer_forward_seconds'] += time.perf_counter() - stage_started
                continue
            self.stage_timings['recognizer_forward_seconds'] += time.perf_counter() - stage_started
            self.stage_timings['recognizer_runs'] += 1
            self.stage_timings['recognizer_images'] = self.stage_timings.get('recognizer_images', 0) + len(group)
            histogram = self.stage_timings.setdefault('recognizer_batch_histogram', {})
            batch_key = str(len(group))
            histogram[batch_key] = histogram.get(batch_key, 0) + 1
            stage_started = time.perf_counter()
            if scores.ndim != 3 or scores.shape[0] != len(group):
                raise RuntimeError(f'Unexpected recognizer output shape: {scores.shape}')
            for (index, _, _), row in zip(group, scores):
                results[index] = self._decode_recognition(row)
            self.stage_timings['recognizer_postprocess_seconds'] += time.perf_counter() - stage_started
        return results

    def _decode_recognition(self, scores):
        np = self.np
        scores = scores.astype(np.float32, copy=False)
        # Paddle exports commonly include softmax probabilities; accept those
        # when rows sum to one. Some ONNX exports expose logits instead, in
        # which case raw maxima are not calibrated confidence values.
        row_sums = scores.sum(axis=-1)
        if (np.any(scores < 0) or np.any(scores > 1)
                or not np.all((row_sums >= 0.98) & (row_sums <= 1.02))):
            shifted = scores - scores.max(axis=-1, keepdims=True)
            exponentials = np.exp(shifted)
            scores = exponentials / exponentials.sum(axis=-1, keepdims=True)
        indices = scores.argmax(axis=-1)
        # Argmax already identifies each step's maximum. Gather that same
        # value instead of reducing the full class row a second time.
        confidences = scores[np.arange(scores.shape[0]), indices]
        out, scores, previous = [], [], -1
        for index, confidence in zip(indices, confidences):
            index = int(index)
            if index != 0 and index != previous:
                if index >= len(self.characters):
                    raise RuntimeError('Recognition dictionary does not match model output.')
                out.append(self.characters[index])
                scores.append(float(confidence))
            previous = index
        return ''.join(out).strip(), sum(scores) / len(scores) if scores else 0.0


def create_ocr(compute: str, models_dir=None):
    """Use only direct ONNX assets."""
    import onnxruntime as ort

    cuda_packages = {}
    if compute in ('cuda', 'auto', 'mixed'):
        from probe import verify_cuda_packages
        cuda_packages = verify_cuda_packages()
        preload = getattr(ort, 'preload_dlls', None)
        if callable(preload):
            preload(directory='')
        available = list(ort.get_available_providers())
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(f'CUDAExecutionProvider was not found: {available}')
    if compute == 'dml':
        available = list(ort.get_available_providers())
        if 'DmlExecutionProvider' not in available:
            raise RuntimeError(f'DmlExecutionProvider was not found: {available}')
    model_root = models_dir or os.environ.get('SUBHOOPER_OCR_MODELS')
    if not model_root:
        raise RuntimeError('Verified direct ONNX assets are required (--models-dir).')
    engine = DirectOnnxOCR(model_root, compute, ort)
    engine.runtime_packages = cuda_packages
    return engine, ort


def _sessions(ocr):
    providers = {}
    for label, attr in (('detector', 'text_det'), ('classifier', 'text_cls'),
                        ('recognizer', 'text_rec')):
        value = getattr(ocr, label, None) or getattr(ocr, attr, None)
        session = getattr(value, 'session', value)
        session = getattr(session, 'session', session)
        getter = getattr(session, 'get_providers', None)
        providers[label] = [str(p) for p in getter()] if callable(getter) else []
    return providers


def _classifier_enabled(ocr):
    for name in ('cfg', 'config'):
        config = getattr(ocr, name, None)
        global_config = getattr(config, 'Global', None) if config is not None else None
        enabled = getattr(global_config, 'use_cls', None) if global_config is not None else None
        if enabled is not None:
            return bool(enabled)
    return False


def _cuda_primary_all(stage_providers, ocr, ocr_count):
    if not ocr_count:
        return False
    required = ['detector', 'recognizer']
    if _classifier_enabled(ocr):
        required.append('classifier')
    active = [label for label, values in stage_providers.items() if values]
    return all(stage_providers.get(label) and
               stage_providers[label][0] == 'CUDAExecutionProvider'
               for label in required) and all(
        stage_providers[label][0] == 'CUDAExecutionProvider' for label in active)


def extract_video(video, output, region, compute='cpu', cancel_check=None,
                  cv2_module=None, ocr=None, models_dir=None, progress=False):
    """Sample at 2 fps, OCR changing subtitle regions, and write SRT plus metrics.

    Region values are normalized frame fractions in top, bottom, left, right order.
    """
    if compute not in ('cpu', 'cuda', 'dml', 'mixed', 'auto'):
        raise ValueError('compute must be cpu, cuda, dml, mixed or auto')
    top, bottom, left, right = map(float, region)
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError('region fractions must satisfy 0 <= top < bottom <= 1 and 0 <= left < right <= 1')
    if cv2_module is None:
        import cv2 as cv2_module
    # OpenCV is used for small ROI transforms; its default all-core pool can
    # contend with the decoder and the two-thread ONNX inference sessions.
    set_threads = getattr(cv2_module, 'setNumThreads', None)
    # Bound OpenCV's helper pool without pinning it to one thread. OCR already
    # uses a small ONNX pool; keeping OpenCV at no more than four workers
    # allows useful parallel resize/color work without claiming every core.
    opencv_threads = min(4, max(2, (os.cpu_count() or 2) // 2))
    if callable(set_threads):
        set_threads(opencv_threads)
    if ocr is None:
        ocr, ort = create_ocr(compute, models_dir)
    else:
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2_module.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video}')
    fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0)
    reported_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        capture.release()
        raise RuntimeError('Video has no usable frame rate.')
    stride = max(1, round(fps / SAMPLE_FPS))
    previous_sample_gray = None
    previous_sample_timestamp = None
    last_ocr_time = None
    last_ocr_text = None
    events = []
    sample_count = ocr_count = skipped = 0
    last_progress_percent = -1
    change_gate_checks = change_gate_triggers = forced_ocr_checks = 0
    forced_only_ocr_calls = transition_backdates = 0
    ocr_total_seconds = gate_seconds = 0.0
    started = time.monotonic()
    reader_stats = {}
    # OpenCV decode and ONNX inference can overlap even on CPU. The queue is
    # bounded to three copied ROI frames; measured CPU benefit is a test gate.
    prefetch_enabled = isinstance(ocr, DirectOnnxOCR)
    frame_batching = (compute in ('cuda', 'auto') and isinstance(ocr, DirectOnnxOCR)
                      and hasattr(ocr, 'detector') and hasattr(ocr, 'recognizer')
                      and getattr(ocr, 'recognizer_batch_size', 1) > 1
                      and callable(getattr(ocr, 'recognize_frames', None)))
    mixed_detector_overlap = (
        compute in ('mixed', 'dml') and isinstance(ocr, DirectOnnxOCR)
        and callable(getattr(ocr, '_detect_crops', None))
        and callable(getattr(ocr, 'recognize_detected_frame', None)))
    mixed_pending = []
    mixed_detector_executor = None
    mixed_detector_submitted = mixed_detector_max_pending = 0
    mixed_detector_elapsed_seconds = mixed_detector_wait_seconds = 0.0
    ocr_frame_batch = []

    def apply_ocr_results(batch, recognized, ocr_elapsed):
        nonlocal ocr_count, ocr_total_seconds, forced_only_ocr_calls
        nonlocal transition_backdates, last_ocr_text
        ocr_total_seconds += ocr_elapsed
        ocr_count += len(batch)
        for item, result in zip(batch, recognized):
            (_crop, _index, timestamp, force, changed_text_band,
             prior_sample_timestamp, prior_ocr_time) = item
            text = _recognized_text(result)
            if force and not changed_text_band and prior_ocr_time is not None:
                forced_only_ocr_calls += 1
            event_timestamp = timestamp
            if (last_ocr_text is not None
                    and _compact(text) != _compact(last_ocr_text)
                    and not changed_text_band):
                if prior_sample_timestamp is None:
                    estimated_boundary = timestamp - stride / fps
                else:
                    estimated_boundary = (prior_sample_timestamp + timestamp) / 2
                event_timestamp = max(prior_ocr_time or 0.0, estimated_boundary)
                transition_backdates += 1
            last_ocr_text = text
            events.append((event_timestamp, text))

    def flush_ocr_frame_batch():
        nonlocal frame_batching
        if not ocr_frame_batch:
            return
        batch = list(ocr_frame_batch)
        ocr_frame_batch.clear()
        ocr_started = time.perf_counter()
        if len(batch) > 1 and frame_batching:
            recognized = ocr.recognize_frames([item[0] for item in batch])
        else:
            recognized = [ocr(item[0]) for item in batch]
        if (compute == 'auto' and getattr(
                ocr, 'recognizer_provider_calibration_status', None) == 'cpu_selected'):
            frame_batching = False
        if frame_batching and cancel_check and cancel_check():
            raise InterruptedError('Processing was cancelled.')
        apply_ocr_results(batch, recognized, time.perf_counter() - ocr_started)

    def flush_mixed_detector():
        nonlocal mixed_detector_elapsed_seconds, mixed_detector_wait_seconds
        if not mixed_pending:
            return
        batch = mixed_pending[:CUDA_RECOGNIZER_FRAME_BATCH]
        del mixed_pending[:len(batch)]
        detected_frames = []
        items = []
        for item, future in batch:
            wait_started = time.perf_counter()
            detected, detector_elapsed = future.result()
            mixed_detector_wait_seconds += time.perf_counter() - wait_started
            mixed_detector_elapsed_seconds += detector_elapsed
            detected_frames.append(detected)
            items.append(item)
        if cancel_check and cancel_check():
            raise InterruptedError('Processing was cancelled.')
        ocr_started = time.perf_counter()
        if (len(detected_frames) > 1 and callable(getattr(
                ocr, 'recognize_detected_frames', None))):
            recognized = ocr.recognize_detected_frames(detected_frames)
        else:
            recognize_one = getattr(ocr, 'recognize_detected_frame', None)
            recognized = ([recognize_one(detected) for detected in detected_frames]
                          if callable(recognize_one) else [ocr(item[0]) for item in items])
        apply_ocr_results(items, recognized, time.perf_counter() - ocr_started)

    if prefetch_enabled:
        sample_stream = _prefetched_frames(
            capture, cv2_module, fps, (top, bottom, left, right), cancel_check,
            reader_stats)
    else:
        sample_stream = _sampled_frames(
            capture, cv2_module, fps, (top, bottom, left, right), cancel_check,
            reader_stats)
    try:
        for crop, current_index, timestamp in sample_stream:
            sample_count += 1
            if progress and reported_frames > 0:
                percent = min(99, max(0, current_index * 100 // reported_frames))
                if percent > last_progress_percent:
                    print(f'OCRProgress={percent}', flush=True)
                    last_progress_percent = percent
            force_ocr = last_ocr_time is None or timestamp - last_ocr_time >= FORCED_OCR_INTERVAL
            gate_started = time.perf_counter()
            changed_text_band = False
            import numpy as np
            current_gray = _gray_thumbnail(crop, np)
            if previous_sample_gray is not None:
                if previous_sample_gray.shape == current_gray.shape:
                    change_gate_checks += 1
                    changed_text_band = _has_subtitle_shaped_change(
                        previous_sample_gray, current_gray, np)
                else:
                    changed_text_band = True
            previous_sample_gray = current_gray
            if force_ocr:
                forced_ocr_checks += 1
            if changed_text_band:
                change_gate_triggers += 1
            prior_sample_timestamp = previous_sample_timestamp
            previous_sample_timestamp = timestamp
            if not force_ocr and not changed_text_band:
                skipped += 1
                gate_seconds += time.perf_counter() - gate_started
                continue
            gate_seconds += time.perf_counter() - gate_started
            prior_ocr_time = last_ocr_time
            ocr_frame_batch.append((
                crop, current_index, timestamp, force_ocr, changed_text_band,
                prior_sample_timestamp, prior_ocr_time))
            last_ocr_time = timestamp
            if mixed_detector_overlap:
                if mixed_detector_executor is None:
                    mixed_detector_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix='subtitle-mixed-detector')
                item = ocr_frame_batch.pop()
                def detect_timed(frame):
                    detector_started = time.perf_counter()
                    detected_frame = ocr._detect_crops(frame)
                    return detected_frame, time.perf_counter() - detector_started

                future = mixed_detector_executor.submit(detect_timed, item[0])
                mixed_pending.append((item, future))
                mixed_detector_submitted += 1
                # Read the completed task duration below; this stays a worker
                # wall-time metric, distinct from summed ONNX substage times.
                mixed_detector_max_pending = max(
                    mixed_detector_max_pending, len(mixed_pending))
                if len(mixed_pending) >= CUDA_RECOGNIZER_FRAME_BATCH:
                    flush_mixed_detector()
            elif not frame_batching or len(ocr_frame_batch) >= CUDA_RECOGNIZER_FRAME_BATCH:
                flush_ocr_frame_batch()
        if mixed_detector_overlap:
            while mixed_pending:
                flush_mixed_detector()
        else:
            flush_ocr_frame_batch()
        if (frame_batching or mixed_detector_overlap) and cancel_check and cancel_check():
            raise InterruptedError('Processing was cancelled.')
    finally:
        if mixed_detector_executor is not None:
            mixed_detector_executor.shutdown(wait=True, cancel_futures=True)
        close_stream = getattr(sample_stream, 'close', None)
        if callable(close_stream):
            close_stream()
        if not prefetch_enabled:
            capture.release()

    frame_index = int(reader_stats.get('frame_index', 0))
    sample_count = int(reader_stats.get('sample_count', sample_count))
    video_decode_seconds = float(reader_stats.get('video_decode_seconds', 0.0))
    video_sampling_seconds = (
        float(reader_stats.get('video_sampling_seconds', 0.0)) + gate_seconds)
    last_timestamp = float(reader_stats.get('last_timestamp', -1.0))
    last_decoded_timestamp = float(
        reader_stats.get('last_decoded_timestamp', -1.0))

    nominal_duration = (frame_index / fps if frame_index > 0
                        else reported_frames / fps)
    decoded_duration = (last_decoded_timestamp + 1 / fps
                        if last_decoded_timestamp >= 0 else nominal_duration)
    video_duration = max(
        nominal_duration,
        decoded_duration,
        last_timestamp + 1 / fps if last_timestamp >= 0 else 0.0,
    )
    cues = []
    for index, (start, text) in enumerate(events):
        if not text:
            continue
        next_time = events[index + 1][0] if index + 1 < len(events) else video_duration
        if next_time <= start:
            next_time = start + 1 / SAMPLE_FPS
        key = _compact(text)
        if (index > 0 and events[index - 1][1]
                and cues and _compact(cues[-1][2]) == key
                and start <= cues[-1][1] + 1.1):
            old_start, _, old_text = cues[-1]
            cues[-1] = (old_start, next_time, old_text)
        else:
            cues.append((start, next_time, text))

    (output_dir / 'result.srt').write_text(_srt(cues), encoding='utf-8')
    stage_providers = _sessions(ocr)
    detector_primary = next(iter(stage_providers.get('detector', [])), None)
    recognizer_primary = next(iter(stage_providers.get('recognizer', [])), None)
    available = list(ort.get_available_providers()) if ort is not None else []
    ocr_stage_seconds = dict(getattr(ocr, 'stage_timings', {}) or {})
    if mixed_detector_overlap:
        ocr_total_seconds += sum(ocr_stage_seconds.get(name, 0.0) for name in (
            'detector_preprocess_seconds', 'detector_forward_seconds',
            'detector_postprocess_seconds'))
    ocr_stage_seconds.setdefault('detector_runs', 0)
    ocr_stage_seconds.setdefault('recognizer_runs', 0)
    ocr_stage_seconds.setdefault('recognizer_images', 0)
    ocr_stage_seconds.setdefault('recognizer_batch_histogram', {})
    for name in ('recognizer_frame_batches', 'recognizer_frames',
                 'recognizer_cross_frame_batches'):
        ocr_stage_seconds.setdefault(name, 0)
    for name in (
        'detector_preprocess_seconds', 'detector_forward_seconds',
        'detector_postprocess_seconds', 'recognizer_preprocess_seconds',
        'recognizer_forward_seconds', 'recognizer_postprocess_seconds',
    ):
        ocr_stage_seconds.setdefault(name, 0.0)
    ocr_stage_sum = sum(ocr_stage_seconds[name] for name in (
        'detector_preprocess_seconds', 'detector_forward_seconds',
        'detector_postprocess_seconds', 'recognizer_preprocess_seconds',
        'recognizer_forward_seconds', 'recognizer_postprocess_seconds',
    ))
    metrics = {
        'mode': 'NATIVE_VIDEO_OCR',
        'inference_backend': ('DIRECT_ONNX' if isinstance(ocr, DirectOnnxOCR)
                              else 'EXISTING_APP_COMPATIBILITY'),
        'compute_requested': compute.upper(),
        'onnxruntime_version': getattr(ort, '__version__', None),
        'available_providers': available,
        'session_providers': sorted({provider for values in stage_providers.values()
                                     for provider in values}),
        'model_session_providers': stage_providers,
        'cuda_runtime_packages': getattr(ocr, 'runtime_packages', {}),
        'compute_effective': ('CUDA_CPU_OVERLAP' if compute == 'mixed'
                              and detector_primary == 'CUDAExecutionProvider'
                              and recognizer_primary == 'CUDAExecutionProvider'
                              else ('MIXED' if detector_primary == 'CUDAExecutionProvider'
                              and recognizer_primary == 'CPUExecutionProvider'
                              else ('CUDA' if detector_primary == 'CUDAExecutionProvider'
                                    else compute.upper()))),
        'recognizer_provider_calibration_status': getattr(
            ocr, 'recognizer_provider_calibration_status', 'disabled'),
        'recognizer_provider_calibration_seconds': round(getattr(
            ocr, 'recognizer_provider_calibration_seconds', 0.0), 3),
        'recognizer_provider_baseline_seconds': getattr(
            ocr, 'recognizer_provider_baseline_seconds', None),
        'recognizer_provider_candidate_seconds': getattr(
            ocr, 'recognizer_provider_candidate_seconds', None),
        'cuda_primary_all': _cuda_primary_all(stage_providers, ocr, ocr_count),
        'dml_primary_all': all((stage_providers.get(label) or [None])[0] == 'DmlExecutionProvider'
                               for label in ('detector', 'recognizer')),
        'directml_device_preference': 'high_performance' if compute == 'dml' else None,
        'sample_fps': SAMPLE_FPS,
        'video_prefetch_enabled': prefetch_enabled,
        'video_prefetch_queue_capacity': reader_stats.get('prefetch_queue_capacity', 0),
        'video_prefetch_max_depth': reader_stats.get('prefetch_max_depth', 0),
        'mixed_detector_overlap_enabled': mixed_detector_overlap,
        'mixed_detector_pending_capacity': CUDA_RECOGNIZER_FRAME_BATCH if mixed_detector_overlap else 0,
        'mixed_cpu_decode_gate_overlap_enabled': mixed_detector_overlap,
        'mixed_gpu_detector_worker_enabled': mixed_detector_overlap,
        'mixed_detector_submitted': mixed_detector_submitted,
        'mixed_detector_max_pending': mixed_detector_max_pending,
        'mixed_detector_elapsed_seconds': round(mixed_detector_elapsed_seconds, 3),
        'mixed_detector_wait_seconds': round(mixed_detector_wait_seconds, 3),
        'opencv_threads': opencv_threads,
        'frame_rate': fps,
        'frame_count': frame_index,
        'sample_count': sample_count,
        'sampled_frames': sample_count,
        'ocr_count': ocr_count,
        'stable_samples_skipped': skipped,
        'change_gate_checks': change_gate_checks,
        'change_gate_ocr_triggers': change_gate_triggers,
        'forced_ocr_checks': forced_ocr_checks,
        'forced_only_ocr_calls': forced_only_ocr_calls,
        'ocr_transition_backdates': transition_backdates,
        'sample_period_seconds': round(stride / fps, 3),
        'cue_count': len(cues),
        'duration_seconds': round(video_duration, 3),
        'last_decoded_timestamp_seconds': round(last_decoded_timestamp, 3),
        'video_decode_seconds': round(video_decode_seconds, 3),
        'video_sampling_seconds': round(video_sampling_seconds, 3),
        'ocr_total_seconds': round(ocr_total_seconds, 3),
        'ocr_unattributed_seconds': round(max(0.0, ocr_total_seconds - ocr_stage_sum), 3),
        'ocr_stage_seconds': {
            key: round(value, 3) if key.endswith('_seconds') else value
            for key, value in ocr_stage_seconds.items()
        },
        'recognizer_batch_size': getattr(ocr, 'recognizer_batch_size', 1),
        'recognizer_batch_histogram': dict(
            ocr_stage_seconds.get('recognizer_batch_histogram', {})),
        'recognizer_batch_calibration': getattr(
            ocr, 'recognizer_calibration_status', 'disabled'),
        'recognizer_batch_calibration_seconds': round(
            getattr(ocr, 'recognizer_calibration_seconds', 0.0), 3),
        'recognizer_frame_batch_size': (
            CUDA_RECOGNIZER_FRAME_BATCH if frame_batching else 1),
        'elapsed_seconds': round(time.monotonic() - started, 3),
    }
    if compute == 'dml':
        metrics['compute_effective'] = ('DIRECTML_CPU_OVERLAP' if metrics['dml_primary_all']
                                        else 'GPU_PROVIDER_MISMATCH')
    (output_dir / 'ocr-metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('output')
    parser.add_argument('--region-top', type=float, required=True)
    parser.add_argument('--region-bottom', type=float, required=True)
    parser.add_argument('--region-left', type=float, required=True)
    parser.add_argument('--region-right', type=float, required=True)
    parser.add_argument('--compute', choices=('cpu', 'cuda', 'dml', 'mixed', 'auto'), default='cpu')
    parser.add_argument('--models-dir')
    args = parser.parse_args(argv)
    metrics = extract_video(
        args.video, args.output,
        (args.region_top, args.region_bottom, args.region_left, args.region_right),
        args.compute, models_dir=args.models_dir, progress=True)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
