"""TXTImages OCR worker with a conservative dual-pass text fusion."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import time
import unicodedata


RECOGNIZER_CONFIDENCE = 0.97


def discover_session_providers(root, max_depth=6):
    """Best-effort inspection of RapidOCR's nested ONNX sessions."""
    providers = set()
    visited = set()

    def visit(value, depth):
        if value is None or depth > max_depth or id(value) in visited:
            return
        visited.add(id(value))
        getter = getattr(value, 'get_providers', None)
        if callable(getter):
            try:
                providers.update(str(item) for item in getter())
            except Exception:
                pass
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set)):
            children = value
        elif value.__class__.__module__.startswith(('rapidocr', 'rapid_videocr')):
            children = getattr(value, '__dict__', {}).values()
        else:
            children = ()
        for child in children:
            visit(child, depth + 1)

    visit(root, 0)
    return sorted(providers)


def model_session_providers(ocr_engine):
    result = {}
    for label, attribute in (('detector', 'text_det'),
                             ('classifier', 'text_cls'),
                             ('recognizer', 'text_rec')):
        stage = getattr(ocr_engine, attribute, None)
        wrapper = getattr(stage, 'session', None)
        session = getattr(wrapper, 'session', wrapper)
        getter = getattr(session, 'get_providers', None)
        result[label] = [str(item) for item in getter()] if callable(getter) else []
    return result


def compact_text(value):
    folded = unicodedata.normalize('NFKC', value.casefold())
    return ''.join(char for char in folded if char.isalnum())


def detector_format_suspicious(value):
    return any(re.match(r'^\s*[.,;:!?]\s+\S', line) for line in value.splitlines())


def fuse_text(detector_text, recognizer_text, recognizer_score):
    detector_text = detector_text.strip()
    recognizer_text = recognizer_text.strip()
    if not recognizer_text:
        return detector_text, 'detector_recognizer_empty'
    if not detector_text:
        if recognizer_score >= RECOGNIZER_CONFIDENCE:
            return recognizer_text, 'recognizer_detector_empty'
        return '', 'detector_both_untrusted'
    if compact_text(detector_text) == compact_text(recognizer_text):
        if (detector_format_suspicious(detector_text)
                and recognizer_score >= RECOGNIZER_CONFIDENCE):
            return recognizer_text, 'recognizer_format_repair'
        # The detector pass normally retains better word spacing and line breaks.
        return detector_text, 'detector_equivalent'
    if recognizer_score >= RECOGNIZER_CONFIDENCE:
        return recognizer_text, 'recognizer_high_confidence'
    return detector_text, 'detector_recognizer_untrusted'


def parse_srt(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    timestamp = r'(\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3})'
    header = re.compile(r'(?m)^\s*\d+\s*\r?\n' + timestamp + r'\s*\r?\n')
    matches = list(header.finditer(text))
    cues = {}
    for position, match in enumerate(matches):
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        cues[match.group(1)] = text[match.end():body_end].strip()
    return cues


def timestamp_from_image(path):
    parts = path.stem.split('_')
    if len(parts) < 9:
        raise RuntimeError(f'Invalid VideoSubFinder image name: {path.name}')

    def format_time(values):
        hour, minute, second, millisecond = values
        return f'{int(hour):02d}:{int(minute):02d}:{int(second):02d},{int(millisecond):03d}'

    return f'{format_time(parts[:4])} --> {format_time(parts[5:9])}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images')
    parser.add_argument('output')
    parser.add_argument('--compute', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()

    import onnxruntime as ort
    if args.compute == 'cuda':
        preload = getattr(ort, 'preload_dlls', None)
        if callable(preload):
            preload(directory='')
    available_providers = list(ort.get_available_providers())
    if args.compute == 'cuda' and 'CUDAExecutionProvider' not in available_providers:
        raise RuntimeError(
            f'CUDAExecutionProvider was not found: {available_providers}')

    import rapidocr
    from rapid_videocr import RapidVideOCR, RapidVideOCRInput

    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Existing runtime model files: require local assets; never trigger downloads.
    model_root = Path(rapidocr.__file__).parent / 'models'
    models = {
        'Det': 'PP-OCRv6_det_small.onnx',
        'Rec': 'PP-OCRv6_rec_small.onnx',
        'Cls': 'ch_ppocr_mobile_v2.0_cls_mobile.onnx',
    }
    params = {
        'EngineConfig.onnxruntime.intra_op_num_threads': 2,
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        'EngineConfig.onnxruntime.use_cuda': args.compute == 'cuda',
    }
    for stage, name in models.items():
        path = model_root / name
        if not path.is_file():
            raise RuntimeError(f'Local OCR model is missing; no download was attempted: {path}')
        params[f'{stage}.model_path'] = str(path)

    detector_started = time.monotonic()
    extractor = RapidVideOCR(
        RapidVideOCRInput(out_format='srt', is_batch_rec=False, ocr_params=params)
    )
    extractor(images_dir, output_dir, 'detector')
    detector_seconds = round(time.monotonic() - detector_started, 3)
    detector_cues = parse_srt(output_dir / 'detector.srt')
    image_paths = extractor.get_img_list(images_dir)

    recognizer_started = time.monotonic()
    ocr_engine = extractor.ocr_processor.ocr_engine
    serialized = []
    reasons = {}
    changed = 0
    recognizer_nonempty = 0
    for index, image_path in enumerate(image_paths, 1):
        timestamp = timestamp_from_image(image_path)
        detector_text = detector_cues.get(timestamp, '')
        result = ocr_engine(
            image_path, use_det=False, use_cls=False, use_rec=True
        )
        texts = getattr(result, 'txts', None)
        scores = getattr(result, 'scores', None)
        recognizer_text = texts[0] if texts is not None and len(texts) else ''
        recognizer_score = float(scores[0]) if scores is not None and len(scores) else 0.0
        if recognizer_text.strip():
            recognizer_nonempty += 1
        selected, reason = fuse_text(detector_text, recognizer_text, recognizer_score)
        reasons[reason] = reasons.get(reason, 0) + 1
        if selected != detector_text and selected:
            changed += 1
        serialized.append(f'{index}\n{timestamp}\n{selected}')

    recognizer_seconds = round(time.monotonic() - recognizer_started, 3)
    session_providers = discover_session_providers(extractor)
    stage_providers = model_session_providers(ocr_engine)
    cuda_primary_all = bool(stage_providers) and all(
        providers and providers[0] == 'CUDAExecutionProvider'
        for providers in stage_providers.values())
    (output_dir / 'result.srt').write_text(
        '\n\n'.join(serialized) + '\n', encoding='utf-8'
    )
    metrics = {
        'mode': 'TXT_DUAL_PASS_FUSION',
        'compute_requested': args.compute.upper(),
        'onnxruntime_version': ort.__version__,
        'onnxruntime_distribution': (
            'onnxruntime-gpu' if args.compute == 'cuda' else 'onnxruntime'),
        'onnxruntime_distribution_version': metadata.version(
            'onnxruntime-gpu' if args.compute == 'cuda' else 'onnxruntime'),
        'available_providers': available_providers,
        'session_providers': session_providers,
        'model_session_providers': stage_providers,
        'cuda_primary_all': cuda_primary_all,
        'confidence_threshold': RECOGNIZER_CONFIDENCE,
        'detector_seconds': detector_seconds,
        'recognizer_seconds': recognizer_seconds,
        'image_count': len(image_paths),
        'recognizer_nonempty': recognizer_nonempty,
        'changed_from_detector': changed,
        'selection_reasons': reasons,
    }
    (output_dir / 'ocr-metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
