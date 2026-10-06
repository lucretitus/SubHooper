import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid
import zipfile
from config import get_region_profile, validate_region_offsets
from probe import verify
from runtime import (Workspace, cleanup_stale_workspaces,
                     normalize_srt, run_process, select_compute_mode)


def resolve_results_root(project):
    configured = os.environ.get('SUBTITLE_RESULTS_ROOT', '').strip()
    root = Path(configured).expanduser() if configured else Path(project) / 'results'
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_package_version(project):
    version_file = Path(project) / 'VERSION.txt'
    try:
        version = version_file.read_text(encoding='ascii').strip()
    except OSError:
        version = os.environ.get('SUBHOOPER_VERSION', '').strip()
    return version or 'unknown'


def result_bundle_files(result, destination):
    allowed_suffixes = {'.json', '.log', '.srt', '.txt'}
    return sorted(
        path for path in Path(result).iterdir()
        if path.is_file() and path != Path(destination)
        and path.suffix.lower() in allowed_suffixes)


def create_result_bundle(result, destination):
    files = result_bundle_files(result, destination)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--region', choices=('bottom', 'lower-half', 'full', 'custom'), default='bottom')
    parser.add_argument('--region-top', type=float)
    parser.add_argument('--region-bottom', type=float)
    parser.add_argument('--region-left', type=float)
    parser.add_argument('--region-right', type=float)
    parser.add_argument('--timeout', type=int, default=14400)
    parser.add_argument('--compute', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--ocr-python', type=Path)
    parser.add_argument('--ocr-compute', choices=('cuda', 'mixed', 'dml', 'auto', 'cpu'), default='cpu')
    parser.add_argument('--ocr-requested', choices=('auto', 'cuda', 'mixed', 'dml', 'cpu'), default='cpu')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--client', choices=('cli', 'gui'), default='cli')
    parser.add_argument('--collect-diagnostics', action='store_true')
    parser.add_argument('--keep-temp', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    results_root = resolve_results_root(project)
    package_version = read_package_version(project)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    result = results_root / run_id
    result.mkdir(parents=True, exist_ok=False)
    report = {'status': 'FAILED', 'run': run_id, 'package_version': package_version,
              'client': args.client.upper()}
    workspace = None
    started = time.monotonic()
    code = 1
    try:
        report['stale_temp_cleaned'] = cleanup_stale_workspaces()
        video = args.video.resolve(strict=True)
        if not video.is_file() or video.suffix.lower() not in {'.mp4','.mkv','.avi','.mov','.webm','.ts','.m2ts','.wmv','.m4v'}:
            raise RuntimeError('Select a supported video file.')
        report['environment'] = verify()
        if args.region == 'custom':
            custom_values = (args.region_top, args.region_bottom, args.region_left, args.region_right)
            if any(value is None for value in custom_values):
                raise RuntimeError('All four custom subtitle region bounds are required.')
            region = validate_region_offsets(
                args.region_top, args.region_bottom, args.region_left, args.region_right)
        else:
            region = get_region_profile(args.region)
        compute = select_compute_mode(args.compute)
        ocr_python = (args.ocr_python or Path(sys.executable)).resolve(strict=True)
        if not ocr_python.is_file():
            raise RuntimeError('The OCR Python executable was not found.')
        compute.update(
            ocr_requested=args.ocr_requested,
            ocr_selected=args.ocr_compute,
            ocr='CUDA_CPU_OVERLAP_REQUESTED' if args.ocr_compute == 'mixed' else ('CUDA_REQUESTED' if args.ocr_compute == 'cuda' else 'CPU'),
            ocr_python=str(ocr_python),
            ocr_note=(
                'GPU sessions are validated per model; explicit GPU failures stop without CPU fallback.'
                if args.ocr_compute in ('cuda', 'mixed', 'dml') and args.ocr_requested != 'auto'
                else ('Automatic mode may retry CPU on CUDA failure.' if args.ocr_requested == 'auto' else 'CPU OCR selected.')),
        )
        report.update(video=str(video), engine='native',
                      subtitle_region=args.region, region_offsets=region,
                      compute=compute)
        before = video.stat()
        workspace = Workspace()
        report['temp'] = str(workspace.path)
        if args.keep_temp:
            (workspace.path / '.keep-temp').write_text('true', encoding='ascii')
        # Native OpenCV reads the selected source directly; only generated files
        # belong in the temporary workspace.
        if not args.models_dir or not args.models_dir.is_dir():
            raise RuntimeError('Verified native OCR models are required. Open Settings > Components.')
        if args.collect_diagnostics:
            report['diagnostics_note'] = 'Native extraction does not retain sampled images.'
        print('1/2 Native engine: sampling subtitle regions and recognizing text...', flush=True)
        report['ocr_attempts'] = []

        def execute_native(mode, python_path, name, log_name):
            output = workspace.path / name
            log_path = result / log_name
            attempt = {'compute': mode.upper(), 'python': str(python_path), 'log': str(log_path)}
            started_attempt = time.monotonic()
            try:
                seconds = run_process(
                    [python_path, Path(__file__).with_name('native_ocr.py'), video, output,
                     '--region-top', str(1 - float(region['top'])),
                     '--region-bottom', str(1 - float(region['bottom'])),
                     '--region-left', region['left'], '--region-right', region['right'],
                     '--compute', mode, '--models-dir', args.models_dir], workspace.path, log_path,
                    args.timeout, progress_prefix='OCRProgress=')
                metrics = json.loads((output / 'ocr-metrics.json').read_text(encoding='utf-8'))
                providers = metrics.get('model_session_providers', {})
                report['ocr_fusion'] = metrics
                report['ocr_seconds'] = seconds
                if mode == 'mixed' and not metrics.get('cuda_primary_all'):
                    raise RuntimeError(f'Native OCR GPU + CPU lost CUDA on detector or recognizer; providers={providers}, OCR calls={metrics.get("ocr_count")}. Details: {log_path}')
                if mode == 'dml' and not metrics.get('dml_primary_all'):
                    raise RuntimeError(f'Native OCR lost DirectML on detector or recognizer; providers={providers}. Details: {log_path}')
                if mode == 'cuda' and not metrics.get('cuda_primary_all'):
                    raise RuntimeError(f'Native OCR CUDA was not primary for every model session; providers={providers}, OCR calls={metrics.get("ocr_count")}. Details: {log_path}')
                attempt.update(status='COMPLETE', seconds=seconds)
                report['ocr_attempts'].append(attempt)
                return output, metrics, seconds
            except Exception as exc:
                attempt.update(status='FAILED', seconds=round(time.monotonic() - started_attempt, 3),
                               error=str(exc))
                report['ocr_attempts'].append(attempt)
                raise

        try:
            ocr_output, ocr_metrics, ocr_seconds = execute_native(
                args.ocr_compute, ocr_python, 'native-primary', 'native.log')
            compute['ocr'] = ('CUDA_CPU_OVERLAP' if args.ocr_compute == 'mixed'
                              else ('CUDA_ACTIVE' if args.ocr_compute == 'cuda'
                                    else ('CPU' if args.ocr_compute == 'cpu'
                                          else ocr_metrics.get('compute_effective', 'AUTO'))))
        except Exception as primary_error:
            if args.ocr_requested != 'auto' or args.ocr_compute not in ('cuda', 'mixed', 'dml', 'auto'):
                raise
            report['ocr_fallback_reason'] = str(primary_error)
            print('Native OCR GPU unavailable; retrying on CPU...', flush=True)
            ocr_output, ocr_metrics, ocr_seconds = execute_native(
                'cpu', Path(sys.executable), 'native-cpu-fallback', 'native-cpu-fallback.log')
            compute['ocr'] = 'CPU_FALLBACK'
            compute['ocr_selected'] = 'cpu'
            compute['ocr_python'] = sys.executable
        if args.ocr_compute == 'dml' and compute['ocr'] != 'CPU_FALLBACK':
            compute['ocr'] = 'DIRECTML_CPU_OVERLAP'
        report['ocr_seconds'] = ocr_seconds
        report['ocr_attempt_seconds'] = round(sum(
            attempt.get('seconds', 0) for attempt in report['ocr_attempts']), 3)
        report['ocr_fusion'] = ocr_metrics
        report['image_count'] = ocr_metrics.get('sampled_frames', 0)
        report['ocr_image_source'] = 'VIDEO_SAMPLED'
        srt = ocr_output / 'result.srt'
        report.update(normalize_srt(srt))
        after = video.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('The source video changed during processing.')
        destination = result / (video.stem + '.srt')
        with destination.open('xb') as output, srt.open('rb') as source:
            shutil.copyfileobj(source, output)
        report.update(status='COMPLETE', srt=str(destination))
        code = 0
    except KeyboardInterrupt:
        report['error'] = 'Processing was cancelled.'
        code = 130
    except Exception as exc:
        report['error'] = str(exc)
        (result / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        if workspace:
            try:
                report['temp_bytes_at_end'] = sum(p.stat().st_size for p in workspace.path.rglob('*') if p.is_file())
            except OSError as exc:
                report['temp_measurement_error'] = str(exc)
            if not args.keep_temp:
                try:
                    workspace.cleanup()
                    report['temp_cleaned'] = True
                except Exception as exc:
                    report['cleanup_error'] = str(exc)
            else:
                report['temp_cleaned'] = False
        report['total_seconds'] = round(time.monotonic() - started, 3)
        bundle = result / 'subhooper-result-bundle.zip'
        report['bundle'] = str(bundle)
        (result / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        suspect_count = max(report.get('suspect_cue_count', 0),
                            report.get('suspicious_short_line_cue_count', 0))
        subtitle_count = report.get('subtitle_count', 0)
        quality = 'REVIEW_REQUIRED' if suspect_count else ('READY_FOR_REVIEW' if subtitle_count else '')
        ocr_metrics = report.get('ocr_fusion', {})
        stage_timings = ocr_metrics.get('ocr_stage_seconds', {})
        timing_brief = {
            key: ocr_metrics.get(key) for key in
            ('elapsed_seconds', 'video_decode_seconds', 'video_sampling_seconds',
             'ocr_total_seconds', 'mixed_detector_elapsed_seconds',
             'mixed_detector_wait_seconds', 'opencv_threads')
        }
        timing_brief.update({key: stage_timings.get(key) for key in
                             ('detector_preprocess_seconds', 'detector_forward_seconds',
                              'detector_postprocess_seconds', 'recognizer_preprocess_seconds',
                              'recognizer_forward_seconds', 'recognizer_postprocess_seconds')})
        count_brief = {
            key: ocr_metrics.get(key) for key in
            ('sample_count', 'ocr_count', 'stable_samples_skipped',
             'mixed_detector_max_pending', 'video_prefetch_max_depth',
             'recognizer_batch_size', 'recognizer_batch_calibration')
        }
        count_brief.update({key: stage_timings.get(key) for key in
                            ('detector_runs', 'recognizer_runs', 'recognizer_images',
                             'recognizer_batch_fallbacks', 'recognizer_batch_histogram',
                             'recognizer_frame_batches', 'recognizer_frames',
                             'recognizer_cross_frame_batches')})
        summary = '\n'.join([f'--- SUBHOOPER PIPELINE {package_version} RESULT START ---',
                             f"Pipeline={report['status']}", f"Version={package_version}",
                             f"Client={report.get('client', '')}",
                             f"SRT={report.get('srt', '')}",
                             f"Region={report.get('subtitle_region', '')}",
                             f"Engine={report.get('engine', '')}",
                             f"SampledFrames={report.get('ocr_fusion', {}).get('sampled_frames', '')}",
                             f"Compute={report.get('compute', {}).get('selected', '')}",
                             f"OCRCompute={report.get('compute', {}).get('ocr', '')}",
                             f"OCRRequested={report.get('compute', {}).get('ocr_requested', '')}",
                             f"OCRFallback={bool(report.get('ocr_fallback_reason'))}",
                             f"OCRProviders={','.join(report.get('ocr_fusion', {}).get('session_providers', []))}",
                             f"OCRPrimaryAll={report.get('ocr_fusion', {}).get('cuda_primary_all', False)}",
                             f"OCRMode={report.get('ocr_fusion', {}).get('mode', '')}",
                             f"OCRFusionChanges={report.get('ocr_fusion', {}).get('changed_from_detector', 0)}",
                             f"OCRSource={report.get('ocr_image_source', '')}",
                             f"Subtitles={subtitle_count}", f"SuspiciousShortCues={suspect_count}",
                             f"Diagnostics={report.get('diagnostic_zip', '')}",
                             f"EmptyDropped={report.get('empty_cues_dropped', 0)}",
                             f"MergedDuplicates={report.get('duplicate_cues_merged', 0)}",
                             f"OCRSeconds={report.get('ocr_seconds', '')}",
                             f"OCRStageSeconds={json.dumps(timing_brief, separators=(',', ':'))}",
                             f"OCRStageCounts={json.dumps(count_brief, separators=(',', ':'))}",
                             f"SRTFormat={report.get('srt_format_profile', '')}",
                             f"OverlappingCues={report.get('overlapping_cue_count', 0)}",
                             f"Quality={quality}",
                             f"Error={report.get('error', '')}", f'Report={result / "report.json"}',
                             f'Bundle={bundle}',
                             f"TempCleanup={report.get('temp_cleaned', 'not-created')}",
                             f'--- SUBHOOPER PIPELINE {package_version} RESULT END ---'])
        (result / 'summary.txt').write_text(summary, encoding='utf-8')
        (results_root / 'latest-summary.txt').write_text(summary, encoding='utf-8')
        try:
            report['bundle_file_count'] = len(result_bundle_files(result, bundle))
            (result / 'report.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            create_result_bundle(result, bundle)
        except Exception as exc:
            report['bundle_error'] = str(exc)
            (result / 'report.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(summary, flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
