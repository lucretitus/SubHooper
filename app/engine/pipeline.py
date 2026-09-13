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
from discovery import find_vsf
from config import build_vsf_command, get_region_profile, validate_region_offsets
from probe import verify
from runtime import (Workspace, archive_diagnostic_sets, cleanup_stale_workspaces,
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
    parser.add_argument('--vsf')
    parser.add_argument('--region', choices=('bottom', 'lower-half', 'full', 'custom'), default='bottom')
    parser.add_argument('--region-top', type=float)
    parser.add_argument('--region-bottom', type=float)
    parser.add_argument('--region-left', type=float)
    parser.add_argument('--region-right', type=float)
    parser.add_argument('--timeout', type=int, default=14400)
    parser.add_argument('--compute', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--ocr-python', type=Path)
    parser.add_argument('--ocr-compute', choices=('cuda', 'cpu'), default='cpu')
    parser.add_argument('--ocr-requested', choices=('auto', 'cuda', 'cpu'), default='cpu')
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
        vsf = find_vsf(args.vsf)
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
            rapidvideocr_requested=args.ocr_requested,
            rapidvideocr_selected=args.ocr_compute,
            rapidvideocr='CUDA_REQUESTED' if args.ocr_compute == 'cuda' else 'CPU',
            rapidvideocr_python=str(ocr_python),
            rapidvideocr_note=(
                'CUDA sessions are validated per model; CPU fallback is used on failure.'
                if args.ocr_compute == 'cuda' else 'CPU OCR secildi.'),
        )
        report.update(video=str(video), videosubfinder=str(vsf),
                      subtitle_region=args.region, region_offsets=region,
                      compute=compute)
        before = video.stat()
        workspace = Workspace()
        report['temp'] = str(workspace.path)
        # Always process a private copy: original media never reaches external components.
        if shutil.disk_usage(workspace.path).free < before.st_size + 1024**3:
            raise RuntimeError('There is not enough free disk space for the video copy and temporary files.')
        staged = workspace.path / ('input' + video.suffix.lower())
        print('Copying the video to the isolated workspace...', flush=True)
        shutil.copyfile(video, staged)
        vsf_output = workspace.path / 'vsf'
        vsf_output.mkdir()
        # VSF loads resources from its own installation; no upstream files copied.
        print('1/2 VideoSubFinder: extracting and cleaning subtitle frames...', flush=True)
        report['vsf_seconds'] = run_process(
            build_vsf_command(vsf, staged, vsf_output, args.region, compute['selected'], region),
            vsf.parent, result / 'videosubfinder.log', args.timeout, allowed_codes=(0, -1, 4294967295))
        rgb_images = vsf_output / 'RGBImages'
        txt_images = vsf_output / 'TXTImages'
        allowed_images = {'.jpeg','.jpg','.png','.bmp'}
        rgb_count = sum(1 for p in rgb_images.glob('*') if p.suffix.lower() in allowed_images)
        txt_count = sum(1 for p in txt_images.glob('*') if p.suffix.lower() in allowed_images)
        report.update(rgb_image_count=rgb_count, txt_image_count=txt_count,
                      image_count=txt_count, ocr_image_source='TXTImages')
        if not rgb_count:
            raise RuntimeError('No subtitle frames were found. Review the VideoSubFinder log and subtitle region.')
        if args.collect_diagnostics:
            diagnostic_zip = result / 'vsf-images-diagnostics.zip'
            report.update(archive_diagnostic_sets(
                (('RGBImages', rgb_images), ('TXTImages', txt_images)), diagnostic_zip))
            report['diagnostic_zip'] = str(diagnostic_zip)
        if not txt_count:
            raise RuntimeError('VideoSubFinder did not produce TXTImages. Review the diagnostics archive and VideoSubFinder log.')
        print(f'2/2 RapidVideOCR: processing {txt_count} cleaned frames...', flush=True)
        report['ocr_attempts'] = []

        def execute_ocr(mode, python_path, output_name, log_name):
            output = workspace.path / output_name
            log_path = result / log_name
            attempt_started = time.monotonic()
            attempt = {'compute': mode.upper(), 'python': str(python_path), 'log': str(log_path)}
            try:
                seconds = run_process(
                    [python_path, Path(__file__).with_name('ocr_worker.py'),
                     txt_images, output, '--compute', mode],
                    workspace.path, log_path, args.timeout)
                metrics_path = output / 'ocr-metrics.json'
                if not metrics_path.is_file():
                    raise RuntimeError('The OCR metrics file was not produced.')
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                if mode == 'cuda' and not metrics.get('cuda_primary_all', False):
                    raise RuntimeError(
                        f"CUDA tum OCR oturumlarinda birincil degil: "
                        f"{metrics.get('model_session_providers', {})}")
                attempt.update(status='COMPLETE', seconds=seconds,
                               model_session_providers=metrics.get('model_session_providers', {}))
                report['ocr_attempts'].append(attempt)
                return output, metrics, seconds
            except Exception as exc:
                attempt.update(status='FAILED', seconds=round(time.monotonic() - attempt_started, 3),
                               error=str(exc))
                report['ocr_attempts'].append(attempt)
                raise

        try:
            ocr_output, ocr_metrics, ocr_seconds = execute_ocr(
                args.ocr_compute, ocr_python, 'ocr-primary', 'ocr.log')
            compute['rapidvideocr'] = (
                'CUDA_ACTIVE' if args.ocr_compute == 'cuda' else 'CPU')
        except Exception as primary_error:
            if args.ocr_compute != 'cuda':
                raise
            report['ocr_fallback_reason'] = str(primary_error)
            print('OCR CUDA is unavailable; falling back to the stable CPU path...', flush=True)
            ocr_output, ocr_metrics, ocr_seconds = execute_ocr(
                'cpu', Path(sys.executable), 'ocr-cpu-fallback', 'ocr-cpu-fallback.log')
            compute['rapidvideocr'] = 'CPU_FALLBACK'
            compute['rapidvideocr_selected'] = 'cpu'
            compute['rapidvideocr_python'] = sys.executable

        report['ocr_seconds'] = ocr_seconds
        report['ocr_attempt_seconds'] = round(
            sum(attempt.get('seconds', 0) for attempt in report['ocr_attempts']), 3)
        report['ocr_fusion'] = ocr_metrics
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
        summary = '\n'.join([f'--- SUBHOOPER PIPELINE {package_version} RESULT START ---',
                             f"Pipeline={report['status']}", f"Version={package_version}",
                             f"Client={report.get('client', '')}",
                             f"SRT={report.get('srt', '')}",
                             f"Region={report.get('subtitle_region', '')}",
                             f"Compute={report.get('compute', {}).get('videosubfinder', '')}",
                             f"OCRCompute={report.get('compute', {}).get('rapidvideocr', '')}",
                             f"OCRRequested={report.get('compute', {}).get('rapidvideocr_requested', '')}",
                             f"OCRFallback={bool(report.get('ocr_fallback_reason'))}",
                             f"OCRProviders={','.join(report.get('ocr_fusion', {}).get('session_providers', []))}",
                             f"OCRPrimaryAll={report.get('ocr_fusion', {}).get('cuda_primary_all', False)}",
                             f"OCRMode={report.get('ocr_fusion', {}).get('mode', '')}",
                             f"OCRFusionChanges={report.get('ocr_fusion', {}).get('changed_from_detector', 0)}",
                             f"RGBImages={report.get('rgb_image_count', 0)}",
                             f"TXTImages={report.get('txt_image_count', 0)}",
                             f"OCRSource={report.get('ocr_image_source', '')}",
                             f"Subtitles={subtitle_count}", f"SuspiciousShortCues={suspect_count}",
                             f"Diagnostics={report.get('diagnostic_zip', '')}",
                             f"EmptyDropped={report.get('empty_cues_dropped', 0)}",
                             f"MergedDuplicates={report.get('duplicate_cues_merged', 0)}",
                             f"VSFSeconds={report.get('vsf_seconds', '')}",
                             f"OCRSeconds={report.get('ocr_seconds', '')}",
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








