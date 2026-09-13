import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import unicodedata
import uuid
import zipfile


def sanitize_log_file(log_path):
    path = Path(log_path)
    try:
        payload = path.read_bytes()
        if b'\x00' not in payload:
            return False
        cleaned = payload.replace(b'\x00', b'')
        cleaned = cleaned.decode('utf-8', errors='replace').encode('utf-8')
        path.write_bytes(cleaned)
        return True
    except OSError:
        return False


def run_process(command, cwd, log_path, timeout, allowed_codes=(0,)):
    started = time.monotonic()
    try:
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(json.dumps([str(v) for v in command], ensure_ascii=False) + '\n')
            log.flush()
            process = subprocess.Popen([str(v) for v in command], cwd=cwd, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                code = process.wait(timeout=timeout)
                log.write(f'\nProcessExitCode={code}\n')
                log.flush()
                if code not in allowed_codes:
                    raise RuntimeError(f'Process code={code}; log={log_path}')
            finally:
                if process.poll() is None:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                       stdout=log, stderr=log, timeout=20,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        process.kill()
                    process.wait(timeout=20)
    finally:
        sanitize_log_file(log_path)
    return round(time.monotonic() - started, 3)


class Workspace:
    def __init__(self):
        self.base = Path(tempfile.gettempdir()).resolve()
        self.path = (self.base / ('subtitle-poc-' + uuid.uuid4().hex)).resolve()
        self.path.mkdir(exist_ok=False)
        self.token = uuid.uuid4().hex
        (self.path / '.owner').write_text(self.token, encoding='ascii')

    def cleanup(self):
        _remove_owned_workspace(self.path, self.base, self.token)


def _remove_owned_workspace(path, base, token):
    path = Path(path).resolve()
    base = Path(base).resolve()
    if path.parent != base or not path.name.startswith('subtitle-poc-'):
        raise RuntimeError('The temporary workspace boundary is invalid; deletion was refused.')
    if path.is_symlink() or path.is_junction():
        raise RuntimeError('Temp baglantisi silinmedi.')
    owner = path / '.owner'
    if owner.is_symlink() or owner.read_text(encoding='ascii') != token:
        raise RuntimeError('Temp sahipligi dogrulanamadi.')
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            item = Path(root) / name
            if item.is_symlink() or item.is_junction():
                raise RuntimeError('Temp icinde baglanti var; silme reddedildi.')
    shutil.rmtree(path)


def cleanup_stale_workspaces(base=None, max_age_seconds=72 * 60 * 60, now=None):
    base = Path(base or tempfile.gettempdir()).resolve()
    now = time.time() if now is None else now
    cleaned = 0
    for path in base.glob('subtitle-poc-*'):
        try:
            if not path.is_dir() or path.is_symlink() or path.is_junction():
                continue
            owner = path / '.owner'
            if not owner.is_file() or owner.is_symlink():
                continue
            token = owner.read_text(encoding='ascii').strip()
            if not re.fullmatch(r'[0-9a-f]{32}', token):
                continue
            if now - owner.stat().st_mtime < max_age_seconds:
                continue
            _remove_owned_workspace(path, base, token)
            cleaned += 1
        except (OSError, RuntimeError, UnicodeError):
            continue
    return cleaned


def select_compute_mode(requested, runner=subprocess.run):
    requested = requested.lower()
    if requested not in {'auto', 'cuda', 'cpu'}:
        raise RuntimeError(f'Invalid compute mode: {requested}')
    nvidia_smi = shutil.which('nvidia-smi')
    names = []
    detector = ''
    if nvidia_smi:
        try:
            probe = runner(
                [nvidia_smi, '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if probe.returncode == 0:
                names = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
                if names:
                    detector = 'nvidia-smi'
        except (OSError, subprocess.SubprocessError):
            names = []
    if not names and os.name == 'nt':
        powershell = shutil.which('powershell.exe') or shutil.which('powershell')
        if powershell:
            try:
                probe = runner(
                    [powershell, '-NoProfile', '-NonInteractive', '-Command',
                     "Get-CimInstance Win32_VideoController | Where-Object Name -Match 'NVIDIA' | Select-Object -ExpandProperty Name"],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                if probe.returncode == 0:
                    names = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
                    if names:
                        detector = 'Win32_VideoController'
            except (OSError, subprocess.SubprocessError):
                names = []
    selected = 'cuda' if requested == 'cuda' or (requested == 'auto' and names) else 'cpu'
    return {
        'requested': requested,
        'selected': selected,
        'nvidia_detected': bool(names),
        'nvidia_gpus': names,
        'nvidia_detector': detector,
        # The VSF CLI can request/disable CUDA but exposes no machine-readable
        # confirmation after its internal fallback. Keep the report honest.
        'videosubfinder': 'CUDA_REQUESTED' if selected == 'cuda' else 'CPU_FORCED',
        'rapidvideocr': 'CPU',
        'rapidvideocr_note': 'The pipeline selects the OCR compute mode separately.',
    }


def archive_diagnostic_sets(image_sets, destination):
    destination = Path(destination)
    allowed = {'.jpeg', '.jpg', '.png', '.bmp'}
    sets = []
    total = 0
    for label, folder in image_sets:
        folder = Path(folder).resolve(strict=True)
        files = sorted(path for path in folder.iterdir()
                       if path.is_file() and not path.is_symlink()
                       and path.suffix.lower() in allowed)
        sets.append((label, files))
        total += len(files)
    if not total:
        raise RuntimeError('No images were found for the diagnostics archive.')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for label, files in sets:
            for path in files:
                archive.write(path, arcname=f'{label}/{path.name}')
    return {'diagnostic_image_count': total,
            'diagnostic_image_sets': {label: len(files) for label, files in sets},
            'diagnostic_zip_bytes': destination.stat().st_size}


def archive_diagnostic_images(images_dir, destination):
    """Backward-compatible single-set helper used by older callers/tests."""
    return archive_diagnostic_sets((('RGBImages', images_dir),), destination)


def normalize_srt(path):
    text = path.read_text(encoding='utf-8-sig')
    timestamp = r'(\d{2,}:\d{2}:\d{2},\d{3}) --> (\d{2,}:\d{2}:\d{2},\d{3})'
    header = re.compile(r'(?m)^[ \t]*(\d+)[ \t]*\r?\n' + timestamp + r'[ \t]*\r?\n')
    matches = list(header.finditer(text))
    if not matches or text[:matches[0].start()].strip():
        raise RuntimeError('Invalid SRT format.')
    previous = -1
    previous_end = -1

    def milliseconds(value):
        h, m, s, ms = map(int, re.split('[:,]', value))
        if m > 59 or s > 59:
            raise RuntimeError('Invalid SRT timecode.')
        return ((h * 60 + m) * 60 + s) * 1000 + ms

    cues = []
    dropped = 0
    merged = 0
    renumbered = 0
    overlaps = 0
    for position, match in enumerate(matches):
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = unicodedata.normalize('NFC', text[match.end():body_end].strip())
        start_text, end_text = match.group(2), match.group(3)
        start, end = map(milliseconds, (start_text, end_text))
        if end <= start or start < previous:
            raise RuntimeError('The SRT cue order is invalid.')
        if previous_end >= 0 and start < previous_end:
            overlaps += 1
        previous = start
        previous_end = end
        if int(match.group(1)) != position + 1:
            renumbered += 1
        if not body:
            dropped += 1
            continue
        normalized_body = ' '.join(body.split()).casefold()
        if (cues and cues[-1]['normalized_body'] == normalized_body
                and start <= cues[-1]['end_ms'] + 750):
            if end > cues[-1]['end_ms']:
                cues[-1]['end_ms'] = end
                cues[-1]['end_text'] = end_text
            merged += 1
            continue
        cues.append({'start_ms': start, 'end_ms': end,
                     'start_text': start_text, 'end_text': end_text,
                     'body': body, 'normalized_body': normalized_body})

    if not cues:
        raise RuntimeError('OCR did not detect text in any frame.')
    serialized = [f"{index}\n{cue['start_text']} --> {cue['end_text']}\n{cue['body']}"
                  for index, cue in enumerate(cues, 1)]
    path.write_text('\n\n'.join(serialized) + '\n', encoding='utf-8', newline='\n')
    suspect_count = sum(len(' '.join(cue['body'].split())) <= 2 for cue in cues)
    suspicious_line_cues = sum(
        any(0 < len(' '.join(line.split())) <= 2 for line in cue['body'].splitlines())
        for cue in cues)
    line_lengths = [len(line) for cue in cues for line in cue['body'].splitlines()]
    return {
        'subtitle_count': len(cues),
        'empty_cues_dropped': dropped,
        'duplicate_cues_merged': merged,
        'suspect_cue_count': suspect_count,
        'suspicious_short_line_cue_count': suspicious_line_cues,
        'srt_format_profile': 'SRT_UTF8_LF_NFC_V1',
        'renumbered_cue_count': renumbered,
        'overlapping_cue_count': overlaps,
        'max_lines_per_cue': max(len(cue['body'].splitlines()) for cue in cues),
        'max_characters_per_line': max(line_lengths, default=0),
    }






