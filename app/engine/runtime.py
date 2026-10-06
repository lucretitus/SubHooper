import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import tempfile
import time
import unicodedata
import uuid


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


def _terminate_process_tree(process, log, windows=None):
    windows = os.name == 'nt' if windows is None else windows
    if process.poll() is not None:
        return
    if windows:
        try:
            result = subprocess.run(
                ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                stdout=log, stderr=log, timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            if result.returncode != 0 and process.poll() is None:
                process.kill()
        except (OSError, subprocess.SubprocessError):
            # Still stop and reap the direct child if taskkill is unavailable
            # or fails during cancellation.
            if process.poll() is None:
                process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        # A successful taskkill should end the full tree. If a child remains,
        # force-stop and reap this direct process.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=20)


def run_process(command, cwd, log_path, timeout, allowed_codes=(0,), progress_prefix=None):
    started = time.monotonic()
    try:
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(json.dumps([str(v) for v in command], ensure_ascii=False) + '\n')
            log.flush()
            process = subprocess.Popen([str(v) for v in command], cwd=cwd,
                                       stdout=subprocess.PIPE if progress_prefix else log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       text=bool(progress_prefix), encoding='utf-8' if progress_prefix else None,
                                       errors='replace' if progress_prefix else None,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            reader = None
            if progress_prefix:
                def forward_progress():
                    for line in process.stdout:
                        # Native Windows diagnostics can be UTF-16LE inside
                        # an otherwise UTF-8 pipe. Its trailing NUL can prefix
                        # the next Python progress line; normalize before
                        # matching, rather than only after the process exits.
                        line = line.replace('\x00', '')
                        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        log.write(line)
                        log.flush()
                        if line.startswith(progress_prefix):
                            print(line.strip(), flush=True)
                reader = threading.Thread(target=forward_progress, daemon=True)
                reader.start()
            try:
                code = process.wait(timeout=timeout)
                if reader:
                    reader.join(timeout=20)
                log.write(f'\nProcessExitCode={code}\n')
                log.flush()
                if code not in allowed_codes:
                    raise RuntimeError(f'Process code={code}; log={log_path}')
            finally:
                if process.poll() is None:
                    _terminate_process_tree(process, log)
                if reader:
                    reader.join(timeout=20)
                    process.stdout.close()
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
        session_token = os.environ.get('SUBHOOPER_SESSION_TOKEN', '').strip()
        if re.fullmatch(r'[0-9a-f]{32}', session_token):
            (self.path / '.session').write_text(session_token, encoding='ascii')

    def cleanup(self):
        _remove_owned_workspace(self.path, self.base, self.token)


def _remove_owned_workspace(path, base, token):
    base = Path(base).resolve()
    # Check the lexical entry before resolving it: resolving first would hide a
    # symlink or junction at the workspace path itself.
    path = Path(os.path.abspath(path))
    if path.parent != base or not path.name.startswith('subtitle-poc-'):
        raise RuntimeError('The temporary workspace boundary is invalid; deletion was refused.')
    if path.is_symlink() or path.is_junction():
        raise RuntimeError('The temporary workspace is a link; deletion was refused.')
    owner = path / '.owner'
    if owner.is_symlink() or owner.read_text(encoding='ascii') != token:
        raise RuntimeError('Temporary workspace ownership could not be verified.')
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            item = Path(root) / name
            if item.is_symlink() or item.is_junction():
                raise RuntimeError('The temporary workspace contains a link; deletion was refused.')
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
            if (path / '.keep-temp').exists():
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


def cleanup_session_workspaces(session_token, base=None, strict=False):
    """Remove only owned workspaces explicitly tied to a terminated session."""
    if not re.fullmatch(r'[0-9a-f]{32}', str(session_token)):
        return 0
    base = Path(base or tempfile.gettempdir()).resolve()
    cleaned = 0
    for path in base.glob('subtitle-poc-*'):
        matching_session = False
        try:
            if not path.is_dir() or path.is_symlink() or path.is_junction():
                continue
            owner = path / '.owner'
            session = path / '.session'
            if (not owner.is_file() or owner.is_symlink()
                    or not session.is_file() or session.is_symlink()):
                continue
            if (path / '.keep-temp').exists():
                continue
            token = owner.read_text(encoding='ascii').strip()
            if not re.fullmatch(r'[0-9a-f]{32}', token):
                continue
            if session.read_text(encoding='ascii').strip() != session_token:
                continue
            matching_session = True
            _remove_owned_workspace(path, base, token)
            cleaned += 1
        except (OSError, RuntimeError, UnicodeError):
            if strict and matching_session:
                raise
            continue
    return cleaned


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup-session')
    cleanup_args = parser.parse_args()
    if cleanup_args.cleanup_session:
        print(cleanup_session_workspaces(cleanup_args.cleanup_session, strict=True))


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
        'ocr': 'CPU',
    }


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
