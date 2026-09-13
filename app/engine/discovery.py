import os
from pathlib import Path
import shutil


def find_vsf(explicit=None):
    if explicit:
        paths = [Path(explicit)]
    else:
        home = Path.home()
        local_app_data = Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local'))
        paths = [
            local_app_data / 'SubHooper/components/VideoSubFinder-6.10/Release_x64/VideoSubFinderWXW.exe',
            home / 'Downloads/VideoSubFinder_6.10_x64/Release_x64/VideoSubFinderWXW.exe',
            Path('C:/OCR/VideoSubFinder_6.10_x64/Release_x64/VideoSubFinderWXW.exe'),
        ]
        found = shutil.which('VideoSubFinderWXW.exe')
        if found:
            paths.append(Path(found))
        for root in (home / 'Downloads', home / 'Desktop', home / 'Documents'):
            if root.is_dir():
                for directory in sorted(root.glob('*VideoSubFinder*')):
                    paths.extend([directory / 'Release_x64/VideoSubFinderWXW.exe', directory / 'VideoSubFinderWXW.exe'])
        ocr_root = Path('C:/OCR')
        if ocr_root.is_dir():
            for directory in sorted(ocr_root.glob('*VideoSubFinder*')):
                paths.extend([directory / 'Release_x64/VideoSubFinderWXW.exe', directory / 'VideoSubFinderWXW.exe'])
    for path in paths:
        if path.is_file() and path.name.lower() == 'videosubfinderwxw.exe':
            return path.resolve()
    raise RuntimeError('VideoSubFinder is not installed. Open Settings > Components in SubHooper.')
