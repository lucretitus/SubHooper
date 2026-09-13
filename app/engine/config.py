REGION_PROFILES = {
    # Values are VideoSubFinder offsets expressed from the image bottom.
    'bottom': {'top': '0.42', 'bottom': '0.02', 'left': '0.03', 'right': '0.97'},
    'lower-half': {'top': '0.55', 'bottom': '0', 'left': '0.02', 'right': '0.98'},
    'full': {'top': '1', 'bottom': '0', 'left': '0', 'right': '1'},
}


def validate_region_offsets(top, bottom, left, right):
    values = {
        'top': float(top),
        'bottom': float(bottom),
        'left': float(left),
        'right': float(right),
    }
    if not all(0 <= value <= 1 for value in values.values()):
        raise RuntimeError('Subtitle region bounds must be between 0 and 1.')
    if values['top'] <= values['bottom'] or values['top'] - values['bottom'] < 0.08:
        raise RuntimeError('The subtitle region height is too small or invalid.')
    if values['right'] <= values['left'] or values['right'] - values['left'] < 0.08:
        raise RuntimeError('The subtitle region width is too small or invalid.')
    return {key: format(value, '.6f').rstrip('0').rstrip('.') for key, value in values.items()}


def get_region_profile(name):
    try:
        return REGION_PROFILES[name]
    except KeyError as exc:
        choices = ', '.join(REGION_PROFILES)
        raise RuntimeError(f'Invalid subtitle region: {name}. Options: {choices}') from exc


def build_vsf_command(executable, video, output, region_name, compute_mode='cpu', region_offsets=None):
    region = (validate_region_offsets(**region_offsets)
              if region_name == 'custom' and region_offsets else get_region_profile(region_name))
    command = [executable, '-r', '-ccti', '-ovffmpeg', '-i', video, '-o', output,
               '-te', region['top'], '-be', region['bottom'],
               '-le', region['left'], '-re', region['right'],
               '-nthr', '2', '-nocrthr', '2']
    if compute_mode == 'cuda':
        command.append('-uc')
    elif compute_mode == 'cpu':
        # VideoSubFinder exposes settings keys as command-line options.
        command.extend(('-use_cuda_gpu', '0'))
    else:
        raise RuntimeError(f'Invalid compute mode: {compute_mode}')
    return command
