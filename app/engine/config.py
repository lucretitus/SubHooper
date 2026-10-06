REGION_PROFILES = {
    # Vertical bounds are offsets measured from the bottom of the frame.
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
