"""CPU image-formation references; no GT, network loading, or scheduler writes.

Blur uses a small-displacement, locally constant velocity approximation. It is
not a reconstruction of the camera's real exposure from sparse 5 Hz frames.
"""
import math

import numpy as np


def _rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError('Expected RGB uint8 HxWx3 image with both sides >=2')
    return image


def _strength(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Strength must be finite and in [0, 1]')
    return value


def srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float32)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_uint8(value):
    value = np.clip(np.asarray(value, dtype=np.float32), 0, 1)
    srgb = np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def fog(image, depth_metres, strength, beta_max=0.015, airlight_srgb=(0.85, 0.85, 0.85)):
    image = _rgb(image); level = _strength(strength)
    depth = np.asarray(depth_metres, dtype=np.float32)
    airlight = np.asarray(airlight_srgb, dtype=np.float32)
    if depth.shape != image.shape[:2] or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError('Depth must be aligned, finite and nonnegative')
    if not math.isfinite(beta_max) or beta_max < 0:
        raise ValueError('Invalid optical extinction')
    if airlight.shape != (3,) or not np.isfinite(airlight).all() or np.any((airlight < 0) | (airlight > 1)):
        raise ValueError('Airlight must contain three sRGB values in [0, 1]')
    transmission = np.exp(-beta_max * level * depth).astype(np.float32)
    if level == 0 or beta_max == 0:
        return image.copy(), transmission
    source = srgb_to_linear(image.astype(np.float32) / 255)
    atmosphere = srgb_to_linear(airlight)
    result = source * transmission[..., None] + atmosphere * (1 - transmission[..., None])
    return linear_to_uint8(result), transmission


def pixel_grid(height, width):
    yy, xx = np.mgrid[:height, :width]
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def sample_bilinear(image, coordinates):
    """Sample pixel-centred x/y coordinates; report out-of-image samples."""
    source = np.asarray(image, dtype=np.float32)
    xy = np.asarray(coordinates, dtype=np.float32)
    if source.ndim not in (2, 3) or xy.shape != (*source.shape[:2], 2):
        raise ValueError('Sampling grid/image shapes differ')
    if not np.isfinite(source).all() or not np.isfinite(xy).all():
        raise ValueError('Nonfinite sampling input')
    height, width = source.shape[:2]
    x, y = xy[..., 0], xy[..., 1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x = np.clip(x, 0, width - 1); y = np.clip(y, 0, height - 1)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    if source.ndim == 3:
        dx = dx[..., None]; dy = dy[..., None]
    value = ((1 - dx) * (1 - dy) * source[y0, x0] + dx * (1 - dy) * source[y0, x1]
             + (1 - dx) * dy * source[y1, x0] + dx * dy * source[y1, x1])
    return value.astype(np.float32), valid


def bidirectional_confidence(forward, backward, absolute_tolerance=1.0, relative_tolerance=0.05):
    forward = np.asarray(forward, dtype=np.float32); backward = np.asarray(backward, dtype=np.float32)
    if forward.ndim != 3 or forward.shape[-1] != 2 or backward.shape != forward.shape:
        raise ValueError('Need two aligned HxWx2 flow maps')
    if min(absolute_tolerance, relative_tolerance) < 0:
        raise ValueError('Flow tolerances must be nonnegative')
    warped, valid = sample_bilinear(backward, pixel_grid(*forward.shape[:2]) + forward)
    error = np.linalg.norm(forward + warped, axis=-1)
    tolerance = absolute_tolerance + relative_tolerance * (
        np.linalg.norm(forward, axis=-1) + np.linalg.norm(warped, axis=-1))
    return valid & (error <= tolerance), error


def estimate_dis_pair(reference, following):
    """Optional CPU flow backend. Never imports torch or invokes a GPU."""
    reference = _rgb(reference); following = _rgb(following)
    if reference.shape != following.shape:
        raise ValueError('Adjacent RGB shapes differ')
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('DIS flow needs OpenCV in this task runtime; no package is installed automatically') from error
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    first = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(following, cv2.COLOR_RGB2GRAY)
    forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(first, second, None)
    backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(second, first, None)
    confidence, error = bidirectional_confidence(forward, backward)
    return forward, backward, confidence, error


def flow_blur(image, flow, confidence, strength, dt_seconds, exposure_seconds=0.02,
              max_interval_fraction=0.1, max_length_pixels=6.0, samples=9):
    image = _rgb(image); level = _strength(strength)
    flow = np.asarray(flow, dtype=np.float32); confidence = np.asarray(confidence)
    if flow.shape != (*image.shape[:2], 2) or confidence.shape != image.shape[:2] or confidence.dtype != np.bool_:
        raise ValueError('Flow/confidence shape or type differs')
    if not np.isfinite(flow).all():
        raise ValueError('Nonfinite flow')
    if not all(math.isfinite(x) for x in (dt_seconds, exposure_seconds, max_interval_fraction, max_length_pixels)):
        raise ValueError('Nonfinite exposure configuration')
    if dt_seconds <= 0 or exposure_seconds < 0 or not 0 <= max_interval_fraction <= 0.1 or max_length_pixels < 0:
        raise ValueError('Invalid short-exposure bounds')
    if type(samples) is not int or samples < 3 or samples % 2 != 1:
        raise ValueError('An odd number of >=3 exposure samples is required')
    fraction = min(exposure_seconds / dt_seconds, max_interval_fraction) * level
    displacement = flow * fraction
    raw_length = np.linalg.norm(displacement, axis=-1)
    scale = np.minimum(1.0, max_length_pixels / np.maximum(raw_length, 1e-12))
    displacement = displacement * scale[..., None] * confidence[..., None]
    length = np.linalg.norm(displacement, axis=-1)
    diagnostics = {
        'method': 'locally_constant_flow_short_exposure_approximation',
        'dt_seconds': float(dt_seconds), 'exposure_seconds': float(fraction * dt_seconds),
        'valid_flow_fraction': float(confidence.mean()),
        'capped_flow_fraction': float(((raw_length > max_length_pixels) & confidence).mean()),
        'mean_blur_length_pixels': float(length.mean()),
        'max_blur_length_pixels': float(length.max()),
    }
    if not np.any(length):
        return image.copy(), length, diagnostics
    source = srgb_to_linear(image.astype(np.float32) / 255)
    grid = pixel_grid(*image.shape[:2]); total = np.zeros_like(source)
    denominator = np.zeros(image.shape[:2], dtype=np.float32)
    for offset in np.linspace(-0.5, 0.5, samples):
        sampled, valid = sample_bilinear(source, grid - float(offset) * displacement)
        total += sampled * valid[..., None]
        denominator += valid
    # The centre exposure sample is always in bounds. No padding contributes.
    result = total / denominator[..., None]
    diagnostics['partial_exposure_border_fraction'] = float((denominator < samples).mean())
    return linear_to_uint8(result), length, diagnostics


def fuse_depth(current, previous, current_to_previous, confidence, current_weight=0.8,
               relative_tolerance=0.1, absolute_tolerance=0.5):
    """Optional guarded temporal ablation, never replaces the stored raw depth."""
    current = np.asarray(current, dtype=np.float32); previous = np.asarray(previous, dtype=np.float32)
    if current.ndim != 2 or previous.shape != current.shape:
        raise ValueError('Aligned depth maps required')
    if not np.isfinite(current).all() or np.any(current < 0) or np.any(previous < 0):
        raise ValueError('Depth must be finite and nonnegative')
    if not 0 <= current_weight <= 1 or min(relative_tolerance, absolute_tolerance) < 0:
        raise ValueError('Invalid depth fusion bounds')
    confidence = np.asarray(confidence)
    if confidence.shape != current.shape or confidence.dtype != np.bool_:
        raise ValueError('Aligned boolean confidence required')
    warped, valid = sample_bilinear(previous, pixel_grid(*current.shape) + current_to_previous)
    coherent = np.abs(current - warped) <= np.maximum(absolute_tolerance, relative_tolerance * current)
    use = valid & confidence & coherent
    result = current.copy()
    result[use] = current_weight * current[use] + (1 - current_weight) * warped[use]
    return result, use
