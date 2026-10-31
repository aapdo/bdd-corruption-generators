"""Shared geometry, validation and photometry. No I/O or global RNG state."""
import hashlib
import math
import numbers
import numpy as np


def image_shape(shape):
    shape = tuple(shape)
    if len(shape) == 3 and shape[2] == 3:
        shape = shape[:2]
    if len(shape) != 2 or any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 2 for x in shape):
        raise ValueError('image_shape must be (H,W) or (H,W,3), H,W >= 2')
    return tuple(map(int, shape))


def intrinsics(K):
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError('K must be finite 3x3')
    if not np.allclose(K[2], [0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError('K must have calibrated-assumption last row [0,0,1]')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or np.linalg.det(K[:2, :2]) <= 0 or np.linalg.cond(K) > 1e8:
        raise ValueError('unsupported or ill-conditioned intrinsics')
    return K.copy()


def rigid_pose(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('pose must be finite 4x4')
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError('invalid homogeneous pose row')
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5, rtol=0) or abs(np.linalg.det(R) - 1) > 1e-5:
        raise ValueError('pose rotation must be proper and orthonormal')
    return T.copy()


def scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(name + ' must be a finite real scalar')
    return float(value)


def validate_inputs(rgb, depth, K, T, strength):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('rgb must be uint8 HxWx3')
    image_shape(rgb.shape)
    if not isinstance(depth, np.ndarray) or depth.dtype != np.float32 or depth.shape != rgb.shape[:2]:
        raise ValueError('depth must be float32 HxW matching rgb')
    # Nonfinite depth is explicitly allowed: it denotes unknown geometry.
    if np.any(np.isfinite(depth) & (depth < 0)):
        raise ValueError('finite negative depth is not supported')
    intrinsics(K)
    rigid_pose(T)
    s = scalar(strength, 'strength')
    if not 0 <= s <= 1:
        raise ValueError('strength must be in [0,1]')


def depth_valid(depth):
    return np.isfinite(depth) & (depth > 0) & (depth <= 120)


def dilate(mask, radius):
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ya, yb = max(0, dy), min(h, h + dy)
            xa, xb = max(0, dx), min(w, w + dx)
            if ya < yb and xa < xb:
                out[ya:yb, xa:xb] |= mask[ya-dy:yb-dy, xa-dx:xb-dx]
    return out


def depth_boundary(depth, dilation=2):
    """Two-sided depth jumps, including known/unknown interfaces, dilated in pixels."""
    if isinstance(dilation, bool) or not isinstance(dilation, numbers.Integral) or not 0 <= dilation <= 16:
        raise ValueError('dilation must be an integer in [0,16]')
    valid = depth_valid(depth)
    d = np.where(valid, depth, 0).astype(np.float64)
    edges = np.zeros(depth.shape, bool)
    for a, b in [((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))]:
        jump = (valid[a] != valid[b]) | (valid[a] & valid[b] & (np.abs(d[a] - d[b]) > np.maximum(.3, .05 * np.minimum(d[a], d[b]))))
        edges[a] |= jump
        edges[b] |= jump
    return dilate(edges, int(dilation))


def backproject_world(depth, K, T):
    """Camera-z unprojection. Unknown samples return [0,0,0]; always use depth_valid."""
    K, T = intrinsics(K), rigid_pose(T)
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise ValueError('depth must be HxW')
    y, x = np.indices(depth.shape, dtype=np.float64)
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(K).T
    valid = depth_valid(depth)
    points = rays * np.where(valid, depth, 0)[..., None]
    world = (points - T[:3, 3]) @ T[:3, :3]
    world[~valid] = 0
    return world


def reproject_depth(depth, K, source_pose, sample_pose):
    """Observed geometry z-buffer at an exposure pose; holes remain unknown."""
    if np.allclose(source_pose, sample_pose, atol=1e-12, rtol=0):
        return np.where(depth_valid(depth), depth, 0).astype(np.float64)
    h, w = depth.shape
    points = backproject_world(depth, K, source_pose)[depth_valid(depth)]
    cam = points @ sample_pose[:3, :3].T + sample_pose[:3, 3]
    cam = cam[cam[:, 2] > 1e-6]
    result = np.full(h*w, np.inf, np.float64)
    if len(cam):
        projected = cam @ K.T
        uv = np.floor(projected[:, :2] / projected[:, 2:] + .5).astype(np.int64)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        np.minimum.at(result, uv[valid, 1]*w + uv[valid, 0], cam[valid, 2])
    result[~np.isfinite(result)] = 0
    return result.reshape(h, w)


def linearize(rgb):
    x = np.asarray(rgb, dtype=np.float64) / 255
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def encode(linear_rgb):
    x = np.asarray(linear_rgb, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('cannot encode nonfinite radiance')
    x = np.clip(x, 0, 1)
    y = np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
    return np.floor(np.clip(y, 0, 1) * 255 + .5).astype(np.uint8)


def se3_exp(xi):
    """exp([v,w]^), with stable small-angle translation Jacobian."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,) or not np.isfinite(xi).all():
        raise ValueError('SE3 tangent must be a finite 6-vector')
    v, w = xi[:3], xi[3:]
    theta = math.hypot(*w)
    if not math.isfinite(theta):
        raise ValueError('rotation magnitude is unsupported')
    def skew(a):
        x, y, z = a
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    if theta < 1e-6:
        W = skew(w)
        W2 = W @ W
        R = np.eye(3) + (1 - theta**2 / 6) * W + (.5 - theta**2 / 24) * W2
        V = np.eye(3) + (.5 - theta**2 / 24) * W + (1/6 - theta**2 / 120) * W2
    else:
        W = skew(w / theta)
        W2 = W @ W
        R = np.eye(3) + math.sin(theta) * W + (1 - math.cos(theta)) * W2
        V = np.eye(3) + ((1 - math.cos(theta)) / theta) * W + (1 - math.sin(theta) / theta) * W2
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, V @ v
    if not np.isfinite(T).all():
        raise ValueError('SE3 exponential overflow')
    return T


class Clock:
    def __init__(self):
        self.last = None

    def check(self, timestamp):
        t = scalar(timestamp, 'timestamp_s')
        if self.last is None:
            return t, 0.0
        dt = t - self.last
        if dt <= 0 or abs(dt - .2) > .002 + 1e-12:
            raise ValueError('expected a monotonic 0.200 +/- 0.002 second cadence; start a new window')
        return t, dt


def setup(seed, shape, K, defaults, config):
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    c = defaults.copy()
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(c):
            raise ValueError('config must be a dict with only documented keys')
        c.update(config)
    for key, value in c.items():
        scalar(value, key)
    return np.random.default_rng(int(seed)), image_shape(shape), intrinsics(K), c


def integer_config(c, key, low, high):
    value = c[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not low <= value <= high:
        raise ValueError(key + ' is outside its supported integer range')


def digest_arrays(arrays, metadata):
    h = hashlib.sha256(metadata.encode('utf8'))
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()
