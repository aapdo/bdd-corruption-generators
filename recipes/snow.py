"""Persistent world-space snow. NumPy/CPU only; no calibrated weather claim."""
import json
import math
import numpy as np
from common import (Clock, depth_valid, digest_arrays, dilate, encode,
                    integer_config, linearize, reproject_depth, rigid_pose,
                    se3_exp, setup, validate_inputs)


SNOW_DEFAULTS = {
    'particle_count': 50000,
    'box_half_extent': 12.0,
    'near': .5,
    'far': 8.0,
    'radius_min': .008,
    'radius_max': .022,
    'fall_speed': 1.0,
    'wind_speed': .3,
    'flutter_amplitude': .012,
    'exposure_s': .010,
    'exposure_samples': 5,
    'flake_opacity': .85,
    'haze_beta': .00002,
    'boundary_dilation': 2,
    'max_projected_radius': 24.0,
}


def _confidence(valid, radius):
    """Smooth only known/unknown interfaces, never valid surface jumps."""
    confidence = valid.astype(np.float64)
    reached = ~valid
    for distance in range(1, radius + 1):
        expanded = dilate(reached, 1)
        band = expanded & ~reached & valid
        x = distance / (radius + 1)
        confidence[band] = x*x*(3-2*x)
        reached = expanded
    return confidence


def _conservative_depth(depth):
    """One-pixel foreground guard, without declaring valid edges unknown."""
    z = np.where(depth > 0, depth, np.inf)
    padded = np.pad(z, 1, constant_values=np.inf)
    h, w = depth.shape
    result = z.copy()
    for y in range(3):
        for x in range(3):
            np.minimum(result, padded[y:y+h, x:x+w], out=result)
    return np.where(depth > 0, result, 0)


class SnowRenderer:
    def __init__(self, seed, image_shape, intrinsics, config=None):
        self.rng, self.shape, self.K, self.config = setup(
            seed, image_shape, intrinsics, SNOW_DEFAULTS, config)
        c = self.config
        integer_config(c, 'particle_count', 1, 1000000)
        integer_config(c, 'exposure_samples', 1, 21)
        integer_config(c, 'boundary_dilation', 0, 16)
        if not (0 < c['near'] < c['far'] <= 120 and
                0 < c['radius_min'] <= c['radius_max'] < c['near']/2):
            raise ValueError('invalid snow distance/radius bounds')
        if not (c['box_half_extent'] > 0 and 0 <= c['fall_speed'] <= 10 and
                abs(c['wind_speed']) <= 10 and 0 <= c['flutter_amplitude'] <= .1 and
                0 <= c['exposure_s'] <= .1 and 0 < c['flake_opacity'] <= 1 and
                0 <= c['haze_beta'] <= .1 and 0 < c['max_projected_radius'] <= 64):
            raise ValueError('unsupported snow configuration')
        h, w = self.shape
        corners = np.array([[-.5, -.5, 1], [w-.5, -.5, 1],
                            [-.5, h-.5, 1], [w-.5, h-.5, 1]])
        rays = corners @ np.linalg.inv(self.K).T
        speed = math.hypot(1.2*c['fall_speed'], c['wind_speed'])
        # The sphere covers every supported footprint, including exposure motion.
        self.guard = (np.linalg.norm(rays, axis=1).max()*c['far'] +
                      c['radius_max'] + math.sqrt(2)*c['flutter_amplitude'] +
                      speed*c['exposure_s']/2 + .05)
        if self.guard >= c['box_half_extent']:
            raise ValueError('supported frustum does not fit population box')
        projected = np.linalg.norm(self.K[:2, :2], axis=1).max()*c['radius_max']/c['near']
        if projected > c['max_projected_radius']:
            raise ValueError('physical radii exceed supported projected footprint')
        self.clock = Clock()
        self._base = None
        self._ids = np.empty(0, dtype=np.int64)
        self._origin_time = None
        self._time = 0.0
        self._next_id = 0

    def _attributes(self, idx):
        n, c = len(idx), self.config
        self.radii[idx] = self.rng.uniform(c['radius_min'], c['radius_max'], n)
        self.threshold[idx] = self.rng.random(n)
        self.angle[idx] = self.rng.uniform(0, 2*np.pi, n)
        self.ratio[idx] = self.rng.uniform(.55, 1, n)
        self.phase[idx] = self.rng.uniform(0, 2*np.pi, (n, 2))
        self.frequency[idx] = self.rng.uniform(3, 5, n)
        fall = c['fall_speed']*self.rng.uniform(.8, 1.2, n)
        self.velocity[idx] = fall[:, None]*self.down + c['wind_speed']*self.right

    def _initialize(self, T, timestamp):
        n, half = self.config['particle_count'], self.config['box_half_extent']
        R, t = T[:3, :3], T[:3, 3]
        self._box_center = -R.T @ t
        self.right, self.down, self.forward = R.T[:, 0], R.T[:, 1], R.T[:, 2]
        self._base = self._box_center + self.rng.uniform(-half, half, (n, 3))
        self._ids = np.arange(n, dtype=np.int64)
        self._next_id = n
        self.radii, self.threshold, self.angle, self.ratio, self.frequency = [np.empty(n) for _ in range(5)]
        self.phase, self.velocity = np.empty((n, 2)), np.empty((n, 3))
        self._attributes(np.arange(n))
        self._origin_time = timestamp

    def _flutter(self, t):
        phase = self.phase + self.frequency[:, None]*t
        return self.config['flutter_amplitude']*(np.sin(phase[:, :1])*self.right +
                                                np.sin(phase[:, 1:])*self.forward)

    @property
    def positions(self):
        return np.empty((0, 3)) if self._base is None else (self._base + self._flutter(self._time)).copy()

    @property
    def particle_ids(self):
        return self._ids.copy()

    @property
    def activation_thresholds(self):
        return np.empty(0) if self._base is None else self.threshold.copy()

    def state_fingerprint(self):
        arrays = [] if self._base is None else [self._base, self._ids, self.velocity,
                  self.radii, self.threshold, self.angle, self.ratio, self.phase,
                  self.frequency, self._box_center, self.right, self.down, self.forward]
        metadata = json.dumps({'rng': self.rng.bit_generator.state, 'time': self._time,
                               'last': self.clock.last, 'origin': self._origin_time,
                               'next_id': self._next_id, 'config': self.config}, sort_keys=True)
        return digest_arrays(arrays, metadata)

    def _advance(self, center, elapsed, guard):
        """Recycle only outside the entire observed sphere, never screen-space."""
        half = self.config['box_half_extent']
        box = self._box_center
        if np.max(np.abs(center-box)) + guard >= half:
            box = center.copy()
        base = self._base + self.velocity*elapsed
        idx = np.flatnonzero(np.any(np.abs(base-box) > half, axis=1))
        new = box + (base[idx]-box+half) % (2*half) - half
        if len(idx) and np.any(np.linalg.norm(base[idx]-center, axis=1) <= guard):
            raise ValueError('camera displacement exceeds guarded recycling support')
        # A moving box can wrap an unseen particle into the guarded sphere.
        # Place those births on the box face furthest from the current camera;
        # this guarantees outside-frustum birth without rejecting normal motion.
        bad=np.linalg.norm(new-center,axis=1)<=guard
        if bad.any():
            axis=int(np.argmax(np.abs(center-box)))
            sign=-1 if center[axis]>box[axis] else 1
            new[bad]=box+self.rng.uniform(-half,half,(int(bad.sum()),3))
            new[bad,axis]=box[axis]+sign*(half-1e-6)
            if np.any(np.linalg.norm(new[bad]-center,axis=1)<=guard):raise ValueError('Cannot place outside visible guard')
        base[idx] = new
        self._base, self._box_center = base, box
        self._ids[idx] = np.arange(self._next_id, self._next_id+len(idx))
        self._next_id += len(idx)
        self._attributes(idx)
        return idx

    def render(self, rgb, depth, T_world_to_camera, strength, timestamp_s, camera_twist=None):
        validate_inputs(rgb, depth, self.K, T_world_to_camera, strength)
        if rgb.shape[:2] != self.shape:
            raise ValueError('frame shape differs from constructor')
        T = rigid_pose(T_world_to_camera)
        timestamp, elapsed = self.clock.check(timestamp_s)
        c, h, w = self.config, *self.shape
        twist = None
        if camera_twist is not None:
            twist = np.asarray(camera_twist, dtype=np.float64)
            if twist.shape != (6,) or not np.isfinite(twist).all():
                raise ValueError('camera_twist must be a finite [v,w] 6-vector')
        offsets = (np.array([0.0]) if c['exposure_s'] == 0 or c['exposure_samples'] == 1
                   else np.linspace(-c['exposure_s']/2, c['exposure_s']/2, c['exposure_samples']))
        motion = 0 if twist is None else math.hypot(*twist[:3])*c['exposure_s']/2
        guard = self.guard + motion
        if not math.isfinite(guard) or guard >= c['box_half_extent']:
            raise ValueError('exposure motion exceeds enclosing guard capacity')
        poses = [T if twist is None else se3_exp(twist*dt) @ T for dt in offsets]
        first = self._base is None
        if first:
            self._initialize(T, timestamp)
        center = -T[:3, :3].T @ T[:3, 3]
        respawned = self._advance(center, elapsed, guard)
        self._time = timestamp-self._origin_time
        self.clock.last = timestamp
        valid = depth_valid(depth)
        confidence = _confidence(valid, c['boundary_dilation'])
        active = np.flatnonzero(self.threshold < float(strength))
        projected = np.zeros(len(self._ids), bool)
        covered, visible = projected.copy(), projected.copy()
        particle = np.zeros(self.shape, np.float64)
        for pose, dt in zip(poses, offsets):
            sample = np.zeros(self.shape, np.float64)
            sample_depth = _conservative_depth(reproject_depth(depth, self.K, T, pose))
            points = self._base[active] + self.velocity[active]*dt + self._flutter(self._time+dt)[active]
            cam = points @ pose[:3, :3].T + pose[:3, 3]
            good = (cam[:, 2] >= c['near']) & (cam[:, 2] <= c['far'])
            for j in np.flatnonzero(good):
                i = active[j]
                z, radius = cam[j, 2], self.radii[i]
                q = self.K @ cam[j]
                u, v = q[:2]/z
                angle = self.angle[i] + .15*math.sin(self.phase[i, 0]+self.frequency[i]*(self._time+dt))
                ca, sa = math.cos(angle), math.sin(angle)
                rotation = np.array([[ca, -sa], [sa, ca]])
                footprint = self.K[:2, :2] @ rotation @ np.diag([radius, radius*self.ratio[i]])/(math.sqrt(2)*z)
                covariance = footprint @ footprint.T + np.eye(2)/12
                inverse = np.linalg.inv(covariance)
                bx, by = 3*np.sqrt(np.diag(covariance))
                xa, xb = int(math.ceil(u-bx)), int(math.floor(u+bx))
                ya, yb = int(math.ceil(v-by)), int(math.floor(v+by))
                x0, x1, y0, y1 = max(0, xa), min(w-1, xb), max(0, ya), min(h-1, yb)
                if x0 > x1 or y0 > y1:
                    continue
                projected[i] = True
                yy, xx = np.mgrid[ya:yb+1, xa:xb+1]
                delta = np.stack((xx-u, yy-v), -1)
                r2 = np.einsum('...i,ij,...j->...', delta, inverse, delta)
                weights = np.exp(-.5*r2)*(r2 < 9)
                normalization = float(weights.sum())
                weights = weights[y0-ya:y1-ya+1, x0-xa:x1-xa+1]
                region = np.s_[y0:y1+1, x0:x1+1]
                usable = (weights > 0) & valid[region] & (sample_depth[region] > 0)
                covered[i] |= bool(usable.any())
                take = usable & (z-radius < sample_depth[region]-.01)
                visible[i] |= bool(take.any())
                area = math.pi*np.linalg.det(self.K[:2, :2])*(radius/z)**2*self.ratio[i]
                # Normalize before cropping or occlusion: no edge energy boost.
                alpha = np.clip(c['flake_opacity']*area*weights/normalization, 0, 1)*take
                sample[region] = 1-(1-sample[region])*(1-alpha)
            particle += sample/len(offsets)
        particle = np.minimum(particle, .85)*confidence
        safe_depth = np.where(valid, depth, 0).astype(np.float64)
        haze = -np.expm1(-c['haze_beta']*float(strength)*safe_depth)*confidence
        transmission = 1-haze
        linear = linearize(rgb)
        airlight = linearize(np.array([.88, .92, .96])*255)
        mixed = (linear*(1-particle[..., None])+particle[..., None])*transmission[..., None] + airlight*haze[..., None]
        out = rgb.copy() if float(strength) == 0 else encode(mixed)
        out[~valid] = rgb[~valid]
        total = 1-(1-particle)*transmission
        pixels = particle > .05
        count = int(pixels.sum())
        qa = {
            'active_flakes': int(len(active)), 'projected_flakes': int(projected.sum()),
            'occluded_flakes': int((covered & ~visible).sum()), 'visible_flakes': int(visible.sum()),
            'respawn_count': int(len(respawned)), 'respawns_in_visible_frustum': 0,
            'initialized_this_frame': bool(first), 'particle_pixel_fraction': float(pixels.mean()),
            'particle_pixel_threshold': .05,
            'particle_left_fraction': float(pixels[:, :w//2].sum()/count) if count else 0.0,
            'mean_particle_alpha': float(particle.mean()), 'mean_haze_opacity': float(haze.mean()),
            'mean_alpha': float(total.mean()), 'max_alpha': float(total.max()),
            'haze_transmission_range': [float(transmission.min()), float(transmission.max())],
            'unknown_preserved': bool(np.array_equal(out[~valid], rgb[~valid])),
            'valid_boundaries_native_protected': False,
            'population_density_per_m3': float(c['particle_count']/(2*c['box_half_extent'])**3),
            'physical_radius_range_m': [float(c['radius_min']), float(c['radius_max'])],
            'supported_camera_z_m': [float(c['near']), float(c['far'])],
            'exposure_samples': int(len(offsets)),
            'camera_motion_approximation': 'center_only_missing_twist' if twist is None else 'constant_left_twist',
            'exposure_approximation': 'center_only' if len(offsets) == 1 else 'uniform_temporal_quadrature',
            'haze_is_approximation': True, 'calibrated_weather_rate': False,
        }
        return {'rgb': out, 'masks': {'particle_alpha': particle.astype(np.float32),
                                     'haze_transmission': transmission.astype(np.float32)}, 'qa': qa}
