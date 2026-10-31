"""Flare7K* PGS equations with explicitly separate temporal extensions."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def seed(video):
    return int.from_bytes(hashlib.sha256(('flare7kstar-v1:' + video).encode()).digest()[:8], 'big')


def bam(depths, points, width, height, fov_degrees=60):
    """Literal rightmost Eq7. DPT inputs must be distances, not inverse depths."""
    d = np.asarray(depths, np.float64)
    p = np.asarray(points, np.float64)
    if d.ndim != 1 or not len(d) or p.shape != (len(d), 2) or not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError('Finite positive source depths required')
    radial = np.linalg.norm(p - np.array([width / 2, height / 2]), axis=1)
    theta = np.arctan(2 * radial / width * np.tan(np.deg2rad(fov_degrees) / 2))
    factors = (d.mean() / d) ** 2 * np.cos(theta)
    if not np.isfinite(factors).all():
        raise ValueError('Invalid BAM factors')
    return factors.astype(np.float32)


def relative_depth(inverse):
    inverse = np.asarray(inverse, np.float32)
    if inverse.ndim != 2 or not np.isfinite(inverse).all():
        raise ValueError('Invalid inverse depth')
    positive = inverse[inverse > 0]
    if not positive.size:
        raise ValueError('No positive inverse depth')
    floor = max(float(np.percentile(positive, 1)), 1e-6)
    return 1.0 / np.maximum(inverse, floor)


def compose(rgb, field, strength, gamma=2.2):
    if not np.isfinite(strength) or strength < 0 or not np.isfinite(field).all() or np.any(field < 0):
        raise ValueError('Invalid additive radiance')
    if strength == 0:
        return rgb.copy()
    value = (rgb.astype(np.float32) / 255) ** gamma + field * strength
    return np.rint(np.clip(value, 0, 1) ** (1 / gamma) * 255).astype(np.uint8)


def measurements(native, rendered):
    change = rendered.astype(np.float32) - native.astype(np.float32)
    lum = native.astype(np.float32).mean(axis=2) / 255
    return dict(mae_255=float(np.abs(change).mean()),
                changed_fraction=float((change.max(axis=2) >= 8).mean()),
                new_saturation_fraction=float(((rendered.max(axis=2) >= 250) & (native.max(axis=2) < 250)).mean()),
                native_mean_luminance=float(lum.mean()))


def triage(rows, motion_fail_fraction):
    """Descriptive frozen thresholds; never declare a visual/training pass."""
    reasons = []
    if np.median([r['changed_fraction'] for r in rows]) < .003: reasons.append('effect_too_small')
    if max(r['new_saturation_fraction'] for r in rows) > .12: reasons.append('large_new_saturation')
    if np.median([r['native_mean_luminance'] for r in rows]) > .45: reasons.append('daytime_domain_gap')
    if motion_fail_fraction > .2: reasons.append('unstable_background_transport')
    if max(r.get('temporal_field_delta', 0) for r in rows) > .035: reasons.append('large_temporal_field_change')
    return dict(status='REVIEW_REQUIRED' if reasons else 'PRIORITY_FOR_VISUAL_REVIEW', reasons=reasons,
                human_reviewed=False, visual_quality_approved=False, training_release_ready=False)


class VideoFlare:
    def __init__(self, video, bank, root, shape=(720, 1280)):
        rng = np.random.default_rng(seed(video))
        self.h, self.w = shape
        item = bank['patterns'][int(rng.integers(len(bank['patterns'])))]; self.pattern_id = item['id']
        images = {}
        for key in ('compound', 'light_source'):
            with Image.open(Path(root) / item[key]['file']) as im:
                images[key] = np.asarray(im.convert('RGB')).astype(np.float32) / 255
        compound = images['compound'] ** 2.2
        # Flare7K preprocessing removes the dark constant background channelwise.
        compound = np.maximum(compound - compound.min(axis=(0, 1), keepdims=True), 0)
        light = images['light_source'].mean(axis=-1)
        mask = (light > max(float(light.max()) * .5, .05)).astype(np.float32)
        if not mask.any() or compound.max() < .01: raise ValueError('Empty official pattern')
        yy, xx = np.indices(mask.shape)
        origin = np.array([(xx * mask).sum() / mask.sum(), (yy * mask).sum() / mask.sum()])
        self.templates = []; self.points = []; self.initial = []
        count = int(rng.integers(1, 4))
        for _ in range(count):
            point = np.array([rng.uniform(.08, .92) * self.w, rng.uniform(.08, .80) * self.h], np.float32)
            angle = float(rng.uniform(-180, 180)); scale = float(rng.uniform(.5, 1.2) * self.h / mask.shape[0])
            matrix = cv2.getRotationMatrix2D(tuple(origin), angle, scale).astype(np.float32)
            matrix[:, 2] += point - origin
            self.templates.append((compound, mask, matrix))
            self.points.append(point)
            self.initial.append(dict(xy=point.tolist(), angle_degrees=angle, scale=scale))
        self.points = np.array(self.points)
        self.transport = np.eye(3, dtype=np.float32)
        self.prior_gray = None; self.depth_ema = None; self.factor_ema = None; self.prior_field = None
        self.motion_failures = 0; self.transitions = 0

    def _motion(self, gray):
        delta = np.eye(3, dtype=np.float32); state = 'first_frame'
        if self.prior_gray is not None:
            self.transitions += 1; state = 'transport_failed_identity_increment'
            corners = cv2.goodFeaturesToTrack(self.prior_gray, 250, .01, 8)
            if corners is not None and len(corners) >= 12:
                nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.prior_gray, gray, corners, None)
                if nxt is not None:
                    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.prior_gray, nxt, None)
                    if back is not None:
                        keep = ok.ravel().astype(bool) & back_ok.ravel().astype(bool) & (np.linalg.norm(back - corners, axis=2).ravel() < 1.5)
                        if keep.sum() >= 12:
                            matrix, inliers = cv2.estimateAffinePartial2D(corners[keep], nxt[keep], method=cv2.RANSAC, ransacReprojThreshold=2)
                            if matrix is not None and inliers is not None:
                                scale = np.linalg.norm(matrix[:, 0]); fraction = float(inliers.mean())
                                if .9 <= scale <= 1.1 and fraction >= .5 and np.linalg.norm(matrix[:, 2]) < 50:
                                    # Flow operates at 640x360. Convert translation to native pixels.
                                    delta[:2] = matrix; delta[:2, 2] *= self.w / gray.shape[1]
                                    state = 'background_partial_affine'
            if state != 'background_partial_affine': self.motion_failures += 1
        self.prior_gray = gray
        self.transport = delta @ self.transport
        return state

    def field(self, rgb, depth):
        if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError('Aligned positive distance map required')
        gray = cv2.cvtColor(cv2.resize(rgb, (640, 360)), cv2.COLOR_RGB2GRAY)
        state = self._motion(gray)
        moved = []; masks = []; points = []; depths = []
        for base, mask, matrix in self.templates:
            affine = (self.transport @ np.vstack([matrix, [0, 0, 1]]))[:2].astype(np.float32)
            layer = cv2.warpAffine(base, affine, (self.w, self.h), flags=cv2.INTER_LINEAR)
            support = cv2.warpAffine(mask, affine, (self.w, self.h), flags=cv2.INTER_NEAREST) > .5
            moved.append(layer); masks.append(support)
            if support.any():
                yy, xx = np.nonzero(support); points.append([float(xx.mean()), float(yy.mean())]); depths.append(float(depth[support].mean()))
            else:
                # Offscreen source remains in the trajectory and is explicitly flagged.
                p = self.transport @ np.r_[self.points[len(points)], 1]
                points.append(p[:2].tolist()); depths.append(float(np.median(depth)))
        d = np.array(depths, np.float32)
        self.depth_ema = d if self.depth_ema is None else .8 * self.depth_ema + .2 * d
        factors = bam(self.depth_ema, points, self.w, self.h)
        self.factor_ema = factors if self.factor_ema is None else .8 * self.factor_ema + .2 * factors
        field = np.zeros_like(rgb, dtype=np.float32)
        for layer, factor in zip(moved, self.factor_ema): field += layer * factor
        delta = 0 if self.prior_field is None else float(np.abs(field - self.prior_field).mean())
        self.prior_field = field
        return field, dict(pattern_id=self.pattern_id, source_count=len(moved), source_origin='synthetically_inserted',
                           xy=points, relative_source_depths=self.depth_ema.tolist(), bam_factors=self.factor_ema.tolist(),
                           visible_source_masks=[bool(m.any()) for m in masks], motion_state=state, temporal_field_delta=delta)
