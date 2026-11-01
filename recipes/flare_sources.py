"""Photometric source candidates and per-source motion; no semantic/API model."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SourceConfig:
    max_candidates: int = 16
    max_sources: int = 3
    confirm_frames: int = 3
    minimum_observations: int = 10
    maximum_gap: int = 2
    min_area: int = 3
    max_area_fraction: float = .003
    minimum_contrast: float = .12
    forward_backward_pixels: float = 2.0


def detect_sources(rgb, config=None):
    """Bright compact peaks can be lamps or reflections; identity stays unverified."""
    cfg = config or SourceConfig()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('RGB uint8 image required')
    h, w = rgb.shape[:2]
    value = rgb.astype(np.float32) / 255
    red, green, blue = np.moveaxis(value, -1, 0)
    luminance = .2126 * red + .7152 * green + .0722 * blue
    daylight = float(np.median(luminance)) > .32
    if daylight:
        # Daytime sky gaps and sunlit paint are usually not small saturated peaks.
        neutral = (luminance > .975) & (value.min(axis=2) > .94) & (value.max(axis=2) > .995)
    else:
        neutral = (luminance > .88) | ((value.max(axis=2) > .995) & (luminance > .70))
    red_light = ((red > (.985 if daylight else .94)) &
                 (red - green > .25) & (red - blue > .25))
    mask = cv2.morphologyEx((neutral | red_light).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        if not cfg.min_area <= area <= max(12, cfg.max_area_fraction * h * w):
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 6 or area / (bw*bh) < .25:
            continue
        cy, cx = np.nonzero(labels[y:y+bh, x:x+bw] == label)
        cy, cx = cy + y, cx + x
        is_red = bool(red_light[cy, cx].mean() > .5)
        signal = red if is_red else luminance
        radius = max(1.0, float(np.sqrt(area / np.pi)))
        border = int(max(6, 2.5 * radius))
        x0, x1 = max(0, x-border), min(w, x+bw+border)
        y0, y1 = max(0, y-border), min(h, y+bh+border)
        ring = np.ones((y1-y0, x1-x0), bool)
        ring[max(0,y-y0):y+bh-y0, max(0,x-x0):x+bw-x0] = False
        ambient = float(np.median(signal[y0:y1, x0:x1][ring])) if ring.any() else 1.0
        peak = float(np.quantile(signal[cy, cx], .8))
        contrast = peak - ambient
        if contrast < cfg.minimum_contrast or peak / (ambient + .02) < 1.18:
            continue
        weight = np.maximum(signal[cy, cx] - ambient, .001)
        point = [float(np.average(cx, weights=weight)), float(np.average(cy, weights=weight))]
        color = value[cy, cx].mean(axis=0)
        candidates.append(dict(xy=point, radius=radius, area=area,
            bbox=[x,y,x+bw,y+bh], peak=peak, contrast=contrast,
            color=color.tolist(), evidence='compact_red_peak' if is_red else 'compact_bright_peak',
            score=float(contrast * np.sqrt(area)), identity_verified=False))
    candidates.sort(key=lambda d: d['score'], reverse=True)
    retained = []
    for candidate in candidates:
        if all(np.linalg.norm(np.array(candidate['xy'])-c['xy']) > max(8, candidate['radius'], c['radius']) for c in retained):
            retained.append(candidate)
        if len(retained) >= cfg.max_candidates:
            break
    return retained


class SourceTracker:
    """Local LK prediction + current-frame peak association, with explicit loss."""
    def __init__(self, config=None):
        self.cfg = config or SourceConfig()
        self.previous = None
        self.tracks = {}
        self.next_id = 1
        self.frame = -1
        self.observations = []

    def _predict(self, gray, track):
        point = np.asarray(track['last']['xy'], np.float32)
        if self.previous is None:
            return point, None
        h, w = gray.shape
        radius = int(np.clip(3 * track['last']['radius'], 12, 40))
        x, y = np.rint(point).astype(int)
        roi = np.zeros_like(gray)
        roi[max(0,y-radius):min(h,y+radius+1), max(0,x-radius):min(w,x+radius+1)] = 255
        corners = cv2.goodFeaturesToTrack(self.previous, 24, .01, 3, mask=roi)
        if corners is None or len(corners) < 2:
            corners = point[None,None,:]
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, corners, None, winSize=(31,31), maxLevel=3)
        if nxt is None:
            return point, None
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None, winSize=(31,31), maxLevel=3)
        if back is None:
            return point, None
        error = np.linalg.norm(back-corners, axis=2).ravel()
        keep = ok.ravel().astype(bool) & back_ok.ravel().astype(bool) & (error <= self.cfg.forward_backward_pixels)
        if not keep.any():
            return point, None
        displacements = (nxt-corners).reshape(-1,2)[keep]
        median = np.median(displacements, axis=0)
        coherent = np.linalg.norm(displacements-median, axis=1) < 4
        if not coherent.any() or np.linalg.norm(median) > 100:
            return point, None
        return point + np.median(displacements[coherent], axis=0), float(np.median(error[keep][coherent]))

    def step(self, rgb):
        self.frame += 1
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        candidates = detect_sources(rgb, self.cfg)
        predictions = {key:self._predict(gray, t) for key,t in self.tracks.items() if t['gap'] <= self.cfg.maximum_gap}
        edges = []
        for key, (prediction, fb) in predictions.items():
            old = self.tracks[key]['last']
            for index, candidate in enumerate(candidates):
                distance = float(np.linalg.norm(prediction - candidate['xy']))
                gate = max(10, min(32, 2.5*old['radius'])) if fb is not None else max(5, min(12,old['radius']))
                ratio = candidate['radius']/old['radius']
                color_difference = np.linalg.norm(np.asarray(candidate['color'])-old['color'])
                if distance <= gate and .35 <= ratio <= 2.8 and color_difference < .65:
                    edges.append((distance/gate + .15*abs(np.log(ratio)), key, index, distance, fb))
        used_tracks, used_candidates, current = set(), set(), []
        for _, key, index, distance, fb in sorted(edges):
            if key in used_tracks or index in used_candidates:
                continue
            track = self.tracks[key]
            candidate = dict(candidates[index])
            consecutive = track['streak']+1 if track['gap'] == 0 else 1
            track.update(last=candidate, gap=0, streak=consecutive, count=track['count']+1)
            track['frames'].append(self.frame)
            used_tracks.add(key); used_candidates.add(index)
            current.append(dict(candidate, track_id=key, consecutive=consecutive,
                association_error_pixels=distance, forward_backward_error_pixels=fb,
                confirmed=consecutive >= self.cfg.confirm_frames))
        for key, track in self.tracks.items():
            if key not in used_tracks:
                track['gap'] += 1
                track['streak'] = 0
        for index, candidate in enumerate(candidates):
            if index in used_candidates:
                continue
            # A temporarily missing nearby track is not replaced by a new identity.
            if any(t['gap'] <= self.cfg.maximum_gap and np.linalg.norm(np.asarray(t['last']['xy'])-candidate['xy']) < max(8,t['last']['radius']) for t in self.tracks.values()):
                continue
            key = self.next_id; self.next_id += 1
            self.tracks[key] = dict(last=dict(candidate), gap=0, streak=1, count=1, frames=[self.frame])
            current.append(dict(candidate, track_id=key, consecutive=1, association_error_pixels=None,
                forward_backward_error_pixels=None, confirmed=False))
        self.previous = gray
        result = dict(frame_index=self.frame, candidates=len(candidates), sources=current,
            lost_track_ids=[key for key,t in self.tracks.items() if t['gap'] == 1])
        self.observations.append(result)
        return result

    def finalize(self, reference_anchors=None, reference_frame=None):
        eligible = [(key,t) for key,t in self.tracks.items() if t['count'] >= self.cfg.minimum_observations]
        anchor_matches=[]
        if reference_anchors is not None:
            if reference_frame is None or not 0 <= reference_frame < len(self.observations):
                raise ValueError('Reference frame inside this clip required')
            admitted=set()
            for anchor in reference_anchors:
                choices=[(float(np.linalg.norm(np.asarray(s['xy'])-anchor)),s['track_id'])
                    for s in self.observations[reference_frame]['sources']]
                if choices:
                    distance,key=min(choices)
                    if distance <= 8:
                        admitted.add(key);anchor_matches.append(dict(reference_xy=anchor,track_id=key,error_pixels=distance))
            eligible=[(key,t)for key,t in eligible if key in admitted]
        eligible.sort(key=lambda row: (-row[1]['count'], -row[1]['last']['score'], row[0]))
        selected = {key for key,_ in eligible[:self.cfg.max_sources]}
        rows = []
        for frame in self.observations:
            active = [dict(s, fade=min(1., (s['consecutive']-self.cfg.confirm_frames+1)/3))
                      for s in frame['sources'] if s['track_id'] in selected and s['confirmed']]
            rows.append(dict(frame, active_sources=active, effect_applied=bool(active),
                status='tracked_source_available' if active else 'no_supported_source',
                source_identity_semantically_verified=False))
        return rows, dict(selected_track_ids=sorted(selected), total_track_ids=len(self.tracks),
            active_frames=sum(bool(row['active_sources']) for row in rows),
            total_frames=len(rows), tracking='per_source_local_LK_FB_and_peak_association',
            reference_frame=reference_frame, reference_anchor_matches=anchor_matches,
            reference_selection='visually_checked_reference_positions' if reference_anchors is not None else 'photometric_only',
            actual_emitter_or_reflection_identity_verified=False)

