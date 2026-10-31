"""Conservative CPU-only first-pass rules; no training or human verification."""
import numpy as np
import cv2
from collections import deque
import hashlib
import math

POLICY = {
    'version': '0.1.0', 'sample_cadence_s': 0.2, 'temporal_window': 5,
    'min_hits': 4, 'threshold_status': 'starting_uncalibrated',
    'thresholds': {'core_linear_peak': 0.65, 'ring_contrast': 0.20,
                   'max_vehicle_core_area_fraction': 0.015,
                   'lk_fb_px': 1.0, 'lk_centroid_px': 3.0},
    'geometry_rule': 'paired_opposite_sides_v1',
    'verification_scope': 'automatic_starting_rules_only'
}
VEHICLES = {'car', 'bus', 'truck', 'train', 'motorcycle'}


def _image(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3:
        raise ValueError('expected uint8 HxWx3 RGB')


def _attrs(attributes):
    if attributes is not None and not isinstance(attributes, dict):
        raise ValueError('attributes must be dict or None')
    a = attributes or {}
    for k in ('timeofday', 'weather', 'scene'):
        if k in a and not isinstance(a[k], str):
            raise ValueError('scene labels must be strings')
    for k in ('sky_verified', 'approved_controlled_assumption'):
        if k in a and type(a[k]) is not bool:
            raise ValueError('explicit flags must be bool')
    d = a.get('sun_direction_confidence')
    if d is not None and (type(d) not in (int, float) or not math.isfinite(d) or not 0 <= d <= 1):
        raise ValueError('invalid external direction confidence')
    p = a.get('sun_direction_xy')
    if p is not None and (not isinstance(p, (list, tuple)) or len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)):
        raise ValueError('invalid external direction')
    return a


def _sky(rgb):
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float32) / 255
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = cv2.absdiff(g, cv2.blur(g, (5, 5)))
    mask = ((f[:, :, 2] > 0.4) & (f[:, :, 2] > f[:, :, 0] * 1.12) & (texture < 9))
    mask[int(h * 0.6):] = False
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    ids = np.unique(labels[0]); ids = ids[ids != 0]
    return np.isin(labels, ids) if n > 1 else np.zeros((h, w), bool)


def scene_evidence(rgb, attributes=None):
    _image(rgb)
    a = _attrs(attributes)
    s = {k: a.get(k, 'unknown').lower() for k in ('timeofday', 'weather', 'scene')}
    s.update(rgb_day_night_hint='night' if float(rgb.mean()) < 40 else 'day',
             rgb_hint_basis='heuristic_not_scene_truth',
             sky_candidate_fraction=float(_sky(rgb).mean()),
             sky_candidate_basis='top_connected_blue_low_texture_upper_mask_not_verified_segmentation',
             sky_verified=a.get('sky_verified') is True,
             sun_direction_confidence=a.get('sun_direction_confidence'),
             sun_direction_basis='external_attributes_only',
             sun_direction_xy=list(a['sun_direction_xy']) if a.get('sun_direction_xy') is not None else None,
             approved_controlled_assumption=a.get('approved_controlled_assumption') is True)
    s['sun_blocked'] = s['timeofday'] == 'night' or 'tunnel' in s['scene'] or(s['timeofday']=='unknown'and s['rgb_day_night_hint']=='night')
    return s


def _objects(objects, shape):
    if not isinstance(objects, list):
        raise ValueError('objects must be list; missing annotations use []')
    seen = set()
    h, w = shape[:2]
    for o in objects:
        if not isinstance(o, dict) or not isinstance(o.get('track_id'), str) or not o['track_id'] or not isinstance(o.get('category'), str):
            raise ValueError('invalid object')
        if o['track_id'] in seen:
            raise ValueError('duplicate track_id')
        seen.add(o['track_id'])
        b = o.get('bbox')
        if not isinstance(b, (list, tuple)) or len(b) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in b):
            raise ValueError('invalid bbox')
        if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
            raise ValueError('bbox outside RGB')
        for k in ('occluded', 'truncated'):
            if k in o and type(o[k]) is not bool:
                raise ValueError('visibility flags must be bool')


def _components(mask,area_limit=None):
    n, lab, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for i in range(1,n):
        _,_,cw,ch,area=stats[i]
        if area<3 or(area_limit is not None and area>area_limit)or max(cw,ch)>3*min(cw,ch)or area/(cw*ch)<.4:continue
        yield lab==i,stats[i],centers[i]


def propose_lights(rgb, objects, scene=None):
    _image(rgb)
    _objects(objects, rgb.shape)
    s = scene_evidence(rgb) if scene is None else scene
    if not isinstance(s, dict):
        raise ValueError('scene must be dict')
    f = rgb.astype(np.float32) / 255
    lin = np.where(f <= .04045, f / 12.92, ((f + .055) / 1.055) ** 2.4)
    r, g, b = [lin[:, :, i] for i in range(3)]
    peak = lin.max(axis=2)
    masks = {'red_tail': (r > .65) & (r > 2.5*g) & (r > 2.5*b),
             'white_head': (lin.min(axis=2) > .65) & (lin.min(axis=2) > .72*peak),
             'amber_lamp': (r > .65) & (g > .30) & (g > .38*r) & (g < .9*r) & (b < .35*g)}
    covered = np.zeros(rgb.shape[:2], bool)
    out = [];sky=_sky(rgb)

    def measure(mask, typ, area_limit, region, owner=None):
        for core, st, xy in _components(mask,area_limit):
            x, y, cw, ch, area = map(int, st)
            if area < 3 or area > area_limit or max(cw, ch) > 3*min(cw, ch) or area / (cw*ch) < .4:
                continue
            ring = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~core & region
            if not ring.any():
                continue
            contrast = float(peak[core].mean() - peak[ring].mean())
            if contrast < .20:
                continue
            side = 'point'
            actual_type=typ
            if owner is None and typ=='generic_point'and not s.get('sun_blocked',True)and float(sky[ring].mean())>.5:
                actual_type='visible_sun_candidate'
            features = {'core_area_px': area, 'ring_contrast_linear': contrast,
                        'compact_fill': area/(cw*ch), 'geometry_rule': 'paired_opposite_sides_v1'}
            if owner is not None:
                x0, y0, x1, y1 = owner['bbox']
                nx, ny = (float(xy[0])-x0)/(x1-x0), (float(xy[1])-y0)/(y1-y0)
                if not (.35 <= ny <= .9) or .38 <= nx <= .62:
                    continue
                side = 'left' if nx < .38 else 'right'
                features.update(normalized_xy=[nx, ny], vehicle_area_fraction=area/((x1-x0)*(y1-y0)))
            sid = f"vehicle:{owner['track_id']}:{actual_type}:{side}" if owner else f'unresolved:{actual_type}:{x//16}:{y//16}'
            color=lin[core].mean(axis=0);color/=max(float(color.max()),1e-8)
            out.append({'type': actual_type, 'source_id': sid, 'xy': [float(v) for v in xy],
                        'bbox': [x, y, x+cw, y+ch], 'radius': math.sqrt(area/math.pi),
                        'color': [float(v) for v in color],
                        'radiance': float(peak[core].mean()), 'photometric_pass': True,
                        'geometry_pass': bool(actual_type=='visible_sun_candidate'and s.get('sky_verified')is True), 'temporal_pass': False,
                        'semantic_basis': 'annotated_vehicle_paired_rule_not_calibrated' if owner else('sky_candidate_only'if actual_type=='visible_sun_candidate'else'unresolved_no_object_semantics'),
                        'features': features})
    for o in objects:
        x0, y0, x1, y1 = o['bbox']
        region = np.zeros_like(covered)
        region[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)] = True
        covered |= region
        if o['category'] not in VEHICLES or o.get('occluded') or o.get('truncated'):
            continue
        start = len(out)
        for typ, mask in masks.items():
            measure(mask & region, typ, .015*(x1-x0)*(y1-y0), region, o)
        group = out[start:]
        for a in group:
            for c in group:
                ax, ay = a['features']['normalized_xy']; cx, cy = c['features']['normalized_xy']
                ratio = a['features']['core_area_px']/c['features']['core_area_px']
                if a['type'] == c['type'] and (ax-.5)*(cx-.5) < 0 and abs(ay-cy) <= .12 and .4 <= ratio <= 2.5:
                    a['geometry_pass'] = True
    generic = (lin.mean(axis=2) > .80) & ~covered
    measure(generic,'generic_point',40,~covered)
    # Ambiguous multiple cores for a single side are not paired-lamp evidence.
    counts = {}
    for c in out:
        counts[c['source_id']] = counts.get(c['source_id'], 0) + 1
    out = [c for c in out if counts[c['source_id']] == 1]
    return sorted(out, key=lambda c: (not c['geometry_pass'], c['source_id']))[:16]


def classify_glare_eligibility(scene, tracked_lights):
    qualified = [c for c in tracked_lights if all(c.get(k) is True for k in ('photometric_pass', 'geometry_pass', 'temporal_pass')) and not (scene.get('sun_blocked') and c.get('type') == 'visible_sun_candidate')]
    controlled = (not scene.get('sun_blocked', True) and scene.get('timeofday') in ('daytime', 'day', 'dawn/dusk', 'dawn', 'dusk') and scene.get('weather') in ('clear', 'partly cloudy') and scene.get('sky_verified') is True and (scene.get('sun_direction_confidence') or 0) >= .8 and scene.get('approved_controlled_assumption') is True and scene.get('sun_direction_xy') is not None)
    status = 'VERIFIED' if qualified else 'CONTROLLED' if controlled else 'UNCERTAIN' if tracked_lights or (not scene.get('sun_blocked', True) and scene.get('sky_candidate_fraction', 0) > .01) else 'INELIGIBLE'
    return {'status': status, 'qualified_sources': qualified,
            'source_type': qualified[0]['type'] if qualified else 'off_frame_sun_assumption' if controlled else None,
            'source_id': qualified[0]['source_id'] if qualified else 'external:off_frame_sun' if controlled else None,
            'reason': {'VERIFIED': 'current source passes automatic photometric, paired/sky and temporal rules', 'CONTROLLED': 'explicit approved external off-frame solar assumption; not observed verification', 'UNCERTAIN': 'unresolved source, temporal evidence or daylight sky without supported direction', 'INELIGIBLE': 'unsupported for this generator; not proof of no physical glare'}[status],
            'verification_scope': 'automatic_starting_rules_only', 'manual_audit_required': True,
            'source_confidence': None, 'training_release_ready': False}


class EligibilityTracker:
    def __init__(self):
        self.previous = None
        self.history = {}

    def observe(self, rgb, objects, timestamp_s, frame_index, native_sha256, attributes=None):
        _image(rgb)
        if type(timestamp_s) not in (int, float) or not math.isfinite(timestamp_s) or timestamp_s < 0 or type(frame_index) is not int or frame_index < 0:
            raise ValueError('invalid timestamp/frame')
        if not isinstance(native_sha256, str) or len(native_sha256) != 64 or any(c not in '0123456789abcdefABCDEF' for c in native_sha256):
            raise ValueError('native_sha256 must be hex SHA256')
        scene = scene_evidence(rgb, attributes)
        candidates = propose_lights(rgb, objects, scene)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        prev = self.previous
        if prev and (timestamp_s <= prev[0] or frame_index <= prev[1]):
            raise ValueError('observations must be sorted and unique')
        if prev and gray.shape != prev[2].shape:
            raise ValueError('video dimensions changed')
        if prev and abs((timestamp_s-prev[0])/.2-round((timestamp_s-prev[0])/.2)) > .01:
            raise ValueError('expected sequential 5fps validation cadence or integral gaps')
        if prev and abs((timestamp_s-prev[0])/.2-(frame_index-prev[1]))>.01:
            raise ValueError('timestamp/frame index cadence binding differs')
        reset = prev is None or timestamp_s-prev[0] > .402
        old = {} if reset else prev[3]
        current = {}
        new_history = {}
        for c in candidates:
            sid = c['source_id']; transition = False
            if sid in old and timestamp_s-prev[0] <= .202:
                p = np.array([[old[sid]['xy']]], np.float32)
                q, ok, _ = cv2.calcOpticalFlowPyrLK(prev[2], gray, p, None, winSize=(15, 15), maxLevel=2)
                if q is not None and ok[0, 0]:
                    back, good, _ = cv2.calcOpticalFlowPyrLK(gray, prev[2], q, None, winSize=(15, 15), maxLevel=2)
                    transition = bool(back is not None and good[0, 0] and np.linalg.norm(back-p) <= 1 and np.linalg.norm(q[0, 0]-c['xy']) <= 3)
            hist = deque(self.history.get(sid, []) if sid in old else [], maxlen=5)
            if sid in old:
                for _ in range(max(0,round((timestamp_s-prev[0])/.2)-1)):hist.append((False,False))
            hit = c['photometric_pass'] and c['geometry_pass']
            hist.append((bool(hit), transition))
            c['features']['lk_adjacent_pass'] = transition
            c['features']['window_observations'] = len(hist)
            c['features']['window_hits'] = sum(v[0] for v in hist)
            c['features']['successful_transitions'] = sum(v[1] for v in list(hist)[1:])
            c['temporal_pass'] = bool(hit and len(hist) == 5 and sum(v[0] for v in hist) >= 4 and sum(v[1] for v in list(hist)[1:]) >= 3)
            current[sid] = c; new_history[sid] = hist
        self.history = new_history
        self.previous = (float(timestamp_s), frame_index, gray.copy(), current)
        return dict(frame_index=frame_index, timestamp_s=float(timestamp_s), native_sha256=native_sha256,
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), scene=scene,
                    candidates=candidates, **classify_glare_eligibility(scene, candidates))


def summarize_segments(records):
    segments = []
    for r in records:
        if segments and r['timestamp_s'] <= segments[-1]['end_s']:
            raise ValueError('records must be time sorted')
        same = segments and r['status'] == segments[-1]['status'] and r['source_id'] == segments[-1]['source_id'] and abs(r['timestamp_s']-segments[-1]['end_s']-.2) <= .002 and r['frame_index'] > segments[-1]['frame_end']
        if same:
            segments[-1].update(frame_end=r['frame_index'], end_s=r['timestamp_s'], frames=segments[-1]['frames']+1)
        else:
            segments.append(dict(frame_start=r['frame_index'], frame_end=r['frame_index'], start_s=r['timestamp_s'], end_s=r['timestamp_s'], status=r['status'], source_id=r['source_id'], source_type=r['source_type'], reason=r['reason'], frames=1))
    return [s for s in segments if s['status'] != 'VERIFIED' or s['frames'] >= 2]
