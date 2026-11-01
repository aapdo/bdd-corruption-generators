"""Exploratory source-anchored scattering; boxes enter only the audit function."""
import cv2
import numpy as np
from recipes.flare_anchored import AnchoredFlare

LEVELS = dict(weak=.25, middle=.7, strong=1.5)
RECIPE = dict(pattern_spread=2.25, maximum_pattern_scale=1.6,
              halo_linear_peak=.12, halo_radius_per_source_radius=14.,
              halo_radius_bounds_at_720p=[100., 240.], halo_power=1.6,
              levels=LEVELS, source_positions='unchanged_completed_automatic_tracks',
              halo='smooth_source_centered_Moffat_profile; exploratory_BDD_extension',
              brightness='cached_depth_BAM_times_tracked_color_radiance_and_fade',
              label_usage='post_render_audit_only', physical_lens_calibration=False)


class ImpactFlare:
    def __init__(self, video, bank, root, spread=2.25, halo_peak=.12):
        self.template = AnchoredFlare(video, bank, root)
        self.colors = {}
        self.spread, self.halo_peak = spread, halo_peak

    def field(self, rgb, row):
        h, w = rgb.shape[:2]
        sources = row['tracking']['active_sources']
        refs = {r['track_id']: r for r in row['metadata']['sources']}
        if len(refs) != len(sources) or len(refs) > 3:
            raise ValueError('Cached source identity/count mismatch')
        if row['metadata']['pattern_id'] != self.template.pattern_id:
            raise ValueError('Pattern differs from completed pilot')
        field = np.zeros(rgb.shape, np.float32)
        records = []
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        for source in sources:
            key = source['track_id']; ref = refs[key]
            if source['xy'] != ref['xy']:
                raise ValueError('Source location differs from depth/BAM record')
            x, y = source['xy']
            current = np.asarray(source['color'], np.float32)**2.2
            color = current if key not in self.colors else .7*self.colors[key]+.3*current
            self.colors[key] = color
            scale = min(ref['scale']*self.spread, 1.6)
            matrix = cv2.getRotationMatrix2D(self.template.origin, self.template.angle, scale).astype(np.float32)
            matrix[:, 2] += np.asarray([x, y])-self.template.origin
            layer = cv2.warpAffine(self.template.template, matrix, (w, h), flags=cv2.INTER_LINEAR)
            radius = float(np.clip(14*ref['radius'], 100*h/720, 240*h/720))
            distance2 = (xx-x)**2+(yy-y)**2
            halo = self.halo_peak*(1+distance2/radius**2)**-1.6
            layer += halo[:, :, None]
            # Smoothly protect the measured native source core; no negative pixels.
            protect_radius = max(1.5, 1.3*source['radius'])
            border = int(np.ceil(3*protect_radius))
            x0, x1 = max(0, int(x)-border), min(w, int(x)+border+1)
            y0, y1 = max(0, int(y)-border), min(h, int(y)+border+1)
            protect = 1-np.exp(-.5*(distance2[y0:y1, x0:x1]/protect_radius**2)**2)
            layer[y0:y1, x0:x1] *= protect[:, :, None]
            layer *= color[None, None, :]*ref['used_bam']*source['fade']
            field += layer
            error = float(np.linalg.norm(matrix@np.r_[self.template.origin, 1]-[x, y]))
            if error > .01:
                raise ValueError('Pattern source alignment changed')
            records.append(dict(track_id=key, xy=[x, y], pattern_scale=scale,
                                halo_radius_pixels=radius, used_bam=ref['used_bam'],
                                cached_relative_depth=ref['relative_depth'], fade=source['fade'],
                                alignment_error_pixels=error))
        if not np.isfinite(field).all() or (field < 0).any():
            raise ValueError('Invalid nonnegative scattering field')
        return field, dict(pattern_id=self.template.pattern_id, sources=records,
                           effect_applied=bool(sources), source_positions_changed=False)


def audit_boxes(native, rendered, boxes):
    """Descriptive visibility proxies, not a detector AP or annotation-validity test."""
    h, w = native.shape[:2]; rows = []
    for box in boxes:
        x0, y0, x1, y1 = box['xyxy']
        x0, x1 = max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)))
        y0, y1 = max(0, int(np.floor(y0))), min(h, int(np.ceil(y1)))
        if x1-x0 < 4 or y1-y0 < 4:
            continue
        a = native[y0:y1, x0:x1].astype(np.float32)/255
        b = rendered[y0:y1, x0:x1].astype(np.float32)/255
        lum_a = a@np.array([.2126, .7152, .0722], np.float32)
        lum_b = b@np.array([.2126, .7152, .0722], np.float32)
        ca = float(lum_a.std()/(lum_a.mean()+.01))
        cb = float(lum_b.std()/(lum_b.mean()+.01))
        rows.append(dict(box_id=box['id'], category=box['category'], xyxy=box['xyxy'],
                         original_occluded=box['original_occluded'],
                         changed_fraction=float(((b-a).max(axis=2)*255 >= 8).mean()),
                         mean_change_255=float(np.abs(b-a).mean()*255),
                         normalized_contrast_ratio=cb/ca if ca > .01 else None,
                         new_saturation_fraction=float(((b.max(axis=2) >= 250/255)&(a.max(axis=2) < 250/255)).mean())))
    return rows
