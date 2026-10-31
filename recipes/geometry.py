"""Bounded CPU RGB/depth odometry for R7 candidate rain generation.

Camera coordinates are x right, y down, z forward. T_prev_to_now maps points
from the previous camera into the current camera. Depth is interpreted as
estimated camera-z metres. K and scale are assumptions, not BDD calibration.
Fit diagnostics are conditional numerical checks, not calibrated pose CIs.
"""
import math
import threading

import numpy as np


class GeometryError(ValueError):
    """Malformed geometry input: caller must hold the clip."""


DEFAULT_GEOMETRY_CONFIG = {
    'grid_stride': 10, 'max_correspondences': 1800,
    'min_correspondences': 100, 'min_inliers': 80,
    'min_inlier_fraction': 0.25, 'min_flow_valid_fraction': 0.20,
    'min_grid_coverage': 0.50, 'min_axis_span': 0.45,
    'depth_min': 0.5, 'depth_max': 120.0,
    'depth_edge_absolute': 0.30, 'depth_edge_relative': 0.05,
    'depth_edge_dilation': 2, 'texture_gradient_min': 2.0,
    'photometric_error_max': 0.20,
    'fb_absolute_px': 1.0, 'fb_relative': 0.05,
    'ransac_reprojection_px': 2.5, 'ransac_iterations': 200,
    'ransac_confidence': 0.999, 'ransac_seed': 7301,
    'median_reprojection_max_px': 1.25, 'p90_reprojection_max_px': 2.5,
    'depth_relative_median_max': 0.12, 'depth_relative_p90_max': 0.30,
    'depth_consistent_fraction_min': 0.65,
    'depth_consistency_absolute': 0.50, 'depth_consistency_relative': 0.15,
    'rotation_max_deg': 12.0, 'translation_max_estimated_m': 8.0,
    'translation_over_depth_max': 0.40,
    'jacobian_condition_max': 10000.0,
    'rotation_fit_std_max_deg': 2.0,
    'translation_fit_std_over_depth_max': 0.03,
    'background_depth_quantile': 0.60, 'min_split_fit_inliers': 20,
    'holdout_far_inlier_fraction_min': 0.70,
    'holdout_far_median_reprojection_max_px': 1.5,
    'holdout_grid_coverage_min': 0.375,
    'split_rotation_disagreement_max_deg': 0.5,
    'split_translation_disagreement_over_depth_max': 0.03,
    'split_translation_disagreement_max_estimated_m': 1.5,
    'minority_consensus_threshold': 0.60,
    'minority_background_depth_ratio_min': 1.5,
    'minority_background_depth_min_estimated_m': 12.0,
}
_RANSAC_LOCK = threading.Lock()


def camera_intrinsics(image_shape=(640, 640)):
    """Assumed native f=900px at1280x720, anisotropically resized to HxW."""
    h, w = map(int, image_shape)
    if min(h, w) < 16:
        raise GeometryError('Image dimensions must be >=16')
    return np.array([[450.0*w/640, 0, 320.0*w/640],
                     [0, 800.0*h/640, 320.0*h/640], [0, 0, 1]], np.float64)


def validate_intrinsics(intrinsics, shape):
    k = camera_intrinsics(shape) if intrinsics is None else np.asarray(intrinsics, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-12):
        raise GeometryError('Finite, zero-skew pinhole intrinsics required')
    return k.copy()


def validate_pose(pose):
    t = np.asarray(pose, np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(t[:3, :3])-1) > 1e-5):
        raise GeometryError('Pose must be a finite proper SE3 world-to-camera transform')
    return t.copy()


def validate_rgb_depth(rgb, depth):
    image, d = np.asarray(rgb), np.asarray(depth, np.float32)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
        raise GeometryError('RGB uint8 HxWx3 required')
    if d.shape != image.shape[:2] or not np.isfinite(d).all() or np.any(d < 0):
        raise GeometryError('Aligned finite nonnegative estimated camera-z depth required')
    return image, d


def _config(values):
    result = dict(DEFAULT_GEOMETRY_CONFIG)
    if values:
        if set(values)-set(result):
            raise GeometryError('Unknown geometry config keys')
        result.update(values)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in result.values()):
        raise GeometryError('Geometry configuration must be finite and nonnegative')
    for key in ('grid_stride', 'max_correspondences', 'min_correspondences', 'min_inliers',
                'depth_edge_dilation', 'ransac_iterations', 'ransac_seed', 'min_split_fit_inliers'):
        if type(result[key]) is not int:
            raise GeometryError('Integer geometry configuration required: '+key)
    if (result['grid_stride'] < 2 or not 6 <= result['min_inliers'] <= result['min_correspondences']
            or result['min_correspondences'] > result['max_correspondences'] or result['max_correspondences'] > 5000
            or not 0 < result['depth_min'] < result['depth_max'] or result['depth_edge_dilation'] > 5
            or not 1 <= result['ransac_iterations'] <= 1000 or not 0 < result['ransac_confidence'] < 1):
        raise GeometryError('Geometry bounds are invalid')
    for key in ('min_inlier_fraction', 'min_flow_valid_fraction', 'min_grid_coverage', 'min_axis_span',
                'depth_consistent_fraction_min', 'background_depth_quantile',
                'holdout_far_inlier_fraction_min', 'holdout_grid_coverage_min', 'minority_consensus_threshold'):
        if not 0 <= result[key] <= 1:
            raise GeometryError('Fraction outside [0,1]')
    return result


def depth_boundary(depth, absolute=0.30, relative=0.05, dilation=2):
    import cv2
    d = np.asarray(depth, np.float32)
    lo = cv2.erode(d, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    hi = cv2.dilate(d, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    edge = ((hi-lo) > np.maximum(absolute, relative*np.maximum(d, 1e-6))) | (d <= 0)
    if dilation:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2*dilation+1, 2*dilation+1), np.uint8)) > 0
    return edge


def _dis_pair(first, second):
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    gray0, gray1 = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (first, second)]
    forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray0, gray1, None)
    backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray1, gray0, None)
    return forward, backward


def _sample(array, xy, nearest=False):
    import cv2
    return cv2.remap(np.asarray(array, np.float32), xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def se3_exp(twist):
    """Exponential map; twist is [rotation radians, translation tangent]."""
    import cv2
    xi = np.asarray(twist, np.float64)
    if xi.shape != (6,) or not np.isfinite(xi).all():
        raise GeometryError('Finite 6-vector twist required')
    omega = xi[:3];theta = float(np.linalg.norm(omega))
    w = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    if theta < 1e-7:
        v = np.eye(3)+0.5*w+(w@w)/6
    else:
        v = np.eye(3)+(1-math.cos(theta))/theta**2*w+(theta-math.sin(theta))/theta**3*(w@w)
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(omega)[0]
    result[:3, 3] = v@xi[3:]
    return result


def se3_log(transform):
    import cv2
    t = validate_pose(transform);omega = cv2.Rodrigues(t[:3, :3])[0].reshape(3)
    theta = float(np.linalg.norm(omega))
    w = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    if theta < 1e-7:
        inverse_v = np.eye(3)-0.5*w+(w@w)/12
    else:
        coefficient = (1-0.5*theta/math.tan(theta/2))/theta**2
        inverse_v = np.eye(3)-0.5*w+coefficient*(w@w)
    return np.r_[omega, inverse_v@t[:3, 3]]


def estimate_pair(rgb_prev, rgb_now, depth_prev, depth_now, *, intrinsics=None, config=None):
    """Return accepted, T_prev_to_now, JSON-safe QA. Rejection never bridges motion.

    Caller must hold the whole clip on any rejected pair. The identity matrix
    in a rejection is a serialization placeholder, not an estimated static pose.
    """
    import cv2
    image0, d0 = validate_rgb_depth(rgb_prev, depth_prev)
    image1, d1 = validate_rgb_depth(rgb_now, depth_now)
    if image0.shape != image1.shape:
        raise GeometryError('Adjacent shapes differ')
    cfg = _config(config);h, w = d0.shape;k = validate_intrinsics(intrinsics, (h, w))
    qa = {'method': 'DIS_bidirectional_depth_PnPRansac_LM', 'K': k.tolist(),
          'intrinsics_source': 'assumed_native_f900_at1280x720_anisotropic_resize',
          'depth_source': 'estimated_camera_z_metres', 'calibrated_camera': False,
          'uncertainty_scope': 'conditional_linearized_fit_diagnostic_not_calibrated_pose_CI',
          'hold_reasons': [], 'input_shape': [h, w]}

    def reject(reason):
        qa['hold_reasons'].append(reason);qa['accepted'] = False
        return {'accepted': False, 'T_prev_to_now': np.eye(4), 'qa': qa}

    forward, backward = _dis_pair(image0, image1)
    if forward.shape != (h, w, 2) or backward.shape != forward.shape or not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise GeometryError('DIS returned invalid flow')
    yy, xx = np.mgrid[:h, :w]
    grid = np.stack([xx, yy], -1).astype(np.float32);target = grid+forward
    within = (target[..., 0] >= 1)&(target[..., 0] < w-2)&(target[..., 1] >= 1)&(target[..., 1] < h-2)
    reverse_at_target = _sample(backward, target)
    fb = np.linalg.norm(forward+reverse_at_target, axis=-1)
    tolerance = cfg['fb_absolute_px']+cfg['fb_relative']*(np.linalg.norm(forward, axis=-1)+np.linalg.norm(reverse_at_target, axis=-1))
    flow_valid = within&(fb <= tolerance)
    qa['flow_valid_fraction'] = float(flow_valid.mean())
    if qa['flow_valid_fraction'] < cfg['min_flow_valid_fraction']:
        return reject('insufficient_bidirectional_flow_support')
    edge0 = depth_boundary(d0, cfg['depth_edge_absolute'], cfg['depth_edge_relative'], cfg['depth_edge_dilation'])
    edge1 = depth_boundary(d1, cfg['depth_edge_absolute'], cfg['depth_edge_relative'], cfg['depth_edge_dilation'])
    gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = np.hypot(cv2.Sobel(gray0, cv2.CV_32F, 1, 0)/8, cv2.Sobel(gray0, cv2.CV_32F, 0, 1)/8)
    next_depth = _sample(d1, target)
    photometric = np.abs(gray0-_sample(gray1, target))/255
    valid = (flow_valid & ~edge0 & (_sample(edge1, target, True) < 0.5)
             & (d0 >= cfg['depth_min']) & (d0 <= cfg['depth_max'])
             & (next_depth >= cfg['depth_min']) & (next_depth <= cfg['depth_max'])
             & (gradient >= cfg['texture_gradient_min']) & (photometric <= cfg['photometric_error_max']))
    stride = cfg['grid_stride'];sample = np.zeros((h, w), bool)
    sample[stride//2::stride, stride//2::stride] = True
    y, x = np.where(valid & sample)
    if len(x) > cfg['max_correspondences']:
        indices = np.linspace(0, len(x)-1, cfg['max_correspondences'], dtype=np.int64);x, y = x[indices], y[indices]
    qa.update(correspondences=int(len(x)), depth_boundary_fraction=float((edge0|edge1).mean()),
              geometric_candidate_fraction=float(valid.mean()))
    if len(x) < cfg['min_correspondences']:
        return reject('insufficient_textured_nonboundary_depth_correspondences')
    z = d0[y, x].astype(np.float64)
    points = np.stack([(x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z], axis=1)
    pixels = target[y, x].astype(np.float64)
    return _solve_correspondences(points, pixels, next_depth[y, x], np.stack([x, y], 1),
                                  k, (h, w), cfg, qa)


def _fit_pnp(points, pixels, k, cfg, minimum):
    import cv2
    if len(points) < minimum:
        return None
    try:
        with _RANSAC_LOCK:
            cv2.setRNGSeed(cfg['ransac_seed'])
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(points, pixels, k, None,
                iterationsCount=cfg['ransac_iterations'], reprojectionError=cfg['ransac_reprojection_px'],
                confidence=cfg['ransac_confidence'], flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < minimum:
            return None
        ids = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(points[ids], pixels[ids], k, None, rvec, tvec)
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
        return rvec, tvec
    except cv2.error:
        return None


def _pose_disagreement(a, b, median_depth):
    import cv2
    ra, rb = cv2.Rodrigues(a[0])[0], cv2.Rodrigues(b[0])[0]
    rotation = float(np.linalg.norm(cv2.Rodrigues(ra@rb.T)[0])*180/np.pi)
    translation = float(np.linalg.norm(a[1]-b[1]))
    return {'rotation_deg': rotation, 'translation_estimated_m': translation,
            'translation_over_depth': translation/median_depth}


def _solve_correspondences(points, pixels, observed_depth, source_xy, k, shape, cfg, qa=None):
    """Robust static-background hypothesis with disjoint spatial holdouts.

    The far-depth prior is explicit and can fail (e.g. moving distant objects).
    Disjoint correspondence fits are a stability test; their errors remain
    correlated through DIS/depth and are not independent calibrated sensors.
    """
    import cv2
    points = np.asarray(points, np.float64);pixels = np.asarray(pixels, np.float64)
    observed_depth = np.asarray(observed_depth, np.float64);source_xy = np.asarray(source_xy, np.float64)
    h, w = shape;x, y = source_xy.T;z = points[:, 2]
    qa = {} if qa is None else qa
    qa.setdefault('hold_reasons', [])

    def reject(reason):
        qa['hold_reasons'].append(reason);qa['accepted'] = False
        return {'accepted': False, 'T_prev_to_now': np.eye(4), 'qa': qa}

    def coverage(indices):
        return len(set(zip(np.clip((x[indices]*4//w).astype(int), 0, 3),
                           np.clip((y[indices]*4//h).astype(int), 0, 3))))/16

    cut = float(np.quantile(z, cfg['background_depth_quantile']))
    # Quantile ties at a large foreground plane must not pull that entire
    # plane into the far support. Uniform/max-depth ties retain their support.
    far = z > cut
    if not far.any():
        far = z >= cut
    order = np.lexsort((x, y));half = np.zeros(len(points), bool);half[order[1::2]] = True
    groups = [np.flatnonzero(far & ~half), np.flatnonzero(far & half)]
    qa.update(background_depth_cut_estimated_m=cut, background_candidate_fraction=float(far.mean()),
              background_depth_comparator='strict_above_quantile_or_tied_maximum',
              background_prior='spatially_distributed_far_depth_static_scene_hypothesis',
              holdout_protocol='disjoint_raster_interleaved_correspondences; correlated_flow/depth_noise_remains',
              split_training_counts=[len(a) for a in groups], correspondences=len(points))
    fits = [_fit_pnp(points[a], pixels[a], k, cfg, cfg['min_split_fit_inliers']) for a in groups]
    if any(fit is None for fit in fits):
        return reject('insufficient_disjoint_background_fit_support')
    qa['holdouts'] = []
    for fit, holdout in zip(fits, groups[::-1]):
        prediction = cv2.projectPoints(points[holdout], fit[0], fit[1], k, None)[0].reshape(-1, 2)
        errors = np.linalg.norm(prediction-pixels[holdout], axis=1)
        good = errors <= cfg['ransac_reprojection_px']
        support = holdout[good]
        record = {'count': len(holdout), 'inliers': int(good.sum()),
                  'inlier_fraction': float(good.mean()), 'median_reprojection_px': float(np.median(errors)),
                  'grid_coverage': coverage(support)}
        record['passed'] = (record['inliers'] >= cfg['min_split_fit_inliers']
                            and record['inlier_fraction'] >= cfg['holdout_far_inlier_fraction_min']
                            and record['median_reprojection_px'] <= cfg['holdout_far_median_reprojection_max_px']
                            and record['grid_coverage'] >= cfg['holdout_grid_coverage_min'])
        qa['holdouts'].append(record)
    median_background_depth = float(np.median(z[far]))
    agreement = _pose_disagreement(fits[0], fits[1], median_background_depth)
    qa['split_pose_disagreement'] = agreement

    def agrees(value):
        return (value['rotation_deg'] <= cfg['split_rotation_disagreement_max_deg'] and
                value['translation_estimated_m'] <= cfg['split_translation_disagreement_max_estimated_m'] and
                value['translation_over_depth'] <= cfg['split_translation_disagreement_over_depth_max'])

    if not all(x['passed'] for x in qa['holdouts']) or not agrees(agreement):
        return reject('independent_background_cross_consistency')
    final = _fit_pnp(points[far], pixels[far], k, cfg, cfg['min_inliers']//2)
    if final is None:
        return reject('background_PnP_fit_failed')
    predicted = cv2.projectPoints(points, final[0], final[1], k, None)[0].reshape(-1, 2)
    ids = np.flatnonzero(np.linalg.norm(predicted-pixels, axis=1) <= cfg['ransac_reprojection_px'])
    if len(ids) < cfg['min_inliers']:
        return reject('PnP_inlier_count_failed')
    rvec, tvec = cv2.solvePnPRefineLM(points[ids], pixels[ids], k, None, final[0], final[1])
    qa['final_vs_split_disagreement'] = [_pose_disagreement((rvec, tvec), fit, median_background_depth) for fit in fits]
    projection, jacobian = cv2.projectPoints(points[ids], rvec, tvec, k, None)
    residual = np.linalg.norm(projection.reshape(-1, 2)-pixels[ids], axis=1)
    rotation = cv2.Rodrigues(rvec)[0];translation = tvec.reshape(3)
    transformed = points[ids]@rotation.T+translation
    measured = observed_depth[ids]
    depth_error = np.abs(transformed[:, 2]-measured)
    relative_depth_error = depth_error/np.maximum(measured, cfg['depth_min'])
    consistent = depth_error <= np.maximum(cfg['depth_consistency_absolute'], cfg['depth_consistency_relative']*measured)
    spatial_coverage = coverage(ids)
    span = [float(np.ptp(x[ids])/w), float(np.ptp(y[ids])/h)]
    median_depth = float(np.median(z[ids]))
    j = jacobian[:, :6].copy();j[:, 3:6] *= median_depth
    singular = np.linalg.svd(j, compute_uv=False)
    condition = float(singular[0]/max(singular[-1], 1e-12))
    noise = max(0.25, float(np.sqrt(np.mean(residual**2))))
    covariance = noise**2*np.linalg.pinv(j.T@j, rcond=1e-12)
    deviations = np.sqrt(np.maximum(0, np.diag(covariance)))
    qa.update(inliers=int(len(ids)), inlier_fraction=float(len(ids)/len(points)),
              median_reprojection_px=float(np.median(residual)), p90_reprojection_px=float(np.percentile(residual, 90)),
              grid_coverage=float(spatial_coverage), axis_span=span,
              depth_relative_median=float(np.median(relative_depth_error)),
              depth_relative_p90=float(np.percentile(relative_depth_error, 90)),
              depth_consistent_fraction=float(consistent.mean()),
              positive_depth_fraction=float((transformed[:, 2] > cfg['depth_min']).mean()),
              rotation_deg=float(np.linalg.norm(rvec)*180/np.pi),
              translation_estimated_m=float(np.linalg.norm(translation)), median_depth_estimated_m=median_depth,
              translation_over_depth=float(np.linalg.norm(translation)/median_depth),
              jacobian_condition=condition, rotation_fit_std_deg=float(max(deviations[:3])*180/np.pi),
              translation_fit_std_over_depth=float(max(deviations[3:])),
              fit_covariance_normalized_parameters=covariance.tolist())
    outside = np.ones(len(points), bool);outside[ids] = False
    outlier_median = float(np.median(z[outside])) if outside.any() else None
    minority = qa['inlier_fraction'] < cfg['minority_consensus_threshold']
    qa.update(minority_consensus=minority, inlier_depth_median_estimated_m=median_depth,
              outlier_depth_median_estimated_m=outlier_median,
              inlier_far_background_fraction=float(far[ids].mean()))
    gates = {
        'inlier_fraction': qa['inlier_fraction'] >= cfg['min_inlier_fraction'],
        'spatial_support': spatial_coverage >= cfg['min_grid_coverage'] and min(span) >= cfg['min_axis_span'],
        'independent_holdouts': all(record['passed'] for record in qa['holdouts']) and agrees(agreement),
        'final_split_agreement': all(agrees(value) for value in qa['final_vs_split_disagreement']),
        'minority_background_evidence': not minority or (
            median_depth >= cfg['minority_background_depth_min_estimated_m'] and outlier_median is not None
            and median_depth >= cfg['minority_background_depth_ratio_min']*outlier_median
            and qa['inlier_far_background_fraction'] >= 0.5),
        'reprojection': qa['median_reprojection_px'] <= cfg['median_reprojection_max_px'] and qa['p90_reprojection_px'] <= cfg['p90_reprojection_max_px'],
        'depth_consistency': qa['depth_relative_median'] <= cfg['depth_relative_median_max'] and qa['depth_relative_p90'] <= cfg['depth_relative_p90_max'] and qa['depth_consistent_fraction'] >= cfg['depth_consistent_fraction_min'],
        'positive_depth': qa['positive_depth_fraction'] == 1,
        'motion_bound': qa['rotation_deg'] <= cfg['rotation_max_deg'] and qa['translation_estimated_m'] <= cfg['translation_max_estimated_m'] and qa['translation_over_depth'] <= cfg['translation_over_depth_max'],
        'conditional_uncertainty': condition <= cfg['jacobian_condition_max'] and qa['rotation_fit_std_deg'] <= cfg['rotation_fit_std_max_deg'] and qa['translation_fit_std_over_depth'] <= cfg['translation_fit_std_over_depth_max'],
    }
    qa['gates'] = gates
    qa['hold_reasons'] = [name for name, passed in gates.items() if not passed]
    accepted = not qa['hold_reasons'];qa['accepted'] = accepted
    transform = np.eye(4)
    if accepted:
        transform[:3, :3] = rotation;transform[:3, 3] = translation
        validate_pose(transform)
    return {'accepted': accepted, 'T_prev_to_now': transform, 'qa': qa}
