"""Persistent state sharing for a severity sweep."""
import copy
import math
def advance_snow(renderer, pose, timestamp, twist=None):
    """Identical particle-only prefix of SnowRenderer.render, with no image work."""
    timestamp, elapsed = renderer.clock.check(timestamp)
    motion = 0 if twist is None else math.hypot(*twist[:3])*renderer.config['exposure_s']/2
    guard = renderer.guard+motion
    if not math.isfinite(guard) or guard >= renderer.config['box_half_extent']:
        raise ValueError('Exposure guard exceeded')
    if renderer._base is None:
        renderer._initialize(pose, timestamp)
    renderer._advance(-pose[:3, :3].T@pose[:3, 3], elapsed, guard)
    renderer._time = timestamp-renderer._origin_time
    renderer.clock.last = timestamp

def snow_levels(renderer, image, depth, pose, timestamp, levels, twist=None):
    results = []
    for level in levels:
        trial = copy.deepcopy(renderer)
        value = trial.render(image, depth, pose, level, timestamp, camera_twist=twist)
        results.append((value, trial.state_fingerprint()))
    if len({h for _, h in results}) != 1:
        raise ValueError('Severity changed particle population or trajectory')
    advance_snow(renderer, pose, timestamp, twist)
    if renderer.state_fingerprint() != results[0][1]:
        raise ValueError('Particle-only advance differs from rendering')
    return [dict(value=v, shared_state_sha256=h) for v, h in results]
