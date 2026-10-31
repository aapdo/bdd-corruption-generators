import hashlib
import json
import cv2
import numpy as np
import pytest
from glare_eligibility import (POLICY, scene_evidence, propose_lights,
    classify_glare_eligibility, EligibilityTracker, summarize_segments)

SHA = 'a' * 64
OBJ = [{'track_id': 'v1', 'bbox': [20, 20, 172, 116], 'category': 'car'}]


def image(color=(255, 255, 255), paired=True):
    a = np.zeros((128, 192, 3), np.uint8)
    # Low-intensity asymmetric texture gives LK an unambiguous neighborhood.
    for x in ([48, 144] if paired else [48]):
        cv2.rectangle(a, (x-8, 69), (x+6, 82), (25, 35, 20), -1)
        cv2.line(a, (x-9, 68), (x-4, 84), (55, 45, 40), 1)
        cv2.circle(a, (x, 78), 3, color, -1)
    return a


def observe(tracker, a, i, objects=None, **kw):
    return tracker.observe(a, OBJ if objects is None else objects,
                           kw.pop('time', i*.2), i, SHA, **kw)


@pytest.mark.parametrize('color', [(255, 255, 255), (255, 0, 0)])
def test_paired_verified_after_five(color):
    a = image(color)
    t = EligibilityTracker()
    records = [observe(t, a, i) for i in range(5)]
    assert all(r['status'] == 'UNCERTAIN' for r in records[:4])
    r = records[-1]
    assert r['status'] == 'VERIFIED'
    assert len(r['qualified_sources']) == 2
    assert all(c['features']['successful_transitions'] == 4 for c in r['qualified_sources'])
    assert r['rgb_sha256'] == hashlib.sha256(a.tobytes()).hexdigest()
    assert r['source_confidence'] is None and not r['training_release_ready']
    json.dumps(r, allow_nan=False)


def test_unpaired_red_never_verified():
    t = EligibilityTracker()
    for i in range(7):
        r = observe(t, image((255, 0, 0), False), i)
        assert r['status'] == 'UNCERTAIN'
        assert not any(c['geometry_pass'] for c in r['candidates'])


def test_broad_paint_and_center_roof_rejected():
    a = np.zeros((128, 192, 3), np.uint8)
    a[60:90, 30:70] = (255, 0, 0)
    assert propose_lights(a, OBJ) == []
    a[:] = 0
    cv2.circle(a, (96, 78), 3, (255, 255, 255), -1)
    cv2.circle(a, (48, 28), 3, (255, 255, 255), -1)
    assert propose_lights(a, OBJ) == []


def test_yellow_supported():
    c = propose_lights(image((255, 210, 0)), OBJ)
    assert len(c) == 2
    assert all(x['type'] == 'amber_lamp' and x['geometry_pass'] for x in c)


def test_night_and_sky_controlled():
    dark = np.zeros((128, 192, 3), np.uint8)
    s = scene_evidence(dark, {'timeofday': 'night'})
    assert classify_glare_eligibility(s, [])['status'] == 'INELIGIBLE'
    sky = np.full_like(dark, (90, 160, 240))
    attrs = {'timeofday': 'daytime', 'weather': 'clear'}
    s = scene_evidence(sky, attrs)
    assert s['sky_candidate_fraction'] > .1
    assert s['sun_direction_confidence'] is None
    assert classify_glare_eligibility(s, [])['status'] == 'UNCERTAIN'
    attrs.update(sky_verified=True, sun_direction_confidence=.9,
                 sun_direction_xy=[1.2, -.1], approved_controlled_assumption=True)
    assert classify_glare_eligibility(scene_evidence(sky, attrs), [])['status'] == 'CONTROLLED'
    for block in ({'timeofday': 'night'}, {'scene': 'tunnel'}):
        assert classify_glare_eligibility(scene_evidence(sky, dict(attrs, **block)), [])['status'] != 'CONTROLLED'


def test_missing_annotations_occlusion_source_change_and_gap():
    a = image()
    unresolved = propose_lights(a, [])
    assert unresolved and all(not c['geometry_pass'] for c in unresolved)
    t = EligibilityTracker()
    for i in range(5):
        r = observe(t, a, i)
    assert r['status'] == 'VERIFIED'
    assert observe(t, a, 5, objects=[])['status'] != 'VERIFIED'
    assert observe(t, a, 6)['status'] != 'VERIFIED'
    changed = [dict(OBJ[0], track_id='v2')]
    assert observe(t, a, 7, objects=changed)['status'] != 'VERIFIED'
    assert observe(t, a, 12, objects=changed)['status'] != 'VERIFIED'
    hidden = [dict(changed[0], occluded=True)]
    assert observe(t, a, 13, objects=hidden)['candidates'] == []


def test_segments_no_gap_bridging_or_single_verified():
    def rec(i, time, sid='v'):
        return dict(frame_index=i, timestamp_s=time, status='VERIFIED',
                    source_id=sid, source_type='white_head', reason='rules')
    rows = [rec(0, 0), rec(1, .2), rec(6, 1.2), rec(7, 1.4), rec(8, 1.6, 'other')]
    s = summarize_segments(rows)
    assert len(s) == 2
    assert [(v['start_s'], v['end_s'], v['frames']) for v in s] == [(0, .2, 2), (1.2, 1.4, 2)]


def test_validation_and_policy():
    assert POLICY['min_hits'] == 4
    with pytest.raises(ValueError):
        scene_evidence(np.zeros((10, 10, 3), float))
    with pytest.raises(ValueError):
        propose_lights(image(), [dict(OBJ[0], bbox=[0, 0, 999, 100])])
    with pytest.raises(ValueError):
        propose_lights(image(), None)
    with pytest.raises(ValueError):
        scene_evidence(image(), {'sun_direction_confidence': float('nan')})
    t = EligibilityTracker()
    observe(t, image(), 0)
    with pytest.raises(ValueError):
        observe(t, image(), 0)
    with pytest.raises(ValueError):
        t.observe(image(), OBJ, .2, 1, 'invalid')


def test_renderer_requires_bound_eligibility_and_keeps_render_quality_separate():
    from glare import render_from_eligibility
    a=image();tracker=EligibilityTracker()
    for i in range(5):r=observe(tracker,a,i)
    out,q=render_from_eligibility(a,r,.5)
    assert np.all(out>=a)and q['source_eligibility']=='VERIFIED'and q['render_quality']=='UNASSESSED'and not q['training_release_ready']
    with pytest.raises(ValueError,match='different RGB'):render_from_eligibility(np.zeros_like(a),r,.5)
    r['status']='UNCERTAIN'
    with pytest.raises(ValueError,match='eligibility'):render_from_eligibility(a,r,.5)
