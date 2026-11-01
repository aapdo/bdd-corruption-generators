"""Moving source identity, occlusion, no-source behavior, and core preservation."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from PIL import Image
from recipes.flare_sources import SourceConfig, SourceTracker, detect_sources
from recipes.flare_anchored import AnchoredFlare
from recipes.flare7kstar import compose


def scene(frame, hide=False):
    # Independent moving lamps over a stationary textured background.
    rng=np.random.default_rng(2)
    rgb=rng.integers(10,45,(160,240,3),dtype=np.uint8)
    cv2.circle(rgb,(35+2*frame,60),3,(255,255,255),-1)
    if not hide:cv2.circle(rgb,(190-frame,105),3,(255,0,0),-1)
    return rgb


class SourceTests(unittest.TestCase):
    def test_no_central_fallback_and_large_white_surface(self):
        self.assertEqual(detect_sources(np.zeros((160,240,3),np.uint8)),[])
        rgb=np.zeros((160,240,3),np.uint8);rgb[20:130,30:200]=255
        self.assertEqual(detect_sources(rgb),[])

    def test_two_independent_moving_sources_keep_identity(self):
        tracker=SourceTracker()
        for frame in range(16):tracker.step(scene(frame))
        rows,summary=tracker.finalize()
        self.assertEqual(len(summary['selected_track_ids']),2)
        first=sorted(rows[3]['active_sources'],key=lambda s:s['xy'][0])
        last=sorted(rows[-1]['active_sources'],key=lambda s:s['xy'][0])
        self.assertEqual([s['track_id']for s in first],[s['track_id']for s in last])
        self.assertAlmostEqual(last[0]['xy'][0],65,delta=1)
        self.assertAlmostEqual(last[1]['xy'][0],175,delta=1)
        self.assertFalse(rows[0]['effect_applied'])

    def test_lost_source_is_not_rendered_and_reappearance_reconfirms(self):
        tracker=SourceTracker()
        for frame in range(20):tracker.step(scene(frame,hide=10<=frame<=11))
        rows,summary=tracker.finalize()
        self.assertEqual(len(rows[9]['active_sources']),2)
        self.assertEqual(len(rows[10]['active_sources']),1)
        self.assertEqual(len(rows[12]['active_sources']),1)
        self.assertEqual(len(rows[15]['active_sources']),2)

    def test_current_frame_brightness_is_required(self):
        tracker=SourceTracker()
        for frame in range(16):tracker.step(scene(frame) if frame<12 else np.zeros((160,240,3),np.uint8))
        rows,_=tracker.finalize()
        self.assertTrue(rows[8]['effect_applied'])
        self.assertFalse(any(row['effect_applied']for row in rows[12:]))

    def test_checked_reference_rejects_other_persistent_bright_regions(self):
        tracker=SourceTracker()
        for frame in range(16):tracker.step(scene(frame))
        rows,summary=tracker.finalize([[45.,60.]],5)
        self.assertEqual(len(summary['selected_track_ids']),1)
        self.assertEqual(len(rows[10]['active_sources']),1)
        self.assertLess(rows[10]['active_sources'][0]['xy'][0],100)
        rows,summary=tracker.finalize([],5)
        self.assertEqual(summary['selected_track_ids'],[])
        self.assertFalse(any(r['effect_applied']for r in rows))

    def test_no_negative_layer_aligned_center_and_severity_order(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);yy,xx=np.mgrid[:100,:100];r=np.hypot(xx-50,yy-50)
            compound=np.repeat((np.exp(-r*r/700)*220).astype(np.uint8)[:,:,None],3,axis=2)
            light=np.repeat((np.exp(-r*r/10)*255).astype(np.uint8)[:,:,None],3,axis=2)
            Image.fromarray(compound).save(root/'c.png');Image.fromarray(light).save(root/'l.png')
            bank=dict(patterns=[dict(id='fixture',compound=dict(file='c.png'),light_source=dict(file='l.png'))])
            renderer=AnchoredFlare('fixture',bank,root)
            rgb=scene(0);depth=np.ones(rgb.shape[:2],np.float32)
            source=dict(detect_sources(rgb)[0],track_id=1,fade=1.,forward_backward_error_pixels=0.)
            field,meta=renderer.field(rgb,depth,[source])
            self.assertTrue((field>=0).all());self.assertGreater(float(field.max()),0)
            self.assertLess(meta['sources'][0]['alignment_error_pixels'],.001)
            outputs=[compose(rgb,field,s)for s in [.25,.5,1.]]
            self.assertTrue(np.all(outputs[0]>=rgb))
            self.assertTrue(np.all(outputs[1]>=outputs[0]));self.assertTrue(np.all(outputs[2]>=outputs[1]))
            x,y=np.rint(source['xy']).astype(int)
            self.assertLess(float(field[y,x].max()),1e-7)
            empty,m=renderer.field(rgb,None,[])
            self.assertEqual(float(empty.max()),0);self.assertFalse(m['effect_applied'])
            np.testing.assert_array_equal(compose(rgb,empty,1),rgb)


if __name__=='__main__':unittest.main()

