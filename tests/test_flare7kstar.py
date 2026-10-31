import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from recipes.flare7kstar import bam, compose, relative_depth, triage, seed, VideoFlare

class RendererTests(unittest.TestCase):
    def test_inverse_depth_is_inverted_before_bam(self):
        d = relative_depth(np.array([[1., 2., 4.]], np.float32))
        self.assertGreater(float(d[0, 1]), float(d[0, 2]))
        factors = bam(d[0, 1:], [[50, 50], [50, 50]], 100, 100)
        self.assertAlmostEqual(float(factors[1] / factors[0]), 4.)

    def test_off_axis_cosine_and_single_source(self):
        factors = bam([1, 1], [[50, 50], [100, 50]], 100, 100)
        self.assertAlmostEqual(float(factors[0]), 1.)
        self.assertAlmostEqual(float(factors[1]), np.cos(np.pi / 6), places=6)

    def test_zero_identity_and_strength_monotonicity(self):
        rgb = np.full((10, 20, 3), 40, np.uint8); field = np.full(rgb.shape, .2, np.float32)
        np.testing.assert_array_equal(compose(rgb, field, 0), rgb)
        levels = [compose(rgb, field, s) for s in [.25, .5, 1.]]
        self.assertTrue(np.all(levels[0] < levels[1])); self.assertTrue(np.all(levels[1] < levels[2]))

    def test_no_visual_approval_or_failure_native_substitution(self):
        result = triage([dict(changed_fraction=0, new_saturation_fraction=0, native_mean_luminance=.5)], 0)
        self.assertFalse(result['visual_quality_approved']); self.assertIn('effect_too_small', result['reasons'])

    def test_bad_depth_rejected(self):
        for d in [np.zeros((2, 2)), np.full((2, 2), np.nan)]:
            with self.assertRaises(ValueError): relative_depth(d)
        self.assertEqual(seed('same'), seed('same')); self.assertNotEqual(seed('same'), seed('other'))

    def test_video_replay_and_native_coordinate_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pattern=np.zeros((32,32,3),np.uint8)
            cv2.circle(pattern,(16,16),8,(100,100,100),-1)
            light=np.zeros_like(pattern);cv2.circle(light,(16,16),2,(255,255,255),-1)
            Image.fromarray(pattern).save(root/'p.png');Image.fromarray(light).save(root/'l.png')
            bank=dict(patterns=[dict(id='test',compound=dict(file='p.png'),light_source=dict(file='l.png'))])
            first=VideoFlare('replay',bank,root);second=VideoFlare('replay',bank,root)
            rgb=np.full((720,1280,3),60,np.uint8);depth=np.ones((720,1280),np.float32)
            a,ma=first.field(rgb,depth);b,mb=second.field(rgb,depth)
            np.testing.assert_array_equal(a,b);self.assertEqual(ma,mb)
            self.assertTrue(a.max()>0);self.assertTrue(all(ma['visible_source_masks']))
            # Known motion in flow's 640x360 coordinates must scale to native pixels.
            motion=VideoFlare('motion',bank,root)
            gray=np.random.default_rng(101).integers(0,256,(360,640),dtype=np.uint8)
            moved=cv2.warpAffine(gray,np.float32([[1,0,4],[0,1,2]]),(640,360))
            motion._motion(gray);state=motion._motion(moved)
            self.assertEqual(state,'background_partial_affine')
            self.assertAlmostEqual(float(motion.transport[0,2]),8,delta=.5)
            self.assertAlmostEqual(float(motion.transport[1,2]),4,delta=.5)

if __name__ == '__main__': unittest.main()

