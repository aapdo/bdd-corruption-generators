import json,tempfile,unittest
from pathlib import Path
import cv2
import numpy as np
from recipes.flare_wu_day import read_pattern,remove_background,white_balance,blur,matrix_for,compose,track_sun,WuFlare,LEVELS


class WuTests(unittest.TestCase):
    def test_DC_and_white_balance_follow_published_formulas(self):
        image=np.array([[[.1,.2,.3],[.5,.7,.9]],[[.2,.3,.4],[.4,.5,.6]]],np.float32)
        corrected=remove_background(image)
        np.testing.assert_allclose(corrected.min((0,1)),0,atol=1e-7)
        np.testing.assert_allclose(corrected.max((0,1)),image.max((0,1)),atol=3e-7)
        balanced=white_balance(corrected);means=balanced.mean((0,1));self.assertLess(float(means.max()-means.min()),1e-6)
        self.assertFalse(np.isnan(remove_background(np.zeros_like(image))).any())

    def test_blur_uses_zero_padding_and_normalized_21px_kernel(self):
        image=np.zeros((51,51,3),np.float32);image[25,25]=1
        result=blur(image,1.)
        np.testing.assert_allclose(result.sum((0,1)),1,atol=1e-6)
        self.assertEqual(float(result[0,0,0]),0.)
        flat=blur(np.ones((51,51,3),np.float32));self.assertLess(float(flat[0,0,0]),1.);self.assertAlmostEqual(float(flat[25,25,0]),1.,places=6)

    def test_gamma_addition_quantization_and_severity_monotonicity(self):
        rgb=np.full((8,8,3),100,np.uint8);field=np.full(rgb.shape,.04,np.float32)
        expected=int(min(255,255.5*np.sqrt((100/255)**2+.04*.5)))
        self.assertEqual(int(compose(rgb,field,.5)[0,0,0]),expected)
        np.testing.assert_array_equal(compose(rgb,field,0),rgb)
        last=rgb
        for strength in LEVELS.values():
            current=compose(rgb,field,strength);self.assertTrue(np.all(current>=last));last=current

    def test_source_and_camera_center_alignment(self):
        matrix,error=matrix_for([30,20],[99,40],.8,15)
        np.testing.assert_allclose(matrix@np.array([30,20,1]),[99,40],atol=1e-5)
        matrix,error=matrix_for([65,20],[120,25],1,reflection_centers=([40,40],[80,60]))
        np.testing.assert_allclose(matrix@np.array([65,20,1]),[120,25],atol=1e-5)
        np.testing.assert_allclose(matrix@np.array([40,40,1]),[80,60],atol=1e-5)

    def test_current_sun_tracking_and_occlusion_stop(self):
        images=[]
        for x in [65,67,69,71,73]:
            image=np.full((120,200,3),80,np.uint8);cv2.circle(image,(x,30),5,(255,255,255),-1);images.append(image)
        images[4][:]=80
        rows=track_sun(images,2,[69,30])
        for i,x in enumerate([65,67,69,71]):
            self.assertTrue(rows[i]['active']);self.assertLess(np.linalg.norm(np.asarray(rows[i]['xy'])-[x,30]),1.5)
        self.assertFalse(rows[4]['active']);json.dumps(rows,allow_nan=False)

    def test_official_16bit_pattern_keeps_low_bits_and_complete_render(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'pattern.png';image=np.zeros((32,32,3),np.uint16);image[16,16]=[60000,61000,62000];image[15,16]=[1001,1002,1003]
            cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));loaded=read_pattern(path)
            self.assertNotEqual(float(loaded[15,16,0]),float(loaded[15,16,1]))
            flare=WuFlare(path,[16,16]);rgb=np.full((96,160,3),100,np.uint8)
            field,metadata=flare.field(rgb.shape,dict(active=True,xy=[80.,30.]))
            result=compose(rgb,field,1.5);self.assertEqual(result.shape,rgb.shape);json.dumps(metadata,allow_nan=False)
            zero,meta=flare.field(rgb.shape,dict(active=False));np.testing.assert_array_equal(compose(rgb,zero,4.5),rgb)


if __name__=='__main__':unittest.main()
