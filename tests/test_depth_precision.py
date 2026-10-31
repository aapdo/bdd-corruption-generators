import unittest
import numpy as np
from recipes.depth_precision import reconstruct_depths

class PrecisionTests(unittest.TestCase):
    def test_only_nonpositive_frame_is_recomputed_in_order(self):
        calls=[]
        def single(image):calls.append(image);return np.full((2,2),5,np.float32)
        outputs,count=reconstruct_depths(['a','b'],[np.full((2,2),4,np.float32),np.array([[-.01,2],[3,4]],np.float32)],single)
        self.assertEqual(calls,['b']);self.assertEqual(count,1)
        np.testing.assert_allclose(outputs[0],.25);np.testing.assert_allclose(outputs[1],.2)

    def test_scalar_path_and_nonfinite_failure_are_preserved(self):
        def forbidden(image):raise AssertionError('Scalar path should not be repeated')
        result,count=reconstruct_depths(['a'],[np.array([[-.01,2],[3,4]],np.float32)],forbidden)
        self.assertEqual(count,0);self.assertEqual(len(result),1)
        with self.assertRaises(ValueError):reconstruct_depths(['a'],[np.full((2,2),np.nan)],forbidden)

if __name__=='__main__':unittest.main()
