import unittest
from recipes.inference_batching import ordered_batches

class BatchTests(unittest.TestCase):
    def test_order_tail_and_exact_coverage(self):
        calls=[]
        def predict(images):calls.append(list(images));return [image*2 for image in images]
        result=list(ordered_batches(list(range(19)),lambda item:(str(item),item),predict,8))
        self.assertEqual([len(x)for x in calls],[8,8,3])
        self.assertEqual([r[0]for r in result],list(range(19)))
        self.assertEqual([r[-1]for r in result],[i*2 for i in range(19)])

    def test_failure_is_propagated_without_retry_or_native_fallback(self):
        def predict(images):raise RuntimeError('GPU failure')
        with self.assertRaisesRegex(RuntimeError,'GPU failure'):
            list(ordered_batches([1,2],lambda x:(x,x),predict,4))

    def test_no_cross_video_state_or_empty_work(self):
        calls=[]
        def predict(images):calls.append(len(images));return images
        for items in [[1,2,3],[4,5]]:
            self.assertEqual([r[1]for r in ordered_batches(items,lambda x:(x,x),predict,4)],items)
        self.assertEqual(calls,[3,2])
        self.assertEqual(list(ordered_batches([],lambda x:(x,x),predict,8)),[])

if __name__=='__main__':unittest.main()
