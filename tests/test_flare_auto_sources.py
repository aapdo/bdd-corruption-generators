"""Decision integration tests; semantic model accuracy needs real-image review."""
import unittest
import cv2
import numpy as np
from recipes.flare_sources import SourceTracker
from recipes.flare_auto_sources import automatic_tracks,decision_from_scores


class MockClassifier:
    def __init__(self,all_rejected=False):self.all_rejected=all_rejected;self.queries=[]
    def score_queries(self,queries):
        result=[]
        for rgb,source in queries:
            self.queries.append(source['xy']);passed=not self.all_rejected and source['xy'][0]<180
            result.append(dict(accepted=passed,positive_type='streetlight',negative_type='road_paint',
                positive_cosine=.35,negative_cosine=.25 if passed else .4,margin=.1 if passed else -.05,
                center_delta=.02,minimum_scale_margin=.05,reasons=[]if passed else['negative_context_competes']))
        return result


def sequence():
    frames=[];tracker=SourceTracker()
    for i in range(18):
        rgb=np.full((160,240,3),20,np.uint8)
        for x,y in [(25,30),(65,80),(105,30),(145,80),(205,30)]:cv2.circle(rgb,(x,y),3,(255,255,255),-1)
        tracker.step(rgb);frames.append(rgb)
    return frames,tracker


class AutoTests(unittest.TestCase):
    def test_stable_bright_negative_is_rejected_and_multiple_lights_are_allowed(self):
        images,tracker=sequence();rows,summary,decisions=automatic_tracks(tracker,images,MockClassifier())
        self.assertFalse(summary['manual_anchors_used'])
        self.assertEqual(summary['maximum_simultaneous_sources'],3)
        self.assertTrue(any(len(r['active_sources'])>1 for r in rows))
        self.assertTrue(all(s['xy'][0]<180 for r in rows for s in r['active_sources']))
        self.assertTrue(any(not d['accepted']for d in decisions))

    def test_no_semantic_acceptance_has_no_central_or_manual_fallback(self):
        images,tracker=sequence();rows,summary,_=automatic_tracks(tracker,images,MockClassifier(True))
        self.assertEqual(summary['active_frames'],0)
        self.assertTrue(all(not r['active_sources']for r in rows))

    def test_center_evidence_rejects_context_only_match(self):
        result=decision_from_scores(np.array([.4,.2]),np.array([.4,.2]),np.array([.4,.2]),['lamp'],['paint'])
        self.assertFalse(result['accepted']);self.assertIn('center_not_responsible_for_source_score',result['reasons'])
        result=decision_from_scores(np.array([.3,.4]),np.array([.3,.4]),np.array([.2,.4]),['lamp'],['paint'])
        self.assertFalse(result['accepted'])


if __name__=='__main__':unittest.main()

