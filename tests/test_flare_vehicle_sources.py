import json
import unittest
import numpy as np
from recipes.flare_vehicle_sources import VehicleTracker,LampTracker,lamp_candidates,select_sources,automatic_vehicle_sources


class VehicleLightTests(unittest.TestCase):
    def test_detects_side_lamps_and_rejects_center_plate(self):
        rgb=np.zeros((120,200,3),np.uint8)
        rgb[42:48,39:45]=255;rgb[42:48,115:121]=255
        rgb[60:70,70:90]=255
        vehicle=dict(box=[20.,20.,140.,100.],vehicle_id=1)
        rows=lamp_candidates(rgb,vehicle)
        json.dumps(rows,allow_nan=False)
        self.assertEqual(len(rows),2);self.assertTrue(all(s['paired']for s in rows))
        self.assertTrue(all(s['xy'][1]<50 for s in rows))

    def test_vehicle_and_lamps_follow_object_motion(self):
        tracker=VehicleTracker();lamps=LampTracker();image=np.zeros((120,240,3),np.uint8)
        ids=[];lights=[]
        for dx in [0,30,60]:
            boxes=tracker.step(image,[dict(box=[20.+dx,20.,120.+dx,90.],score=.9,category=3)])
            ids.append(boxes[0]['vehicle_id'])
            source=dict(vehicle_id=ids[-1],relative_xy=[.15,.45],xy=[35.+dx,51.5],is_red=False)
            lights.append(lamps.step([source])[0])
        self.assertEqual(len(set(ids)),1)
        self.assertEqual(len({s['track_id']for s in lights}),1)
        self.assertTrue(lights[-1]['confirmed'])
        self.assertEqual(lights[-1]['xy'],[95.,51.5])

    def test_lost_lamp_stops_and_reentry_requires_confirmation(self):
        tracker=LampTracker();source=dict(vehicle_id=1,relative_xy=[.2,.5],xy=[20,20],is_red=True)
        tracker.step([source]);self.assertTrue(tracker.step([source])[0]['confirmed'])
        self.assertEqual(tracker.step([]),[])
        self.assertFalse(tracker.step([source])[0]['confirmed'])

    def test_environment_cannot_consume_vehicle_slots(self):
        vehicle=[dict(track_id='v'+str(i),vehicle_id=i//2,score=1.)for i in range(6)]
        env=[dict(track_id='e'+str(i),score=100.)for i in range(8)]
        chosen=select_sources(vehicle,env,set())
        self.assertEqual(len(chosen),6)
        self.assertEqual(sum(s['track_id'].startswith('v')for s in chosen),4)
        self.assertEqual(len({s['vehicle_id']for s in chosen if'vehicle_id'in s}),3)

    def test_complete_automatic_decisions_can_be_saved(self):
        rgb=np.zeros((120,200,3),np.uint8);rgb[42:48,39:45]=255;rgb[42:48,115:121]=255
        class Classifier:
            def score_queries(self,queries):
                return [dict(positive_cosine=.3,margin=.04,minimum_scale_margin=.02,
                             positive_type='headlight',center_delta=.02)for _ in queries]
        detections=[[dict(box=[20.,20.,140.,100.],score=.9,category=3)]for _ in range(4)]
        old=[dict(tracking=dict(sources=[],semantic_candidates=[]))for _ in range(4)]
        rows,summary,decisions=automatic_vehicle_sources([rgb]*4,detections,old,Classifier())
        json.dumps(dict(frames=rows,summary=summary,decisions=decisions),allow_nan=False)
        self.assertEqual(summary['vehicle_light_frames'],3)
        self.assertEqual(len(rows[-1]['active_sources']),2)


if __name__=='__main__':unittest.main()
