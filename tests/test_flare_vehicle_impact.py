import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from recipes.flare_anchored import AnchoredFlare
from recipes.flare_vehicle_impact import ImpactFlare, LEVELS, audit_boxes
from recipes.flare7kstar import compose


class ImpactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        yy, xx = np.mgrid[:64, :64]; r2=(xx-32)**2+(yy-32)**2
        core=np.exp(-r2/8); surrounding=.3*np.exp(-r2/200)
        for name, signal in [('compound', np.clip(core+surrounding,0,1)), ('light',core)]:
            Image.fromarray(np.repeat((signal*255).astype(np.uint8)[:,:,None],3,axis=2)).save(self.root/(name+'.png'))
        self.bank=dict(patterns=[dict(id='synthetic', compound=dict(file='compound.png'),light_source=dict(file='light.png'))])
        self.rgb=np.full((96,160,3),30,np.uint8)
        self.sources=[dict(track_id=1,xy=[65.,30.],radius=4.,color=[.95,.9,.8],fade=1.,peak=.95,contrast=.7,forward_backward_error_pixels=0.)]
        self.depth=np.ones((96,160),np.float32)

    def tearDown(self):self.tmp.cleanup()

    def test_replay_matches_original_field_without_new_components(self):
        old=AnchoredFlare('unit',self.bank,self.root)
        new=ImpactFlare('unit',self.bank,self.root,spread=1.,halo_peak=0.)
        for _ in range(2):
            expected,metadata=old.field(self.rgb,self.depth,self.sources)
            actual,_=new.field(self.rgb,dict(tracking=dict(active_sources=self.sources),metadata=metadata))
            np.testing.assert_allclose(actual,expected,atol=2e-7)

    def test_severity_is_monotone_and_extends_beyond_source(self):
        old=AnchoredFlare('unit',self.bank,self.root);_,metadata=old.field(self.rgb,self.depth,self.sources)
        field,record=ImpactFlare('unit',self.bank,self.root).field(self.rgb,dict(tracking=dict(active_sources=self.sources),metadata=metadata))
        prior=self.rgb
        for strength in LEVELS.values():
            rendered=compose(self.rgb,field,strength)
            self.assertTrue(np.all(rendered>=prior));prior=rendered
        self.assertGreater(float(field[70,65].max()),.0001)
        self.assertFalse(record['source_positions_changed'])

    def test_no_source_creates_no_effect_and_audit_has_no_generation_control(self):
        flare=ImpactFlare('unit',self.bank,self.root)
        row=dict(tracking=dict(active_sources=[]),metadata=dict(sources=[],pattern_id=flare.template.pattern_id))
        field,_=flare.field(self.rgb,row)
        np.testing.assert_array_equal(compose(self.rgb,field,1.5),self.rgb)
        boxes=[dict(id='vehicle',category='car',xyxy=[10,10,50,50],original_occluded=False)]
        self.assertEqual(audit_boxes(self.rgb,self.rgb,boxes)[0]['changed_fraction'],0)


if __name__=='__main__':unittest.main()
