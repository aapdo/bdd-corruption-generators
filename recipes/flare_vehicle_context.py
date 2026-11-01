"""Whole detected-vehicle context check with the already loaded local CLIP."""
from PIL import Image
import numpy as np
from recipes.flare_auto_sources import LocalClipSources

POSITIVE=['a photo of the side of a car on a road','a photo of the front of a car',
          'a photo of the back of a car','a photo of a bus','a photo of a truck']
NEGATIVE=['a metal fence and railings','a roadside guard rail or barrier',
          'a building facade with windows','an illuminated shopfront or shop sign',
          'a pile of snow or rocks beside a road','an empty road and sidewalk',
          'trees and bushes beside a road']


class VehicleAwareClip(LocalClipSources):
    def __init__(self,ready,request):
        super().__init__(ready,request)
        from transformers import CLIPTokenizerFast
        tokenizer=CLIPTokenizerFast.from_pretrained(ready['path'],local_files_only=True)
        tokens=tokenizer(POSITIVE+NEGATIVE,return_tensors='pt',padding=True,truncation=True).to('cuda')
        with self.torch.inference_mode():
            labels=self.model.get_text_features(**tokens);self.vehicle_labels=labels/labels.norm(dim=-1,keepdim=True)
        self.context_queries=0

    def score_vehicle_boxes(self,queries):
        crops=[]
        for rgb,box in queries:
            h,w=rgb.shape[:2];x0,y0,x1,y1=box
            pad=.06*max(x1-x0,y1-y0)
            bounds=(max(0,int(x0-pad)),max(0,int(y0-pad)),min(w,int(x1+pad)+1),min(h,int(y1+pad)+1))
            crops.append(Image.fromarray(rgb).crop(bounds))
        if not crops:return []
        matrices=[]
        for start in range(0,len(crops),32):
            inputs=self.processor(images=crops[start:start+32],return_tensors='pt')
            with self.torch.inference_mode():
                features=self.model.get_image_features(pixel_values=inputs['pixel_values'].to('cuda',dtype=self.torch.float32))
                features=features/features.norm(dim=-1,keepdim=True)
                matrices.append((features@self.vehicle_labels.T).cpu().numpy())
        scores=np.concatenate(matrices);rows=[]
        for score in scores:
            a=int(np.argmax(score[:len(POSITIVE)]));b=int(np.argmax(score[len(POSITIVE):]))
            positive=float(score[a]);negative=float(score[len(POSITIVE)+b]);margin=positive-negative
            rows.append(dict(accepted=positive>=.20 and margin>=.005,positive_cosine=positive,negative_cosine=negative,
                margin=margin,positive_text=POSITIVE[a],negative_text=NEGATIVE[b],score_is_probability=False))
        self.context_queries+=len(queries);return rows

    def stats(self):return dict(super().stats(),whole_vehicle_context_queries=self.context_queries)

    def close(self):del self.vehicle_labels;super().close()
