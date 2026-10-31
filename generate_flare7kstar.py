"""Standalone offline PGS video synthesis. Assets and model are supplied locally."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from recipes.flare7kstar import VideoFlare,compose,measurements,relative_depth,seed
from recipes.inference_batching import ordered_batches

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,help='JSON: video_id and time-ordered frames [{path}]')
    parser.add_argument('--patterns',type=Path,required=True,help='External paired-pattern MANIFEST.json directory')
    parser.add_argument('--model',type=Path,required=True,help='Local Intel/dpt-hybrid-midas checkpoint directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--batch-size',type=int,default=4,choices=[1,4,8])
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import DPTConfig,DPTForDepthEstimation,DPTImageProcessor
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(1101)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    processor=DPTImageProcessor.from_pretrained(args.model,local_files_only=True)
    model=DPTForDepthEstimation(DPTConfig.from_pretrained(args.model,local_files_only=True))
    model.load_state_dict(torch.load(args.model/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
    model=model.float().to(args.device).eval()
    manifest=json.loads(args.manifest.read_text());bank=json.loads((args.patterns/'MANIFEST.json').read_text())
    if len(bank['patterns'])!=5000:raise ValueError('All 5000 official scattering pairs required')
    rng=np.random.default_rng(seed(manifest['video_id']));chosen=bank['patterns'][int(rng.integers(5000))]
    for key in ['compound','light_source']:
        r=chosen[key];data=(args.patterns/r['file']).read_bytes()
        if len(data)!=r['bytes']or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Pattern integrity failure')
    args.output.mkdir(parents=True,exist_ok=False);renderer=None
    def reader(frame):
        data=Path(frame['path']).read_bytes()
        with Image.open(io.BytesIO(data))as image:rgb=np.asarray(image.convert('RGB'))
        if rgb.shape!=(720,1280,3):raise ValueError('Expected native MOT 1280x720')
        return data,rgb
    def predict_many(images):
        values=processor(images=list(images),return_tensors='pt')['pixel_values'].to(args.device,dtype=torch.float32)
        with torch.inference_mode():
            inverse=model(pixel_values=values).predicted_depth
            inverse=torch.nn.functional.interpolate(inverse.unsqueeze(1),size=images[0].shape[:2],mode='bicubic',align_corners=False)[:,0].cpu().numpy()
        return [relative_depth(value)for value in inverse]
    with (args.output/'FRAMES.jsonl').open('x')as ledger:
        for index,frame,data,rgb,depth in ordered_batches(manifest['frames'],reader,predict_many,args.batch_size):
            if renderer is None:renderer=VideoFlare(manifest['video_id'],bank,args.patterns)
            field,metadata=renderer.field(rgb,depth);metadata['inference_batch_size']=args.batch_size;variants={}
            for level,strength in [('weak',.25),('middle',.5),('strong',1.)]:
                result=compose(rgb,field,strength);target=args.output/level/(str(index).zfill(6)+'.jpg')
                target.parent.mkdir(exist_ok=True);Image.fromarray(result).save(target,quality=95,subsampling=0)
                variants[level]=dict(file=str(target.relative_to(args.output)),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),metrics=measurements(rgb,result))
            ledger.write(json.dumps(dict(frame_index=index,source_sha256=hashlib.sha256(data).hexdigest(),metadata=metadata,variants=variants,API_calls=0,human_reviewed=False))+'\n')

if __name__=='__main__':main()
