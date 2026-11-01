"""Offline Wu-pattern, source-aligned RGB sequence CLI; no model/API calls."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from recipes.flare_wu_day import WuFlare,LEVELS,track_sun,compose

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.manifest.read_text());pattern=config['pattern'];frames=config['frames']
    if args.output.exists():raise ValueError('Output already exists; use a fresh attempt directory')
    if hashlib.sha256(Path(pattern['path']).read_bytes()).hexdigest()!=pattern['sha256']:raise ValueError('Pattern SHA differs')
    images=[np.asarray(Image.open(p).convert('RGB'))for p in frames]
    if not images or any(im.shape!=images[0].shape for im in images):raise ValueError('Input RGB geometry differs')
    sources=track_sun(images,config['anchor_index'],config['reviewed_sun_xy'])
    flare=WuFlare(pattern['path'],pattern['origin'],pattern['kind']);args.output.mkdir(parents=True);records=[]
    for i,(rgb,source)in enumerate(zip(images,sources)):
        field,metadata=flare.field(rgb.shape,source);outputs={}
        for name,gain in LEVELS.items():
            target=args.output/f'{i:06d}_{name}.png';Image.fromarray(compose(rgb,field,gain)).save(target)
            outputs[name]=dict(path=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        records.append(dict(index=i,input_sha256=hashlib.sha256(Path(frames[i]).read_bytes()).hexdigest(),sun=source,render=metadata,outputs=outputs))
    result=dict(pattern=pattern,levels=LEVELS,frames=records,model_API_calls=0,training_release_ready=False,
        method='Wu equations NumPy port plus explicit BDD sun alignment/tracking; original RGB exposure and geometry preserved')
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(state='generated',frames=len(records),images=len(records)*3,model_API_calls=0)))

if __name__=='__main__':main()
