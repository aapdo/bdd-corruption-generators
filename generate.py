"""Portable diagnostic CPU preview; no cluster configuration or API credentials."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--type',required=True,choices=['Fog','LowLight','MotionBlur','Glare','Snow'])
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--level',type=float,required=True);p.add_argument('--depth',type=Path);p.add_argument('--flow',type=Path)
    p.add_argument('--video',default='preview');p.add_argument('--frame',type=int,default=0);p.add_argument('--seed',type=int,default=7301)
    p.add_argument('--bbox',type=float,nargs=4,help='Previously reviewed emitting-lamp box in ORIGINAL image coordinates')
    p.add_argument('--controlled',action='store_true',help='Permit and label an assumed off-frame sun or static Snow camera')
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).parent/'recipes'))
    import numpy as np
    from PIL import Image
    import cv2
    from appearance import lighting_profile,lowlight_linear,fog_scene
    from motion import target_blur
    from glare import sources_for_frame,render_glare
    from snow_scene import SceneSnowRenderer
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.output.exists()or a.output.with_suffix('.json').exists():raise ValueError('Refuse existing output')
    raw=np.asarray(Image.open(a.image).convert('RGB'));rgb=cv2.resize(raw,(640,360),interpolation=cv2.INTER_CUBIC)
    d=np.full((360,640),30,np.float32)
    if a.depth:
        d=np.load(a.depth,allow_pickle=False)
        if d.ndim!=2 or not np.isfinite(d).all()or np.any(d<=0):raise ValueError('Aligned finite positive camera-z depth required')
        d=cv2.resize(d.astype(np.float32),(640,360))
    elif a.type in('Fog','Snow'):raise ValueError('--depth required')
    K=np.array([[450.,0,320],[0,450.,180],[0,0,1.]])
    light=lighting_profile([(rgb,d)])
    if a.type=='LowLight':out,q=lowlight_linear(rgb,a.level,a.video,a.frame)
    elif a.type=='Fog':out,q=fog_scene(rgb,d,a.level,light)
    elif a.type=='MotionBlur':
        if not a.flow:raise ValueError('--flow NPZ required: flow640x360x2, confidence640x360 bool')
        with np.load(a.flow,allow_pickle=False)as f:out,q=target_blur(rgb,f['flow'],f['confidence'],a.level,a.video,K)
    elif a.type=='Glare':
        anchor=None
        if a.bbox:
            box=np.asarray(a.bbox)*[640/raw.shape[1],360/raw.shape[0],640/raw.shape[1],360/raw.shape[0]]
            anchor=dict(accepted=True,bbox=box.tolist())
        sources=sources_for_frame(rgb,light,anchor)
        if any(not s['verified']for s in sources)and not a.controlled:raise ValueError('Unverified light requires explicit --controlled')
        out,q=render_glare(rgb,sources,a.level)
    else:
        if not a.controlled:raise ValueError('Static Snow camera assumption requires --controlled')
        renderer=SceneSnowRenderer(a.seed,d.shape,K,light)
        value=renderer.render(rgb,d,np.eye(4),a.level,0.,camera_twist=np.zeros(6));out,q=value['rgb'],value['qa']
        q.update(mode='CONTROLLED',camera_assumption='stationary_single_snapshot')
    q.update(type=a.type,level=a.level,training_release_ready=False,depth_metric_accuracy_verified=False,
        intrinsics_assumed=True,full_video_temporal_qualified=False,new_image_API_review=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(out).save(a.output,quality=95,subsampling=0)
    with a.output.with_suffix('.json').open('x')as f:json.dump(q,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
