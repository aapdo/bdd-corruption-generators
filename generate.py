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
    p.add_argument('--bbox',type=float,nargs=4,help='Emitter candidate box in ORIGINAL coordinates; does not imply verification')
    p.add_argument('--source-type',choices=['tail_light','headlight','streetlight'])
    p.add_argument('--source-record',type=Path,help='Explicit identity and adjacent-frame tracking evidence JSON')
    p.add_argument('--eligibility',type=Path,help='Glare frame decision or video manifest from the separate source filter')
    p.add_argument('--verified-by-user',action='store_true',help='Explicitly confirm source identity; tracking is still independently required')
    p.add_argument('--verified-sky',type=Path,help='Explicitly verified aligned boolean sky NPY; unknown depth alone is not sky')
    p.add_argument('--scene-haze-beta-max',type=float,default=.004)
    p.add_argument('--noise-model',choices=['exposure_only','independent','correlated','fixed_pattern'],default='independent')
    p.add_argument('--temporal-exposure-ms',type=float);p.add_argument('--timestamp-s',type=float,default=0.)
    p.add_argument('--controlled',action='store_true',help='Permit and label an assumed off-frame sun or static Snow camera')
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).parent/'recipes'))
    import numpy as np
    from PIL import Image
    import cv2
    from appearance import lighting_profile,lowlight_linear,fog_scene
    from motion import target_blur,TemporalExposureRenderer
    from glare import sources_for_frame,render_glare,render_from_eligibility
    from snow_scene import SceneSnowRenderer
    from input_validation import aligned_depth,aligned_mask
    from sample_io import write_sample
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.output.exists()or a.output.with_suffix('.json').exists():raise ValueError('Refuse existing output')
    raw=np.asarray(Image.open(a.image).convert('RGB'));rgb=cv2.resize(raw,(640,360),interpolation=cv2.INTER_CUBIC)
    if raw.shape[:2]not in((720,1280),(360,640)):raise ValueError('BDD RGB must be1280x720 or640x360; intrinsics are assumed for this aspect ratio')
    d=np.full((360,640),30,np.float32)
    if a.depth:
        d=np.load(a.depth,allow_pickle=False)
        d=aligned_depth(d,raw.shape)
    elif a.type in('Fog','Snow'):raise ValueError('--depth required')
    K=np.array([[450.,0,320],[0,450.,180],[0,0,1.]])
    light=lighting_profile([(rgb,d)])
    sky=aligned_mask(np.load(a.verified_sky,allow_pickle=False),raw.shape)if a.verified_sky else None
    if a.type=='LowLight':out,q=lowlight_linear(rgb,a.level,a.video,a.frame,noise_model=a.noise_model)
    elif a.type=='Fog':out,q=fog_scene(rgb,d,a.level,light,sky_mask=sky)
    elif a.type=='MotionBlur':
        if not a.flow:raise ValueError('--flow NPZ required: flow640x360x2, confidence640x360 bool')
        with np.load(a.flow,allow_pickle=False)as f:
            if a.temporal_exposure_ms is None:out,q=target_blur(rgb,f['flow'],f['confidence'],a.level,a.video,K)
            else:out,q=TemporalExposureRenderer(K,a.temporal_exposure_ms,a.video).render(rgb,f['flow'],f['confidence'],a.timestamp_s)
    elif a.type=='Glare':
        if not a.eligibility:raise ValueError('Separate --eligibility decision required before Glare synthesis')
        if a.bbox or a.source_record or a.verified_by_user or a.source_type:raise ValueError('Legacy direct-source flags are replaced by the eligibility record')
        decision=json.loads(a.eligibility.read_text())
        if 'frames'in decision:
            matches=[r for r in decision['frames']if r['frame_index']==a.frame]
            if len(matches)!=1:raise ValueError('Exactly one eligibility record for --frame is required')
            decision=matches[0]
        if decision.get('native_sha256')!=__import__('hashlib').sha256(a.image.read_bytes()).hexdigest():raise ValueError('Eligibility source image SHA differs')
        out,q=render_from_eligibility(rgb,decision,a.level,allow_controlled=a.controlled)
    else:
        if not a.controlled:raise ValueError('Static Snow camera assumption requires --controlled')
        renderer=SceneSnowRenderer(a.seed,d.shape,K,light,scene_haze_beta_max=a.scene_haze_beta_max)
        value=renderer.render(rgb,d,np.eye(4),a.level,0.,camera_twist=np.zeros(6),sky_mask=sky);out,q=value['rgb'],value['qa']
        q.update(mode='CONTROLLED',camera_assumption='stationary_single_snapshot')
    q.update(type=a.type,level=a.level,training_release_ready=False,depth_metric_accuracy_verified=False,
        intrinsics_assumed=True,full_video_temporal_qualified=False,new_image_API_review=False)
    write_sample(a.output,out,q)


if __name__=='__main__':main()
