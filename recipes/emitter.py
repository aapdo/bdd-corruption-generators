"""Typed photometry and explicit identity/tracking evidence for emitting sources."""
import numpy as np
import cv2
import hashlib
from baseline_render import srgb_to_linear

TYPES=('tail_light','headlight','streetlight')


def source_binding(record,core):
    box=np.asarray(record['bbox'],dtype='<f8').tobytes()
    return hashlib.sha256(box+record['source_type'].encode()+core.astype(np.uint8).tobytes()).hexdigest()


def detect_emitter(linear,bbox,source_type):
    if source_type not in TYPES:raise ValueError('Unsupported source type')
    h,w=linear.shape[:2];box=np.asarray(bbox,float)
    if box.shape!=(4,)or not np.isfinite(box).all()or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
        raise ValueError('Source box must be nonempty and within the image')
    x0,y0,x1,y1=np.floor(box).astype(int);x1=max(x0+1,x1);y1=max(y0+1,y1)
    region=np.zeros((h,w),bool);region[y0:y1,x0:x1]=True
    R,G,B=np.moveaxis(linear,-1,0);Y=.2126*R+.7152*G+.0722*B
    if source_type=='tail_light':candidate=(R>.3)&(R>1.6*G)&(R>1.6*B)
    elif source_type=='headlight':candidate=(Y>.25)&(np.minimum.reduce([R,G,B])>.5*np.maximum.reduce([R,G,B]))
    else:candidate=Y>.25
    n,labels,stats,centers=cv2.connectedComponentsWithStats((candidate&region).astype(np.uint8),8)
    choices=[j for j in range(1,n)if 3<=stats[j,4]<=max(80,.002*h*w)]
    core=labels==max(choices,key=lambda j:stats[j,4])if choices else np.zeros((h,w),bool)
    ring=cv2.dilate(region.astype(np.uint8),np.ones((11,11),np.uint8))>0;ring&=~region
    measured=float(np.quantile(Y[core],.8))if core.any()else 0.
    ambient=float(np.median(Y[ring]))if ring.any()else 0.
    contrast=measured-ambient
    passed=bool(core.sum()>=3 and contrast>=.08 and measured/(ambient+.01)>=1.4 and box[3]<.87*h)
    return core,dict(photometric_pass=passed,source_type=source_type,core_pixels=int(core.sum()),
        linear_luminance=measured,ambient_linear_luminance=ambient,contrast=contrast,
        thresholds_calibrated=False,box_excludes_hood=bool(box[3]<.87*h))


class LampTracker:
    """An adjacent-frame photometric/flow check; initialization is never a tracking pass."""
    def __init__(self):self.previous=None

    def observe(self,rgb,timestamp,record):
        if not np.isfinite(timestamp):raise ValueError('Finite source timestamp required')
        linear=srgb_to_linear(rgb.astype(np.float32)/255)
        mask,photo=detect_emitter(linear,record['bbox'],record['source_type'])
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);passed=False;error=None
        identity=record.get('track_id');center=None
        if mask.any():
            yy,xx=np.nonzero(mask);center=np.array([xx.mean(),yy.mean()],np.float32)
        old=self.previous
        if old is not None:
            dt=timestamp-old['timestamp']
            if dt<=0:raise ValueError('Source tracking timestamps must increase')
            if identity and identity==old['track_id']and record['source_type']==old['source_type']and dt<=.5 and center is not None and old['center'] is not None:
                p=old['center'][None,None,:]
                q,status,_=cv2.calcOpticalFlowPyrLK(old['gray'],gray,p,None,winSize=(15,15),maxLevel=2)
                if q is not None and status[0,0]:
                    back,s2,_=cv2.calcOpticalFlowPyrLK(gray,old['gray'],q,None,winSize=(15,15),maxLevel=2)
                    if back is not None and s2[0,0]:
                        error=float(np.linalg.norm(back-p));position=float(np.linalg.norm(q[0,0]-center))
                        passed=bool(error<=1. and position<=3. and old['photo']['photometric_pass']and photo['photometric_pass'])
        self.previous=dict(gray=gray,timestamp=float(timestamp),center=center,photo=photo,track_id=identity,source_type=record['source_type'])
        return dict(passed=passed,method='adjacent_LK_forward_backward_and_typed_photometry',track_id=identity,
            initialized=old is None,forward_backward_error_pixels=error,frames_checked=1 if old is None else 2,
            source_type=record['source_type'],rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_binding=source_binding(record,mask),
            observed_bbox=list(record['bbox']),timestamp_s=float(timestamp))


def select_sources(rgb,light,record=None,allow_controlled_sun=False):
    linear=srgb_to_linear(rgb.astype(np.float32)/255);h,w=rgb.shape[:2];sources=[]
    records=[]if record is None else(record if isinstance(record,list)else[record])
    for r in records:
        if 'source_type'not in r:raise ValueError('Explicit source_type required; bbox/accepted alone does not verify an emitter')
        mask,photo=detect_emitter(linear,r['bbox'],r['source_type'])
        if not photo['photometric_pass']:continue
        identity=bool(r.get('verified_by_user')or r.get('identity_verification',{}).get('passed',False))
        tracking=r.get('tracking',{});tracked=bool(tracking.get('passed',False)and tracking.get('track_id')==r.get('track_id')and r.get('track_id')
            and tracking.get('frames_checked',0)>=2 and tracking.get('source_type')==r['source_type']
            and tracking.get('rgb_sha256')==hashlib.sha256(rgb.tobytes()).hexdigest()and tracking.get('source_binding')==source_binding(r,mask))
        color=linear[mask].mean(axis=0);color/=max(float(color.max()),1e-6);yy,xx=np.nonzero(mask)
        sources.append(dict(type=r['source_type'],xy=[float(xx.mean()),float(yy.mean())],color=color.tolist(),color_space='linear_RGB',
            radius=float(np.clip(np.sqrt(mask.sum()/np.pi),1,8)),radiance=float(np.quantile(linear[mask].max(axis=-1),.8)),
            identity_verified=identity,photometric_pass=True,tracking_pass=tracked,tracking=tracking,
            identity_verification=r.get('identity_verification',{'method':'user_explicit_flag'if r.get('verified_by_user')else'none'}),
            track_id=r.get('track_id'),verified=bool(identity and tracked),photometry=photo))
    if not sources and allow_controlled_sun and light.get('controlled_sun_eligible',False)and light.get('scene_regime')!='night':
        x=-.12*w if light['sky_light_xy'][0]<w/2 else 1.12*w
        sources=[dict(type='controlled_off_frame_sun',xy=[x,-.08*h],color=srgb_to_linear(np.asarray(light['airlight_srgb'],np.float32)).tolist(),
            color_space='linear_RGB',radiance=1.,radius=.6*max(h,w),verified=False,identity_verified=False,tracking_pass=False,
            assumed_sun_source=True,eligibility=light['sun_eligibility_evidence'],direction_fixed_per_video=True)]
    return sources
