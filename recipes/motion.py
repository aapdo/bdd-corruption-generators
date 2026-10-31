"""Background-residual motion checks and measured blur, with explicit shake cases."""
import math
import numpy as np
import cv2
from motion_estimator import best_effort_motion
from geometry import se3_log
from baseline_render import estimate_dis_pair,srgb_to_linear,linear_to_uint8,pixel_grid,sample_bilinear
from appearance import regions,seed_for
from input_validation import flow_inputs
from blur_quality import sharpness_loss


def rigid_flow(T,depth,K):
    h,w=depth.shape;yy,xx=np.mgrid[:h,:w]
    rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones((h,w))],-1)
    points=rays*depth[...,None];warped=points@T[:3,:3].T+T[:3,3]
    z=np.maximum(warped[...,2],1e-6)
    xy=np.stack([warped[...,0]*K[0,0]/z+K[0,2],warped[...,1]*K[1,1]/z+K[1,2]],-1)
    return(xy-np.stack([xx,yy],-1)).astype(np.float32)


def estimate_scene_motion(a,b,depth,K,dt=.2):
    K=np.asarray(K,np.float64);r=regions(a,depth);d=depth.copy();d[r['hood']|r['sky']]=0
    initial=best_effort_motion(a,b,d,K,dt)
    observed,_,valid,_=estimate_dis_pair(a,b);pred=rigid_flow(initial['T'],depth,K)
    residual=np.linalg.norm(observed-pred,axis=-1)
    static=valid & (d>.5)&(r['gradient']>2)&(residual<2.)
    d[~static]=0
    main=best_effort_motion(a,b,d,K,dt);q=main['qa']
    check=np.linalg.norm(observed-rigid_flow(main['T'],depth,K),axis=-1)
    support=static&(check<3.);n=int(support.sum())
    spatial=sum(bool(support[y:y+max(1,depth.shape[0]//3),x:x+max(1,depth.shape[1]//3)].any())for y in range(0,depth.shape[0],max(1,depth.shape[0]//3))for x in range(0,depth.shape[1],max(1,depth.shape[1]//3)))
    halves=[]
    for left in [True,False]:
        split=d.copy()
        if left:split[:,depth.shape[1]//2:]=0
        else:split[:,:depth.shape[1]//2]=0
        halves.append(best_effort_motion(a,b,split,K,dt))
    discrepancy=se3_log(halves[0]['T']@np.linalg.inv(halves[1]['T']))
    med=float(np.median(depth[support]))if n else 1.
    rot=float(np.linalg.norm(discrepancy[:3])*180/np.pi);trans=float(np.linalg.norm(discrepancy[3:])/max(med,1e-6))
    median=float(np.median(check[support]))if n else None
    accepted=bool(n>=80 and spatial>=6 and median is not None and median<=1.5 and rot<=.5 and trans<=.03
        and not q['fallback_assumption']and not q['twist_capped']and all(not x['qa']['fallback_assumption']and not x['qa']['twist_capped']for x in halves))
    q.update(validated_motion=accepted,validation_profile='RGB_heuristic_background_residual_and_disjoint_half_fit_v1',
        background_mask_is_semantic_GT=False,dynamic_residual_rejected_fraction=float((valid&~static).mean()),
        static_support_pixels=n,background_spatial_cells=spatial,median_reprojection_pixels=median,
        half_rotation_disagreement_deg=rot,half_translation_over_depth=trans,
        acceptance_profile_is_new=True,prior_acceptance_records_preserved=True,
        rain_adapter='full_estimated_SE3_local_exposure_endpoint_correction',
        motion_input_status='estimated_6DoF'if accepted else'unverified_best_effort_6DoF')
    return main


def integrate_displacement(rgb,D):
    length=np.linalg.norm(D,axis=-1);D=D*np.minimum(1.,20/np.maximum(length,1e-9))[...,None]
    length=np.linalg.norm(D,axis=-1);samples=max(9,int(math.ceil(float(length.max())*2))|1)
    linear=srgb_to_linear(rgb.astype(np.float32)/255);grid=pixel_grid(*rgb.shape[:2]);total=np.zeros_like(linear);den=np.zeros(rgb.shape[:2],np.float32)
    for t in np.linspace(-.5,.5,samples):
        value,valid=sample_bilinear(linear,grid-t*D);total+=value*valid[...,None];den+=valid
    return linear_to_uint8(total/den[...,None]),length,samples


def target_blur(rgb,flow,confidence,target_p90,video,K,dt=.2,boxes=None):
    flow,confidence=flow_inputs(flow,confidence,rgb.shape[:2])
    if not 0<=target_p90<=12 or flow.shape!=(*rgb.shape[:2],2):raise ValueError('Invalid target blur input')
    if not np.isfinite(flow).all()or not np.isfinite(dt)or dt<=0:raise ValueError('Finite flow and positive interval required')
    h,w=rgb.shape[:2];yy,xx=np.mgrid[:h,:w];supported=confidence&(yy<h*.87);reliable=supported.sum()>=50
    roi=supported if reliable else yy<h*.87
    magnitudes=np.linalg.norm(flow,axis=-1);raw=float(np.quantile(magnitudes[supported],.9))if reliable else 0.
    desired=target_p90/max(raw,1e-9)*dt
    mode='observed_motion';shake=0.
    if target_p90==0:return rgb.copy(),dict(target_p90_pixels=0.,actual_p90_pixels=0.,mode=mode,generation_mode='STATIC_TARGET_P90',
        sharpness=sharpness_loss(rgb,rgb,confidence,boxes),temporal_qualified=False)
    if desired<=.06 and raw>.1:
        exposure=desired;D=flow*(exposure/dt)*confidence[...,None]
    else:
        mode='controlled_camera_shake';exposure=.02;D=flow*(exposure/dt)*confidence[...,None]if reliable else np.zeros_like(flow)
        rng=np.random.default_rng(seed_for(video));angle=float(rng.uniform(-.6,.6))
        normalized=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones((h,w))],-1)
        omega=np.array([math.sin(angle),math.cos(angle),.05])/max(K[0,0],K[1,1])
        rotated=normalized@cv2.Rodrigues(omega)[0].T
        basis=np.stack([rotated[...,0]/rotated[...,2]*K[0,0]+K[0,2]-xx,rotated[...,1]/rotated[...,2]*K[1,1]+K[1,2]-yy],-1)
        basis/=max(float(np.quantile(np.linalg.norm(basis,axis=-1)[roi],.9)),1e-6)
        lo,hi=0.,30.
        for _ in range(28):
            mid=(lo+hi)/2
            if np.quantile(np.linalg.norm(D+basis*mid,axis=-1)[roi],.9)<target_p90:lo=mid
            else:hi=mid
        shake=(lo+hi)/2;D=D+basis*shake
    out,length,samples=integrate_displacement(rgb,D)
    return out,dict(mode=mode,target_p90_pixels=float(target_p90),actual_p90_pixels=float(np.quantile(length[roi],.9)),
        max_blur_pixels=float(length.max()),length_quantiles_pixels=np.quantile(length,[0,.5,.9,.99,1]).tolist(),
        observed_flow_p90_pixels=raw,desired_observed_exposure_ms=float(desired*1000),effective_exposure_ms=float(exposure*1000),
        assumed_shake_p90_pixels=shake,cap_pixels=20,samples=samples,valid_flow_fraction=float(confidence.mean()),observed_support_pixels=int(supported.sum()),
        camera_shake_is_observed=False if mode=='controlled_camera_shake'else None,
        method='linear_RGB_adaptive_quadrature_with_explicit_controlled_rotational_shake_when_needed',
        generation_mode='STATIC_TARGET_P90',sharpness=sharpness_loss(rgb,out,confidence,boxes),temporal_qualified=False)


def exposure_plan(flow_p90_pixels,targets=(1,2,3,4,6,8,12),dt=.2):
    values=np.asarray(flow_p90_pixels,float)
    if values.ndim!=1 or not values.size or not np.isfinite(values).all()or np.any(values<0)or not np.isfinite(dt)or dt<=0:raise ValueError('Valid segment flow statistics required')
    median=float(np.median(values));plan=[]
    for target in targets:
        if not np.isfinite(target)or target<0:raise ValueError('Invalid blur target')
        desired=1000*target/max(median,1e-9)*dt
        plan.append(dict(reference_target_p90_pixels=float(target),exposure_ms=float(min(60,desired)),
            target_attainable_from_reference=bool(median>0 and desired<=60),reference_flow_p90=median))
    return dict(generation_mode='TEMPORAL_EXPOSURE',segment_reference_frames=len(values),plan=plan,
        exposure_constant_within_segment=True,automatic_shake_fallback=False,temporal_qualified=False)


class TemporalExposureRenderer:
    """Fixed segment exposure and an optional continuous, explicitly virtual camera path."""
    def __init__(self,K,exposure_ms,video,shake_position_amplitude_px=0.,shake_hz=1.2):
        self.K=np.asarray(K,np.float64)
        if self.K.shape!=(3,3)or not np.isfinite(self.K).all()or min(self.K[0,0],self.K[1,1])<=0:raise ValueError('Valid K required')
        if not np.isfinite(exposure_ms)or not 0<=exposure_ms<=60:raise ValueError('Fixed exposure must be0..60ms')
        if not np.isfinite(shake_position_amplitude_px)or not 0<=shake_position_amplitude_px<=30 or not np.isfinite(shake_hz)or not 0<shake_hz<=10:raise ValueError('Invalid continuous shake trajectory')
        self.exposure_s=exposure_ms/1000;self.amplitude=shake_position_amplitude_px;self.frequency=shake_hz
        self.phase=np.random.default_rng(seed_for(video)).uniform(0,2*np.pi,2);self.last=None

    def position(self,t):
        return self.amplitude*np.array([np.sin(2*np.pi*self.frequency*t+self.phase[0]),np.sin(2*np.pi*.7*self.frequency*t+self.phase[1])])

    def render(self,rgb,flow,confidence,timestamp_s,dt=.2,boxes=None):
        flow,confidence=flow_inputs(flow,confidence,rgb.shape[:2])
        if not np.isfinite(timestamp_s)or(self.last is not None and timestamp_s<=self.last)or not np.isfinite(dt)or dt<=0 or self.exposure_s>dt:
            raise ValueError('Increasing timestamps and interval at least as long as exposure required')
        h,w=rgb.shape[:2];yy,xx=np.mgrid[:h,:w]
        # Erode low-confidence object/occlusion boundaries rather than warp untrusted flow there.
        safe=cv2.erode(confidence.astype(np.uint8),np.ones((3,3),np.uint8))>0
        observed=flow*(self.exposure_s/dt)*safe[...,None]
        trajectory_delta=self.position(timestamp_s+self.exposure_s/2)-self.position(timestamp_s-self.exposure_s/2)
        omega=np.array([trajectory_delta[1]/self.K[1,1],trajectory_delta[0]/self.K[0,0],0.])
        rays=np.stack([(xx-self.K[0,2])/self.K[0,0],(yy-self.K[1,2])/self.K[1,1],np.ones((h,w))],-1)
        rotated=rays@cv2.Rodrigues(omega)[0].T
        shake=np.stack([rotated[...,0]/rotated[...,2]*self.K[0,0]+self.K[0,2]-xx,rotated[...,1]/rotated[...,2]*self.K[1,1]+self.K[1,2]-yy],-1)
        out,length,samples=integrate_displacement(rgb,observed+shake);self.last=float(timestamp_s)
        return out,dict(generation_mode='TEMPORAL_EXPOSURE',mode='controlled_camera_shake'if self.amplitude else'observed_motion',
            effective_exposure_ms=self.exposure_s*1000,actual_p90_pixels=float(np.quantile(length,.9)),max_blur_pixels=float(length.max()),
            observed_displacement_p90_pixels=float(np.quantile(np.linalg.norm(observed,axis=-1),.9)),
            virtual_shake_delta_pixels=trajectory_delta.tolist(),shake_position_amplitude_px=float(self.amplitude),shake_hz=float(self.frequency),
            fixed_exposure=True,automatic_shake_fallback=False,low_confidence_flow_policy='eroded_confidence_zero_observed_displacement',
            samples=samples,cap_pixels=20,sharpness=sharpness_loss(rgb,out,safe,boxes),temporal_qualified=False)
