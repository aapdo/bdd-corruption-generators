"""Background-residual motion checks and measured blur, with explicit shake cases."""
import math
import numpy as np
import cv2
from motion_estimator import best_effort_motion
from geometry import se3_log
from baseline_render import estimate_dis_pair,srgb_to_linear,linear_to_uint8,pixel_grid,sample_bilinear
from appearance import regions,seed_for


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


def target_blur(rgb,flow,confidence,target_p90,video,K,dt=.2):
    if not 0<=target_p90<=12 or flow.shape!=(*rgb.shape[:2],2):raise ValueError('Invalid target blur input')
    if not np.isfinite(flow).all()or not np.isfinite(dt)or dt<=0:raise ValueError('Finite flow and positive interval required')
    h,w=rgb.shape[:2];yy,xx=np.mgrid[:h,:w];supported=confidence&(yy<h*.87);reliable=supported.sum()>=50
    roi=supported if reliable else yy<h*.87
    magnitudes=np.linalg.norm(flow,axis=-1);raw=float(np.quantile(magnitudes[supported],.9))if reliable else 0.
    desired=target_p90/max(raw,1e-9)*dt
    mode='observed_motion';shake=0.
    if target_p90==0:return rgb.copy(),dict(target_p90_pixels=0.,actual_p90_pixels=0.,mode=mode)
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
    raw_length=np.linalg.norm(D,axis=-1);D*=np.minimum(1.,20/np.maximum(raw_length,1e-9))[...,None]
    length=np.linalg.norm(D,axis=-1);samples=max(9,int(math.ceil(float(length.max())*2))|1)
    linear=srgb_to_linear(rgb.astype(np.float32)/255);grid=pixel_grid(h,w);total=np.zeros_like(linear);den=np.zeros((h,w),np.float32)
    for t in np.linspace(-.5,.5,samples):
        value,valid=sample_bilinear(linear,grid-t*D);total+=value*valid[...,None];den+=valid
    out=linear_to_uint8(total/den[...,None])
    return out,dict(mode=mode,target_p90_pixels=float(target_p90),actual_p90_pixels=float(np.quantile(length[roi],.9)),
        max_blur_pixels=float(length.max()),length_quantiles_pixels=np.quantile(length,[0,.5,.9,.99,1]).tolist(),
        observed_flow_p90_pixels=raw,desired_observed_exposure_ms=float(desired*1000),effective_exposure_ms=float(exposure*1000),
        assumed_shake_p90_pixels=shake,cap_pixels=20,samples=samples,valid_flow_fraction=float(confidence.mean()),observed_support_pixels=int(supported.sum()),
        camera_shake_is_observed=False if mode=='controlled_camera_shake'else None,
        method='linear_RGB_adaptive_quadrature_with_explicit_controlled_rotational_shake_when_needed')
