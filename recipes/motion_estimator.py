"""Finite RGB/depth PnP proposal; quality acceptance is a separate operation."""
import numpy as np
import cv2
from baseline_render import estimate_dis_pair
from geometry import se3_log


def best_effort_motion(previous_rgb,current_rgb,previous_depth,K,dt=.2):
    if dt<=0 or not np.isfinite(dt):raise ValueError('Positive finite frame interval required')
    h,w=previous_depth.shape;K=np.asarray(K,np.float64)
    if previous_rgb.shape!=(h,w,3)or current_rgb.shape!=previous_rgb.shape or K.shape!=(3,3):raise ValueError('Aligned motion inputs required')
    flow,_,valid,_=estimate_dis_pair(previous_rgb,current_rgb)
    yy,xx=np.mgrid[:h,:w];target=np.stack([xx,yy],axis=-1)+flow
    mask=valid&np.isfinite(previous_depth)&(previous_depth>.5)&(previous_depth<120)
    mask&=(target[...,0]>=1)&(target[...,0]<w-2)&(target[...,1]>=1)&(target[...,1]<h-2)
    sample=np.zeros((h,w),bool);sample[4::8,4::8]=True
    gray=cv2.cvtColor(previous_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture=np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))>8
    y,x=np.nonzero(mask&sample&texture);qa=dict(validated_motion=False,correspondences=len(x),
        method='DIS_depth_PnP_proposal_acceptance_is_separate',fallback_assumption=None,camera_depth_metric_accuracy_verified=False)
    T=np.eye(4);success=False
    if len(x)>=8:
        z=previous_depth[y,x].astype(np.float64)
        points=np.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z],axis=1);cv2.setRNGSeed(7301)
        try:ok,r,t,ids=cv2.solvePnPRansac(points,target[y,x].astype(np.float64),K,None,iterationsCount=200,reprojectionError=3.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:ok,r,t,ids=False,None,None,None
        if ok and ids is not None and len(ids)>=6 and np.isfinite(r).all()and np.isfinite(t).all():
            inliers=ids.ravel()
            try:r,t=cv2.solvePnPRefineLM(points[inliers],target[y,x][inliers].astype(np.float64),K,None,r,t)
            except cv2.error:pass
            T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.ravel();success=bool(np.isfinite(T).all())
            qa.update(inliers=len(ids),inlier_fraction=float(len(ids)/len(x)))
    if not success:T=np.eye(4);qa['fallback_assumption']='assumed_stationary_camera_for_diagnostic_only'
    xi=se3_log(T)/dt;twist=np.r_[xi[3:],xi[:3]];raw_twist=twist.copy()
    for sl,limit in [(slice(0,3),50.),(slice(3,6),2.)]:
        norm=float(np.linalg.norm(twist[sl]))
        if norm>limit:twist[sl]*=limit/norm
    raw_speed=float(np.linalg.norm(T[:3,3])/dt*3.6);speed=float(min(144.,raw_speed))
    qa.update(raw_speed_kmh=raw_speed,used_speed_kmh=speed,raw_twist=raw_twist.tolist(),used_twist=twist.tolist(),
        twist_capped=not np.array_equal(raw_twist,twist),speed_capped=raw_speed>144,transform=T.tolist(),dt_seconds=dt)
    return dict(T=T,twist=twist,speed_kmh=speed,qa=qa)
