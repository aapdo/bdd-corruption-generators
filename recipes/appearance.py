"""Linear-light exposure and fixed per-video illumination; no Bayer resampling."""
import hashlib
import numpy as np
import cv2
from baseline_render import srgb_to_linear,linear_to_uint8,fog


def seed_for(video,frame=0):return int.from_bytes(hashlib.sha256(f'improved-r1:{video}:{frame}'.encode()).digest()[:8],'big')


def regions(rgb,depth):
    """Heuristic context masks, never described as semantic ground truth."""
    h,w=depth.shape;gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient=np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0)/8,cv2.Sobel(gray,cv2.CV_32F,0,1)/8)
    yy,xx=np.mgrid[:h,:w];valid=np.isfinite(depth)&(depth>0)
    far=np.quantile(depth[valid],.7)if valid.any()else 120
    mask=((yy<h*.5)&valid&(depth>=max(20,far))&(gradient<12)).astype(np.uint8)
    n,labels=cv2.connectedComponents(mask,8);touch=set(labels[:max(1,h//20)].ravel())-{0}
    sky=np.isin(labels,list(touch))
    hood=yy>=h*.87
    road=(yy>h*.52)&(yy<h*.86)&(np.abs(xx-w*.5)<(yy/h-.35)*w*.8)&valid
    return dict(sky=sky,road=road,hood=hood,gradient=gradient,
        method='RGB_depth_connected_top_region_heuristics_not_semantic_GT')


def lighting_profile(frames):
    colors=[];sky_fraction=[];directions=[]
    for rgb,depth in frames:
        r=regions(rgb,depth);mask=r['sky'];h,w=depth.shape
        if mask.sum()>=64:
            pixels=rgb[mask].astype(np.float32)/255
            color=np.quantile(pixels,.75,axis=0)
            luma=rgb.astype(np.float32)@np.asarray([.2126,.7152,.0722]);weights=np.maximum(luma-np.quantile(luma[mask],.8),0)*mask
            yy,xx=np.mgrid[:h,:w]
            directions.append([float((xx*weights).sum()/max(weights.sum(),1)),float((yy*weights).sum()/max(weights.sum(),1))])
        else:color=np.quantile(rgb[:max(1,h//5)].reshape(-1,3)/255,.75,axis=0)
        colors.append(color);sky_fraction.append(float(mask.mean()))
    A=np.clip(np.median(colors,axis=0),.48,.96)
    xy=np.median(directions,axis=0).tolist()if directions else[frames[0][0].shape[1]*.25,0.]
    return dict(airlight_srgb=A.tolist(),mean_sky_fraction=float(np.mean(sky_fraction)),sky_light_xy=xy,
        source='fixed_clip_median_heuristic_sky_or_upper_background',calibrated_illumination=False)


def lowlight_linear(rgb,ev,video,frame,noise=True):
    if not np.isfinite(ev)or not -4<=ev<=0:raise ValueError('EV outside [-4,0]')
    if ev==0:return rgb.copy(),dict(ev=0.,exposure_multiplier=1.,noise=False)
    L=srgb_to_linear(rgb.astype(np.float32)/255)*2**float(ev)
    profile=np.random.default_rng(seed_for(video));a=float(profile.uniform(1/16000,1/6000));b=float(profile.uniform(.5e-7,2e-7))
    variance=a*L+b;realized=L.copy();rho=.85
    if noise:
        rng=np.random.default_rng(seed_for(video,frame));common=rng.normal(size=(*L.shape[:2],1));channel=rng.normal(size=L.shape)
        realized+=np.sqrt(variance)*(np.sqrt(rho)*common+np.sqrt(1-rho)*channel)
    result=linear_to_uint8(realized)
    return result,dict(method='linear_RGB_exposure_plus_correlated_signal_dependent_noise',ev=float(ev),
        exposure_multiplier=2**float(ev),shot_coefficient=a,read_variance=b,channel_correlation=rho,
        noise_variance_mean=float(variance.mean()),noise_seed=seed_for(video,frame),noise=noise,
        no_Bayer_or_demosaic=True,calibrated_sensor=False,mean_linear_signal=float(L.mean()),
        estimated_signal_to_noise=float(np.sqrt(np.mean(L*L)/np.mean(variance))))


def fog_scene(rgb,depth,beta,light):
    r=regions(rgb,depth);d=depth.copy();d[r['sky']]=np.maximum(d[r['sky']],90.)
    out,t=fog(rgb,d,1.,beta_max=beta,airlight_srgb=light['airlight_srgb'])
    return out,dict(beta=float(beta),airlight_srgb=light['airlight_srgb'],mean_transmission=float(t.mean()),
        transmission_quantiles=np.quantile(t,[.1,.5,.9]).tolist(),sky_fraction=float(r['sky'].mean()),
        sky_depth_policy='at_least90_estimated_m_for_connected_sky_proxy',metric_scale_verified=False,
        stress_test=beta>=.03,lighting_fixed_per_video=True)
