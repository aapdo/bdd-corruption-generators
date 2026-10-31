"""Linear-light exposure and fixed per-video illumination; no Bayer resampling."""
import hashlib
import numpy as np
import cv2
from baseline_render import srgb_to_linear,linear_to_uint8,fog
from input_validation import rgb_image


def seed_for(video,frame=0):return int.from_bytes(hashlib.sha256(f'improved-r1:{video}:{frame}'.encode()).digest()[:8],'big')


def regions(rgb,depth):
    """Heuristic context masks, never described as semantic ground truth."""
    rgb_image(rgb)
    if depth.ndim!=2 or depth.shape!=rgb.shape[:2]:raise ValueError('Depth must align with RGB')
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


def lighting_profile(frames,time_of_day=None):
    if not frames:raise ValueError('At least one aligned RGB/depth pair is required')
    colors=[];sky_fraction=[];directions=[];scene_luma=[];sky_luma=[]
    for rgb,depth in frames:
        r=regions(rgb,depth);mask=r['sky'];h,w=depth.shape
        luma=rgb.astype(np.float32)@np.asarray([.2126,.7152,.0722])/255
        scene_luma.append(float(np.median(luma)));sky_luma.append(float(np.median(luma[mask]))if mask.any()else 0.)
        if mask.sum()>=64:
            pixels=rgb[mask].astype(np.float32)/255
            color=np.quantile(pixels,.75,axis=0)
            weights=np.maximum(luma-np.quantile(luma[mask],.8),0)*mask
            yy,xx=np.mgrid[:h,:w]
            if weights.sum()>1e-8:directions.append([float((xx*weights).sum()/weights.sum()),float((yy*weights).sum()/weights.sum())])
        else:color=np.quantile(rgb[:max(1,h//5)].reshape(-1,3)/255,.75,axis=0)
        colors.append(color);sky_fraction.append(float(mask.mean()))
    luminance=float(np.median(scene_luma));sky_brightness=float(np.median(sky_luma))
    night=time_of_day=='night'or(luminance<.10 and sky_brightness<.20)
    regime='night'if night else('twilight'if time_of_day in('dawn/dusk','dusk','dawn')or luminance<.25 else'day')
    # A dark scene is allowed to have dark airlight; never impose the old .48 floor.
    A=np.clip(np.median(colors,axis=0),.002,.99)
    xy=np.median(directions,axis=0).tolist()if directions else[frames[0][0].shape[1]*.25,0.]
    support=sum(f>=.03 for f in sky_fraction);required=max(1,int(np.ceil(len(frames)*.6)))
    eligible=bool(not night and support>=required and sky_brightness>=.20)
    return dict(airlight_srgb=A.tolist(),mean_sky_fraction=float(np.mean(sky_fraction)),sky_light_xy=xy,
        source='fixed_clip_median_heuristic_sky_or_upper_background',calibrated_illumination=False,
        scene_regime=regime,scene_luma_srgb=luminance,sky_luma_srgb=sky_brightness,
        sky_supported_frames=support,profile_frames=len(frames),controlled_sun_eligible=eligible,
        sun_eligibility_evidence='daylight_plus_repeated_top_connected_sky_proxy_not_observed_sun',
        verified_sky=False,airlight_floor=.002,spatially_varying_night_scatter_modeled=False)


def lowlight_linear(rgb,ev,video,frame,noise=True,noise_model='independent',rho=.85):
    rgb_image(rgb)
    if not np.isfinite(ev)or not -4<=ev<=0:raise ValueError('EV outside [-4,0]')
    if noise_model not in('exposure_only','independent','correlated','fixed_pattern'):raise ValueError('Unknown noise model')
    if not np.isfinite(rho)or not 0<=rho<=1:raise ValueError('Channel correlation must be in[0,1]')
    if not noise:noise_model='exposure_only'
    if ev==0:return rgb.copy(),dict(ev=0.,exposure_multiplier=1.,noise=False,noise_model=noise_model)
    L=srgb_to_linear(rgb.astype(np.float32)/255)*2**float(ev)
    profile=np.random.default_rng(seed_for(video));a=float(profile.uniform(1/16000,1/6000));b=float(profile.uniform(.5e-7,2e-7))
    variance=a*L+b;realized=L.copy();fixed=np.zeros_like(L)
    rng=np.random.default_rng(seed_for(video,frame))
    if noise_model=='correlated':
        common=rng.normal(size=(*L.shape[:2],1));channel=rng.normal(size=L.shape)
        realized+=np.sqrt(variance)*(np.sqrt(rho)*common+np.sqrt(1-rho)*channel)
    elif noise_model in('independent','fixed_pattern'):
        realized+=np.sqrt(a*L)*rng.normal(size=L.shape)+np.sqrt(b)*rng.normal(size=L.shape)
        if noise_model=='fixed_pattern':
            camera=np.random.default_rng(seed_for(video,2**31-1))
            fixed=.0025*L*camera.normal(size=L.shape)+.25*np.sqrt(b)*camera.normal(size=L.shape)
            realized+=fixed
    result=linear_to_uint8(realized)
    realized_noise=srgb_to_linear(result.astype(np.float32)/255)-L
    rms=float(np.sqrt(np.mean(realized_noise**2)))
    return result,dict(method='linear_RGB_exposure_with_explicit_noise_components',ev=float(ev),noise_model=noise_model,
        exposure_multiplier=2**float(ev),shot_coefficient=a,read_variance=b,channel_correlation=rho if noise_model=='correlated'else 0.,
        noise_variance_mean=float(variance.mean())if noise_model!='exposure_only'else 0.,noise_seed=seed_for(video,frame),noise=noise_model!='exposure_only',
        shot_noise_distribution='Gaussian_approximation_in_processed_linear_RGB',fixed_pattern_camera_seed=seed_for(video,2**31-1)if noise_model=='fixed_pattern'else None,
        fixed_pattern_rms=float(np.sqrt(np.mean(fixed**2))),realized_linear_error_rms=rms,
        realized_signal_to_noise=float(np.sqrt(np.mean(L**2))/max(rms,1e-12)),
        no_Bayer_or_demosaic=True,calibrated_sensor=False,mean_linear_signal=float(L.mean()),
        estimated_signal_to_noise=float(np.sqrt(np.mean(L*L)/np.mean(variance))))


def fog_scene(rgb,depth,beta,light,sky_mask=None):
    if not np.isfinite(beta)or not 0<=beta<=.1:raise ValueError('Fog beta must be finite in[0,.1]')
    r=regions(rgb,depth);valid=np.isfinite(depth)&(depth>0)&(depth<=120);d=np.where(valid,depth,0).astype(np.float32)
    sky=np.zeros(depth.shape,bool)if sky_mask is None else np.asarray(sky_mask)
    if sky.shape!=depth.shape or sky.dtype!=np.bool_:raise ValueError('Verified Fog sky mask must be aligned boolean')
    d[sky]=90.;usable=valid|sky
    out,t=fog(rgb,d,1.,beta_max=beta,airlight_srgb=light['airlight_srgb'])
    out[~usable]=rgb[~usable];t[~usable]=1
    bands=[]
    for a,b in [(0,10),(10,30),(30,60),(60,120)]:
        mask=valid&(depth>a)&(depth<=b)
        bands.append(dict(estimated_depth_range=[a,b],pixels=int(mask.sum()),mean_transmission=float(t[mask].mean())if mask.any()else None))
    return out,dict(beta=float(beta),airlight_srgb=light['airlight_srgb'],mean_transmission=float(t.mean()),
        transmission_quantiles=np.quantile(t,[.1,.5,.9]).tolist(),sky_fraction=float(r['sky'].mean()),
        sky_depth_policy='original_depth_unless_explicit_verified_sky_90m_proxy',verified_sky_fraction=float(sky.mean()),metric_scale_verified=False,
        stress_test=beta>=.03,lighting_fixed_per_video=True,scene_regime=light.get('scene_regime','unknown'),
        estimated_depth_bands=bands,valid_depth_fraction=float(valid.mean()),depth_scale_verified=False,
        depth_quantiles=np.quantile(depth[valid],[.1,.5,.9]).tolist()if valid.any()else[],
        unknown_depth_policy='preserve_original',depth_registration_verified=False)
