"""Correct official camera-local drop endpoints with estimated full SE3 exposure."""
import math
from pathlib import Path
from xml.etree.ElementTree import parse
import numpy as np
from common import se3_exp


def vector(text):return np.asarray([float(x)for x in text[1:-1].split(';')],np.float64)
def encoded(v):return '['+';'.join(format(float(x),'.10g')for x in v)+']'


def correct_stream(path,output,motions,base_speeds,exposure_s=.003,focal_pixels=900.,width=1280,height=720,depths=None):
    """Keep official rainfall particles; replace only the axial camera contribution.

    XML world coordinates are camera-local x-right,y-up,z-back. Images use
    bottom-left origin. This is a local constant-twist correction, not a full
    trajectory re-simulation or a claim of calibrated BDD camera motion.
    """
    tree=parse(path);counts=[];removed=0
    for index,frame in enumerate(tree.getroot()):
        if index>=len(motions):break
        twist=np.asarray(motions[index]['twist'],np.float64);target=se3_exp(twist*exposure_s)
        old_speed=float(base_speeds[index])/3.6;n=0
        for drop in list(frame):
            if 'wp1'not in drop.attrib:continue
            raw1=vector(drop.attrib['wp1']);raw2=vector(drop.attrib['wp2'])
            p1=raw1*np.asarray([1.,-1.,-1.]);p2=raw2*np.asarray([1.,-1.,-1.])
            # Remove the original forward-camera displacement from the drop path.
            rainfall_delta=p2-p1+np.asarray([0.,0.,old_speed*exposure_s])
            corrected=target[:3,:3]@(p1+rainfall_delta)+target[:3,3]
            if not np.isfinite(corrected).all()or min(p1[2],corrected[2])<=.01:
                frame.remove(drop);removed+=1;continue
            if depths is not None:
                visible=[]
                d=depths[index];dh,dw=d.shape
                for f in np.linspace(0,1,5):
                    q=p1+(corrected-p1)*f;u=focal_pixels*q[0]/q[2]+width/2;v=focal_pixels*q[1]/q[2]+height/2
                    x,y=int(u*dw/width),int(v*dh/height)
                    visible.append(bool(0<=x<dw and 0<=y<dh and d[y,x]>q[2]+.03))
                runs=[];start=None
                for j,value in enumerate(visible+[False]):
                    if value and start is None:start=j
                    if not value and start is not None:runs.append((start,j-1));start=None
                if not runs:
                    frame.remove(drop);removed+=1;continue
                a,b=max(runs,key=lambda r:r[1]-r[0]);origin=p1.copy();delta=corrected-origin
                p1=origin+delta*a/4;corrected=origin+delta*b/4
                if a==b:
                    frame.remove(drop);removed+=1;continue
                drop.set('wp1',encoded(p1*np.asarray([1.,-1.,-1.])))
                drop.set('ip1',encoded([focal_pixels*p1[0]/p1[2]+width/2,-focal_pixels*p1[1]/p1[2]+height/2]))
                drop.set('iw1',format(float(drop.attrib['wd1'])*focal_pixels/p1[2],'.10g'))
            xy=np.asarray([focal_pixels*corrected[0]/corrected[2]+width/2,
                -focal_pixels*corrected[1]/corrected[2]+height/2])
            drop.set('wp2',encoded(corrected*np.asarray([1.,-1.,-1.])))
            drop.set('ip2',encoded(xy));drop.set('iw2',format(float(drop.attrib['wd2'])*focal_pixels/corrected[2],'.10g'));n+=1
        frame.set('rs',str(n));counts.append(n)
    tree.write(output,encoding='utf-8',xml_declaration=True)
    return dict(method='camera_local_official_particle_SE3_exposure_endpoint_correction',frames=len(counts),
        particle_counts=counts,removed_invalid=removed,exposure_ms=exposure_s*1000,full_upstream_6DoF_resimulation=False,
        estimated_camera_motion=True,calibrated_metric_scale=False,scene_depth_occlusion_samples=5 if depths is not None else 0)


def compose_rain(rgb,depth,raw_rain,raw_fogged,mask,light,rate):
    from baseline_render import srgb_to_linear,linear_to_uint8
    import cv2
    source=srgb_to_linear(rgb.astype(np.float32)/255)
    # Upstream backgrounds use normalized display RGB; decode both exported layers.
    rainy=srgb_to_linear(np.clip(raw_rain[...,::-1],0,1).astype(np.float32))
    fogged=srgb_to_linear(np.clip(raw_fogged[...,::-1],0,1).astype(np.float32))
    residual=rainy-fogged
    coverage=np.clip(mask.astype(np.float32),0,1)
    if residual.shape!=source.shape:raise ValueError('Official Rain layer geometry differs')
    # Keep the official signed particle effect, avoid global mean-brightness subtraction.
    ambient=np.asarray(light['airlight_srgb'],np.float32);illum=cv2.GaussianBlur(source,(0,0),15)
    local_luma=illum@np.asarray([.2126,.7152,.0722],np.float32)
    gain=.35+.45*np.sqrt(np.clip(local_luma,0,1))
    chroma=ambient/max(float(ambient.max()),1e-6)
    signed=residual*gain[...,None]*chroma
    beta=.0006*(float(rate)/25.)**.75
    valid_depth=np.isfinite(depth)&(depth>0)
    transmission=np.exp(-beta*np.where(valid_depth,depth,0)).astype(np.float32)
    A=srgb_to_linear(ambient)
    mixed=(source+signed)*transmission[...,None]+A*(1-transmission[...,None])
    result=linear_to_uint8(mixed);result[~valid_depth]=rgb[~valid_depth]
    return result,dict(method='official_signed_streak_layer_scene_illumination_separate_atmosphere',
        rate_mm_h=float(rate),exposure_ms=3,mean_streak_gain=float(gain.mean()),particle_mask_fraction=float((coverage>0).mean()),
        atmosphere_beta=beta,mean_transmission=float(transmission.mean()),airlight_srgb=ambient.tolist(),
        upstream_global_mean_shift_removed=True,wet_road_layer_enabled=False,calibrated_weather=False,
        invalid_depth_policy='preserve_original_pixels',invalid_depth_fraction=float((~valid_depth).mean()))
