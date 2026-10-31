"""Correct official camera-local drop endpoints with estimated full SE3 exposure."""
import math
import numbers
import hashlib
from pathlib import Path
from xml.etree.ElementTree import parse
import numpy as np
from common import se3_exp


def vector(text):return np.asarray([float(x)for x in text[1:-1].split(';')],np.float64)
def encoded(v):return '['+';'.join(format(float(x),'.10g')for x in v)+']'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def validate_motion_for_release(record):
    qa=record.get('qa',{});twist=np.asarray(record.get('twist'),np.float64)
    if twist.shape!=(6,)or not np.isfinite(twist).all():raise ValueError('Invalid camera twist')
    if not qa.get('validated_motion',False):raise ValueError('Unverified camera motion')
    if 'fallback_assumption'not in qa or qa['fallback_assumption']:raise ValueError('Fallback or unspecified motion assumption')
    if qa.get('twist_capped',True):raise ValueError('Clipped or unspecified camera motion')
    if qa.get('static_support_pixels',0)<80 or not np.isfinite(qa.get('median_reprojection_pixels',np.nan))or qa['median_reprojection_pixels']>1.5:
        raise ValueError('Insufficient motion support or reprojection quality')
    return dict(numerical_motion_profile_pass=True,calibrated_camera=bool(qa.get('calibrated_camera',False)),
        metric_depth_verified=bool(qa.get('metric_depth_verified',False)),training_release_ready=False)


def correct_stream(path,output,motions,base_speeds,exposure_s=.003,focal_pixels=900.,width=1280,height=720,depths=None,
                   render_exposure_indices=None,diagnostic=False):
    """Keep official rainfall particles; replace only the axial camera contribution.

    XML world coordinates are camera-local x-right,y-up,z-back. Images use
    bottom-left origin. This is a local constant-twist correction, not a full
    trajectory re-simulation or a claim of calibrated BDD camera motion.
    """
    if Path(output).exists():raise ValueError('Refuse to overwrite an existing particle stream')
    if not all(np.isfinite(x)and x>0 for x in [exposure_s,focal_pixels,width,height]):raise ValueError('Invalid exposure/camera settings')
    tree=parse(path);frames=[e for e in tree.getroot()if e.tag=='i'];counts=[];removed=0
    required=list(range(len(frames)))if render_exposure_indices is None else list(render_exposure_indices)
    if not required or any(type(i)is not int or not 0<=i<len(frames)for i in required)or len(set(required))!=len(required):
        raise ValueError('Invalid render exposure indices')
    if len(motions)!=len(required)or len(base_speeds)!=len(required)or(depths is not None and len(depths)!=len(required)):
        raise ValueError('Every required exposure needs motion, baseline speed and optional depth')
    bindings={};mapping=[];timestamps=[];frame_ids=[]
    for j,record in enumerate(motions):
        i=record.get('xml_index',required[j]if diagnostic else None)
        if type(i)is not int or i not in required or i in bindings:raise ValueError('Missing or duplicate XML exposure binding')
        twist=np.asarray(record.get('twist'),np.float64)
        if twist.shape!=(6,)or not np.isfinite(twist).all():raise ValueError('Invalid camera twist')
        speed=float(base_speeds[j])
        if not np.isfinite(speed)or not 0<=speed<=144:raise ValueError('Invalid baseline camera speed')
        if not diagnostic:validate_motion_for_release(record)
        if 'frame_id'not in record or 'timestamp_s'not in record:
            if not diagnostic:raise ValueError('Explicit source frame_id and timestamp_s required')
        timestamp=record.get('timestamp_s');frame_id=record.get('frame_id')
        valid_id=(isinstance(frame_id,str)and bool(frame_id.strip()))or(isinstance(frame_id,numbers.Integral)and not isinstance(frame_id,bool)and frame_id>=0)
        if not diagnostic and not valid_id:raise ValueError('Strict motion needs a non-null valid source frame_id')
        valid_time=isinstance(timestamp,numbers.Real)and not isinstance(timestamp,bool)and np.isfinite(timestamp)
        if (not diagnostic or timestamp is not None)and not valid_time:raise ValueError('Finite numeric source timestamp_s required')
        if 'd'in frames[i].attrib and not np.isclose(float(frames[i].attrib['d'])/1e9,exposure_s,rtol=0,atol=1e-8):raise ValueError('XML exposure duration differs')
        if depths is not None:
            d=np.asarray(depths[j])
            if d.ndim!=2 or not np.issubdtype(d.dtype,np.number)or np.any(np.isfinite(d)&(d<0)):raise ValueError('Invalid aligned depth')
        bindings[i]=(j,record);mapping.append(dict(xml_index=i,xml_id=frames[i].get('id'),xml_timestamp_ns=frames[i].get('t'),frame_id=frame_id,timestamp_s=timestamp))
    if set(bindings)!=set(required):raise ValueError('Incomplete XML correction')
    ordered=sorted(mapping,key=lambda x:x['xml_index'])
    known=[m for m in ordered if m['timestamp_s']is not None]
    if any(b['timestamp_s']<=a['timestamp_s']for a,b in zip(known,known[1:])):raise ValueError('Source timestamps must increase with XML exposure order')
    ids=[m['frame_id']for m in mapping if m['frame_id']is not None]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate source frame IDs')
    for index in sorted(required):
        binding_index,record=bindings[index];frame=frames[index]
        twist=np.asarray(record['twist'],np.float64);target=se3_exp(twist*exposure_s)
        old_speed=float(base_speeds[binding_index])/3.6;n=0
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
                d=depths[binding_index];dh,dw=d.shape
                for f in np.linspace(0,1,5):
                    q=p1+(corrected-p1)*f;u=focal_pixels*q[0]/q[2]+width/2;v=focal_pixels*q[1]/q[2]+height/2
                    x,y=math.floor(u*dw/width),math.floor(v*dh/height)
                    visible.append(bool(0<=x<dw and 0<=y<dh and np.isfinite(d[y,x])and d[y,x]>q[2]+.03))
                runs=[];start=None
                for sample_index,value in enumerate(visible+[False]):
                    if value and start is None:start=sample_index
                    if not value and start is not None:runs.append((start,sample_index-1));start=None
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
    # All required bindings were validated before any output becomes visible.
    with Path(output).open('xb')as stream:tree.write(stream,encoding='utf-8',xml_declaration=True)
    return dict(method='camera_local_official_particle_SE3_exposure_endpoint_correction',frames=len(counts),
        particle_counts=counts,removed_invalid=removed,exposure_ms=exposure_s*1000,full_upstream_6DoF_resimulation=False,
        estimated_camera_motion=True,calibrated_metric_scale=False,scene_depth_occlusion_samples=5 if depths is not None else 0,
        mode='DIAGNOSTIC'if diagnostic else'NUMERICAL_MOTION_PROFILE_PASS',
        source_XML_sha256=file_sha(path),corrected_XML_sha256=file_sha(output),
        render_exposure_indices=required,corrected_indices=sorted(bindings),source_to_XML_mapping=mapping,
        unused_XML_exposure_indices=sorted(set(range(len(frames)))-set(required)),training_release_ready=False)


def compose_rain(rgb,depth,raw_rain,raw_fogged,mask,light,rate):
    from baseline_render import srgb_to_linear,linear_to_uint8
    import cv2
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('RGB uint8 input required')
    if depth.shape!=rgb.shape[:2]or mask.shape!=depth.shape or raw_rain.shape!=rgb.shape or raw_fogged.shape!=rgb.shape:
        raise ValueError('Rain layers, mask and depth must align with RGB')
    if not all(np.isfinite(x).all()for x in [raw_rain,raw_fogged,mask])or not np.isfinite(rate)or rate<0:
        raise ValueError('Finite Rain layers/mask and nonnegative rate required')
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
    magnitude=np.mean(np.abs(residual),axis=-1);inside=coverage>1e-6;half=inside.shape[1]//2
    left=float(inside[:,:half].mean());right=float(inside[:,half:].mean());valid=valid_depth
    correlation=float(np.corrcoef(depth[valid],magnitude[valid])[0,1])if valid.sum()>2 and np.std(depth[valid])>1e-8 and np.std(magnitude[valid])>1e-8 else None
    return result,dict(method='official_signed_streak_layer_scene_illumination_separate_atmosphere',
        rate_mm_h=float(rate),exposure_ms=3,mean_streak_gain=float(gain.mean()),particle_mask_fraction=float((coverage>0).mean()),
        atmosphere_beta=beta,mean_transmission=float(transmission.mean()),airlight_srgb=ambient.tolist(),
        upstream_global_mean_shift_removed=True,wet_road_layer_enabled=False,calibrated_weather=False,
        invalid_depth_policy='preserve_original_pixels',invalid_depth_fraction=float((~valid_depth).mean()),
        residual_inside_mask=float(magnitude[inside].mean())if inside.any()else 0.,
        residual_outside_mask=float(magnitude[~inside].mean())if (~inside).any()else 0.,
        particle_left_density=left,particle_right_density=right,left_right_density_ratio=left/right if right>0 else None,
        no_right_particles=right==0,residual_depth_correlation=correlation,
        mask_raw_range=[float(mask.min()),float(mask.max())],atmosphere_calibrated_to_real_rain=False)
