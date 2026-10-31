"""Measured source bloom or explicitly controlled off-frame sun veiling."""
import numpy as np
import cv2
import hashlib
from baseline_render import srgb_to_linear,linear_to_uint8
from emitter import select_sources,LampTracker,detect_emitter


def render_from_eligibility(rgb,decision,level,allow_controlled=False):
    if decision.get('rgb_sha256')!=hashlib.sha256(rgb.tobytes()).hexdigest():raise ValueError('Eligibility belongs to different RGB')
    status=decision.get('status');sources=[]
    if status=='VERIFIED':
        for c in decision.get('qualified_sources',[]):
            if not all(c.get(k)is True for k in ['photometric_pass','geometry_pass','temporal_pass']):raise ValueError('Incomplete source eligibility evidence')
            sources.append(dict(type=c['type'],xy=c['xy'],radius=c['radius'],color=c['color'],radiance=c['radiance'],verified=True,
                source_id=c['source_id'],verification_scope=decision.get('verification_scope')))
        if not sources:raise ValueError('No qualified sources in VERIFIED decision')
    elif status=='CONTROLLED'and allow_controlled:
        from glare_eligibility import classify_glare_eligibility
        scene=decision.get('scene',{})
        if classify_glare_eligibility(scene,[])['status']!='CONTROLLED':raise ValueError('Unsupported controlled solar assumption')
        direction=scene['sun_direction_xy'];h,w=rgb.shape[:2]
        sources=[dict(type='controlled_off_frame_sun',xy=[direction[0]*w,direction[1]*h],color=[1.,.92,.78],radius=.6*max(h,w),
            radiance=1.,verified=False,assumed_sun_source=True,assumption_color_is_controlled=True)]
    else:raise ValueError('Glare eligibility must be VERIFIED or explicitly approved CONTROLLED')
    out,q=render_glare(rgb,sources,level)
    q.update(source_eligibility=status,source_eligibility_reason=decision['reason'],verification_scope=decision.get('verification_scope'),
        manual_source_audit_required=True,render_quality='UNASSESSED',task_relevance='UNASSESSED',training_release_ready=False)
    return out,q


def sources_for_frame(rgb,light,reviewed_anchor=None,allow_controlled_sun=False):
    return select_sources(rgb,light,reviewed_anchor,allow_controlled_sun=allow_controlled_sun)


def render_glare(rgb,sources,level,bloom_gain=1.,veiling_gain=1.):
    if not np.isfinite(level)or not 0<=level<=1.5:raise ValueError('Invalid glare gain')
    if not all(np.isfinite(g)and 0<=g<=3 for g in [bloom_gain,veiling_gain]):raise ValueError('Invalid optical component gain')
    if not sources:return None,dict(gain=float(level),sources=[],mode='INELIGIBLE',reason='no_supported_emitter_or_daylight_sun',training_release_ready=False)
    if level==0:return rgb.copy(),dict(gain=0.,sources=sources,mode='CONTROLLED'if any(not s['verified']for s in sources)else'SOURCE_VERIFIED')
    h,w=rgb.shape[:2];yy,xx=np.mgrid[:h,:w];linear=srgb_to_linear(rgb.astype(np.float32)/255);added=np.zeros_like(linear)
    for s in sources:
        x,y=s['xy'];color=np.asarray(s['color'],np.float32)
        if s['type']=='controlled_off_frame_sun':
            # Broad continuous veil entering from outside the frame; no colored disk.
            distance=(xx-x)**2+(yy-y)**2;scale=.62*max(h,w)
            veil=np.exp(-distance/(2*scale*scale))
            rays=np.exp(-((yy-y)-.28*(xx-x))**2/(2*(.12*h)**2))*veil
            added+=(.14*veil+.012*rays)[...,None]*color*level*veiling_gain
        else:
            sigma=float(np.clip(1.5*s['radius'],2,12));distance=(xx-x)**2+(yy-y)**2
            core=np.exp(-distance/(2*max(.7,s['radius']*.6)**2))
            wings=(1+distance/(sigma*sigma))**-2
            # Compact PSF wings and small scatter, bounded source-scaled radius.
            gain=.18*float(s['radiance'])*level
            bloom=(.5*core+.5*wings)*bloom_gain
            # Broad, weak scatter has its own gain; severity changes amplitude, not radius.
            veil=.04*np.exp(-distance/(2*(.18*max(h,w))**2))*veiling_gain
            added+=(bloom+veil)[...,None]*color*gain
    proposed=linear+added;out=np.maximum(rgb,linear_to_uint8(proposed))
    controlled=any(not s['verified']for s in sources)
    return out,dict(method='source_colored_compact_PSF_or_off_frame_sun_veil',gain=float(level),sources=sources,
        mode='CONTROLLED'if controlled else'SOURCE_VERIFIED',all_sources_verified=not controlled,
        bloom_gain=float(bloom_gain),veiling_gain=float(veiling_gain),optical_radii_fixed_across_severity=True,
        mean_added_linear=float(added.mean()),max_added_linear=float(added.max()),
        clipped_frame_fraction=float(((proposed>1.01)&(added>1e-6)).any(axis=-1).mean()),
        no_subtractive_black_center=True,no_arbitrary_red_point_fallback=True,calibrated_lens=False)
