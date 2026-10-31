"""Measured source bloom or explicitly controlled off-frame sun veiling."""
import numpy as np
import cv2
from baseline_render import srgb_to_linear,linear_to_uint8


def sources_for_frame(rgb,light,reviewed_anchor=None):
    h,w=rgb.shape[:2];linear=srgb_to_linear(rgb.astype(np.float32)/255);sources=[]
    if reviewed_anchor and reviewed_anchor.get('accepted'):
        box=reviewed_anchor['bbox'];x0,y0,x1,y1=np.rint(box).astype(int)
        x0,x1=max(0,x0),min(w,x1);y0,y1=max(0,y0),min(h,y1)
        region=np.zeros((h,w),np.uint8);region[y0:y1,x0:x1]=1
        R,G,B=np.moveaxis(linear,-1,0)
        core=(region>0)&(R>.3)&(R>1.6*G)&(R>1.6*B)
        if core.sum()>=3 and y1<h*.85:
            color=linear[core].mean(axis=0);color/=max(float(color.max()),1e-6)
            for name,mask in [('reviewed_vehicle_light',core)]:
                yy,xx=np.nonzero(mask);sources.append(dict(type=name,xy=[float(xx.mean()),float(yy.mean())],
                    color=color.tolist(),radiance=float(np.quantile(R[mask],.8)),radius=float(np.clip(np.sqrt(mask.sum()/np.pi),1,8)),
                    confidence='previous_high_confidence_visual_anchor_and_current_photometry',verified=True))
            # A symmetric counterpart is an RGB-supported candidate, never auto-verified.
            center=sources[0]['xy'];delta=np.abs(np.arange(w)-center[0]);band=(np.abs(np.arange(h)[:,None]-center[1])<12)
            candidate=band&(delta[None,:]>20)&(delta[None,:]<140)&(R>.5)&(R>1.8*G)&(R>1.8*B)
            n,labels,stats,centroids=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
            choices=[i for i in range(1,n)if 2<=stats[i,4]<=80]
            if choices:
                j=min(choices,key=lambda i:abs(centroids[i,1]-center[1]));mask=labels==j
                sources.append(dict(type='paired_vehicle_light_RGB_candidate',xy=centroids[j].tolist(),color=color.tolist(),
                    radiance=float(np.quantile(R[mask],.8)),radius=float(np.clip(np.sqrt(mask.sum()/np.pi),1,8)),verified=False,
                    confidence='paired_RGB_photometry_only_not_semantic_verification'))
    if not sources:
        # No red paint/hood point is promoted into an emitting source.
        hint=light['sky_light_xy'];left=hint[0]<w/2
        sources=[dict(type='controlled_off_frame_sun',xy=[-.12*w if left else 1.12*w,-.08*h],
            color=srgb_to_linear(np.asarray(light['airlight_srgb'],np.float32)).tolist(),color_space='linear_RGB',radiance=1.,radius=.6*max(h,w),verified=False,
            confidence='explicit_lighting_assumption_from_clip_sky_direction_not_an_observed_sun',assumed_sun_source=True)]
    return sources


def render_glare(rgb,sources,level):
    if not np.isfinite(level)or not 0<=level<=1.5:raise ValueError('Invalid glare gain')
    if level==0:return rgb.copy(),dict(gain=0.,sources=sources)
    h,w=rgb.shape[:2];yy,xx=np.mgrid[:h,:w];linear=srgb_to_linear(rgb.astype(np.float32)/255);added=np.zeros_like(linear)
    for s in sources:
        x,y=s['xy'];color=np.asarray(s['color'],np.float32)
        if s['type']=='controlled_off_frame_sun':
            # Broad continuous veil entering from outside the frame; no colored disk.
            distance=(xx-x)**2+(yy-y)**2;scale=.62*max(h,w)
            veil=np.exp(-distance/(2*scale*scale))
            rays=np.exp(-((yy-y)-.28*(xx-x))**2/(2*(.12*h)**2))*veil
            added+=(.14*veil+.012*rays)[...,None]*color*level
        else:
            sigma=float(np.clip(1.5*s['radius'],2,12));distance=(xx-x)**2+(yy-y)**2
            core=np.exp(-distance/(2*max(.7,s['radius']*.6)**2))
            wings=(1+distance/(sigma*sigma))**-2
            # Compact PSF wings and small scatter, bounded source-scaled radius.
            gain=.18*float(s['radiance'])*level
            added+=(.5*core+.5*wings)[...,None]*color*gain
    proposed=linear+added;out=np.maximum(rgb,linear_to_uint8(proposed))
    controlled=any(not s['verified']for s in sources)
    return out,dict(method='source_colored_compact_PSF_or_off_frame_sun_veil',gain=float(level),sources=sources,
        mode='CONTROLLED'if controlled else'DIAGNOSTIC',all_sources_verified=not controlled,
        mean_added_linear=float(added.mean()),max_added_linear=float(added.max()),
        clipped_frame_fraction=float(((proposed>1.01)&(added>1e-6)).any(axis=-1).mean()),
        no_subtractive_black_center=True,no_arbitrary_red_point_fallback=True,calibrated_lens=False)
