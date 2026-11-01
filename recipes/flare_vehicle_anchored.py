"""BDD extension: source-centered, source-core-preserving Flare7K scattering."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from recipes.flare7kstar import bam, compose


class AnchoredFlare:
    def __init__(self, video, bank, root):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(('anchored-v1:'+video).encode()).digest()[:8], 'big'))
        item = bank['patterns'][int(rng.integers(len(bank['patterns'])))]; self.pattern_id = item['id']
        images = {}
        for key in ('compound','light_source'):
            with Image.open(Path(root)/item[key]['file']) as image:
                images[key] = np.asarray(image.convert('RGB')).astype(np.float32)/255
        light = images['light_source']
        signal = light.max(axis=2)
        mask = signal > max(.1, .55*float(signal.max()))
        if not mask.any():
            raise ValueError('Template has no source annotation')
        yy,xx = np.nonzero(mask)
        self.origin = (float(xx.mean()),float(yy.mean()))
        self.template_radius = max(1., float(np.sqrt(mask.sum()/np.pi)))
        compound = images['compound']**2.2
        compound = np.maximum(compound-compound.min(axis=(0,1),keepdims=True),0)
        # Keep the original scene's source; add only the nonnegative surrounding flare.
        without_source = np.maximum(compound-light**2.2,0)
        soft = cv2.GaussianBlur(mask.astype(np.float32),(0,0),max(1.,self.template_radius*.2))
        self.template = without_source*(1-soft[:,:,None])
        self.angle = float(rng.uniform(-180,180))
        self.history = {}

    def field(self, rgb, depth, sources):
        h,w = rgb.shape[:2]
        image_yy,image_xx = np.mgrid[:h,:w].astype(np.float32)
        field = np.zeros(rgb.shape,np.float32)
        if not sources:
            return field, dict(pattern_id=self.pattern_id, sources=[], effect_applied=False,
                status='no_supported_source', source_origin='original_photometric_candidates')
        if depth is None or depth.shape != (h,w) or not np.isfinite(depth).all() or (depth<=0).any():
            raise ValueError('Positive aligned relative distance map required')
        distances, states = [], []
        for source in sources:
            x,y = source['xy']; radius = source['radius']; key=source['track_id']
            x0,x1=max(0,int(x-radius)),min(w,int(x+radius)+2)
            y0,y1=max(0,int(y-radius)),min(h,int(y+radius)+2)
            d = float(depth[y0:y1,x0:x1].mean())
            color = np.asarray(source['color'],np.float32)**2.2
            old = self.history.get(key)
            state = dict(radius=radius,depth=d,color=color)
            if old is not None:
                state = dict(radius=.7*old['radius']+.3*radius,depth=.8*old['depth']+.2*d,
                             color=.7*old['color']+.3*color)
            self.history[key]=state; states.append(state); distances.append(state['depth'])
        raw_factors = bam(distances,[s['xy'] for s in sources],w,h)
        records=[]
        for source,state,raw_factor in zip(sources,states,raw_factors):
            x,y=source['xy'];radius=state['radius']
            scale=min(float(np.clip(1.2*radius/self.template_radius,.04,.8))*2.25,1.6)
            matrix=cv2.getRotationMatrix2D(self.origin,self.angle,scale).astype(np.float32)
            matrix[:,2]+=np.asarray([x,y])-self.origin
            layer=cv2.warpAffine(self.template,matrix,(w,h),flags=cv2.INTER_LINEAR)
            halo_radius=float(np.clip(14*radius,100*h/720,240*h/720))
            halo=.12*(1+((image_xx-x)**2+(image_yy-y)**2)/halo_radius**2)**-1.6
            layer+=halo[:,:,None]
            # Smooth native source protection: no replacement, negative layer or hard ring.
            protect_radius=max(1.5,1.3*source['radius'])
            border=int(np.ceil(3*protect_radius))
            x0,x1=max(0,int(x)-border),min(w,int(x)+border+1)
            y0,y1=max(0,int(y)-border),min(h,int(y)+border+1)
            yy,xx=np.mgrid[y0:y1,x0:x1]
            distance=np.hypot(xx-x,yy-y)
            protect=1-np.exp(-.5*(distance/protect_radius)**4)
            layer[y0:y1,x0:x1]*=protect[:,:,None]
            gain=min(float(raw_factor),3.)
            color=state['color']/max(float(state['color'].max()),1e-6)
            radiance=float(state['color'].max())
            layer*=color[None,None,:]*gain*radiance*source['fade']
            field+=layer
            records.append(dict(track_id=source['track_id'],xy=source['xy'],
                radius=float(radius),relative_depth=state['depth'],raw_bam=float(raw_factor),
                used_bam=gain, bam_clipped=bool(raw_factor>3), scale=scale, halo_radius_pixels=halo_radius,
                observed_peak=source['peak'], observed_contrast=source['contrast'],
                fade=source['fade'], alignment_error_pixels=float(np.linalg.norm(matrix@np.r_[self.origin,1]-[x,y])),
                forward_backward_error_pixels=source['forward_backward_error_pixels'],
                source_identity_semantically_verified=False))
        return field,dict(pattern_id=self.pattern_id,sources=records,effect_applied=True,
            source_origin='original_photometric_candidates',
            core_preservation='nonnegative_compound_minus_source_plus_smooth_native_protection')

    def render(self,rgb,depth,sources,strength):
        field,metadata=self.field(rgb,depth,sources)
        return compose(rgb,field,strength),metadata
