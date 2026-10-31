"""Persistent snow with soft depth-dependent PSFs, scene light and separate haze."""
import math
import numpy as np
from snow import SnowRenderer,_confidence,_conservative_depth
from common import validate_inputs,rigid_pose,depth_valid,reproject_depth,se3_exp,linearize,encode


class SceneSnowRenderer(SnowRenderer):
    def __init__(self,seed,image_shape,K,light,config=None):
        super().__init__(seed,image_shape,K,config=config);self.light=light

    def render(self,rgb,depth,T_world_to_camera,strength,timestamp_s,camera_twist=None):
        import cv2
        validate_inputs(rgb,depth,self.K,T_world_to_camera,strength);T=rigid_pose(T_world_to_camera)
        timestamp,elapsed=self.clock.check(timestamp_s);c=self.config;h,w=self.shape
        twist=None if camera_twist is None else np.asarray(camera_twist,np.float64)
        if twist is not None and(twist.shape!=(6,)or not np.isfinite(twist).all()):raise ValueError('Invalid exposure twist')
        guard=self.guard+(0 if twist is None else float(np.linalg.norm(twist[:3]))*c['exposure_s']/2)
        if guard>=c['box_half_extent']:raise ValueError('Exposure guard exceeded')
        first=self._base is None
        if first:self._initialize(T,timestamp)
        births=self._advance(-T[:3,:3].T@T[:3,3],elapsed,guard)
        self._time=timestamp-self._origin_time;self.clock.last=timestamp
        active=np.flatnonzero(self.threshold<float(strength));valid=depth_valid(depth);confidence=_confidence(valid,c['boundary_dilation'])
        offsets=np.linspace(-c['exposure_s']/2,c['exposure_s']/2,c['exposure_samples'])
        particle=np.zeros((h,w),np.float64);visible=np.zeros(len(self._ids),bool);projected=visible.copy();covered=visible.copy();strata=np.zeros(3,int)
        for offset in offsets:
            pose=T if twist is None else se3_exp(twist*offset)@T
            sample_depth=_conservative_depth(reproject_depth(depth,self.K,T,pose));sample=np.zeros((h,w),np.float64)
            points=self._base[active]+self.velocity[active]*offset+self._flutter(self._time+offset)[active]
            cam=points@pose[:3,:3].T+pose[:3,3]
            good=(cam[:,2]>=c['near'])&(cam[:,2]<=c['far'])
            for j in np.flatnonzero(good):
                i=active[j];z=float(cam[j,2]);radius=self.radii[i];q=self.K@cam[j];u,v=q[:2]/z
                angle=self.angle[i]+.15*math.sin(self.phase[i,0]+self.frequency[i]*(self._time+offset))
                co,si=math.cos(angle),math.sin(angle);rotation=np.array([[co,-si],[si,co]])
                footprint=self.K[:2,:2]@rotation@np.diag([radius,radius*self.ratio[i]])/(math.sqrt(2)*z)
                # Near flakes defocus; distant flakes remain soft subpixel scatter.
                defocus=1.5*max(0.,1-z/2.)**2
                covariance=footprint@footprint.T+np.eye(2)*(.42+defocus**2)
                inverse=np.linalg.inv(covariance);bx,by=4*np.sqrt(np.diag(covariance))
                xa,xb=int(math.ceil(u-bx)),int(math.floor(u+bx));ya,yb=int(math.ceil(v-by)),int(math.floor(v+by))
                x0,x1,y0,y1=max(0,xa),min(w-1,xb),max(0,ya),min(h-1,yb)
                if x0>x1 or y0>y1:continue
                projected[i]=True
                yy,xx=np.mgrid[ya:yb+1,xa:xb+1];delta=np.stack([xx-u,yy-v],-1)
                r2=np.einsum('...i,ij,...j->...',delta,inverse,delta);weights=np.exp(-.5*r2)*(r2<16)
                norm=float(weights.sum());weights=weights[y0-ya:y1-ya+1,x0-xa:x1-xa+1];region=np.s_[y0:y1+1,x0:x1+1]
                usable=(weights>0)&valid[region]&(sample_depth[region]>0);take=usable&(z-radius<sample_depth[region]-.01)
                covered[i]|=bool(usable.any());visible[i]|=bool(take.any())
                area=math.pi*np.linalg.det(self.K[:2,:2])*(radius/z)**2*self.ratio[i]
                alpha=np.clip(.65*c['flake_opacity']*area*weights/max(norm,1e-12),0,.7)*take
                sample[region]=1-(1-sample[region])*(1-alpha)
            particle+=sample/len(offsets)
        particle=np.minimum(particle,.8)*confidence
        linear=linearize(rgb);A=linearize(np.asarray(self.light['airlight_srgb'])*255)
        local=cv2.GaussianBlur(linear,(0,0),18)
        particle_light=np.clip(.35*A+.65*local+.025,0,.75)
        beta=float(np.interp(strength,[0,.1,.25,.5,.8,1],[0,0,.00015,.0009,.0025,.004]))
        trans=np.exp(-beta*np.where(valid,depth,0));mixed=(linear*(1-particle[...,None])+particle_light*particle[...,None])*trans[...,None]+A*(1-trans[...,None])
        out=rgb.copy()if strength==0 else encode(mixed);out[~valid]=rgb[~valid]
        return dict(rgb=out,masks=dict(particle_alpha=particle.astype(np.float32),haze_transmission=trans.astype(np.float32)),qa=dict(
            method='scene_lit_persistent_3D_snow_soft_depth_PSF',active_flakes=len(active),visible_flakes=int(visible.sum()),
            projected_flakes=int(projected.sum()),occluded_flakes=int((covered&~visible).sum()),respawn_count=len(births),
            initialized_this_frame=first,particle_pixel_fraction=float((particle>.05).mean()),mean_particle_alpha=float(particle.mean()),
            mean_haze_opacity=float((1-trans).mean()),haze_beta=beta,mean_transmission=float(trans.mean()),
            particle_radiance_mean=particle_light.mean(axis=(0,1)).tolist(),airlight_srgb=self.light['airlight_srgb'],
            soft_PSF_cutoff_sigma=4.,camera_motion_approximation='constant_left_twist'if twist is not None else'center_only_missing_twist',
            exposure_samples=len(offsets),calibrated_weather_rate=False,continuous_temporal_qualified=False))
