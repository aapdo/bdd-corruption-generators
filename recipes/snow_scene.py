"""Persistent snow with soft depth-dependent PSFs, scene light and separate haze."""
import math
import numpy as np
from snow import SnowRenderer,_confidence,_conservative_depth
from common import validate_inputs,rigid_pose,depth_valid,reproject_depth,se3_exp,linearize,encode


class SceneSnowRenderer(SnowRenderer):
    def __init__(self,seed,image_shape,K,light,config=None,scene_haze_beta_max=.004):
        if config is not None and 'haze_beta'in config:
            raise ValueError('SceneSnowRenderer uses scene_haze_beta_max; legacy haze_beta is not silently ignored')
        if not np.isfinite(scene_haze_beta_max)or not 0<=scene_haze_beta_max<=.1:
            raise ValueError('scene_haze_beta_max must be finite in[0,.1]')
        super().__init__(seed,image_shape,K,config=config);self.light=light;self.scene_haze_beta_max=float(scene_haze_beta_max)

    def render(self,rgb,depth,T_world_to_camera,strength,timestamp_s,camera_twist=None,sky_mask=None):
        import cv2
        validate_inputs(rgb,depth,self.K,T_world_to_camera,strength);T=rigid_pose(T_world_to_camera)
        c=self.config;h,w=self.shape
        sky=np.zeros(self.shape,bool)if sky_mask is None else np.asarray(sky_mask)
        if sky.shape!=self.shape or sky.dtype!=np.bool_:raise ValueError('Explicit verified sky mask must be aligned boolean')
        geometry=depth_valid(depth)&~sky;unknown=~(geometry|sky)
        geometry_depth=np.where(geometry,depth,0).astype(np.float32)
        timestamp,elapsed=self.clock.check(timestamp_s)
        twist=None if camera_twist is None else np.asarray(camera_twist,np.float64)
        if twist is not None and(twist.shape!=(6,)or not np.isfinite(twist).all()):raise ValueError('Invalid exposure twist')
        guard=self.guard+(0 if twist is None else float(np.linalg.norm(twist[:3]))*c['exposure_s']/2)
        if guard>=c['box_half_extent']:raise ValueError('Exposure guard exceeded')
        first=self._base is None
        if first:self._initialize(T,timestamp)
        births=self._advance(-T[:3,:3].T@T[:3,3],elapsed,guard)
        self._time=timestamp-self._origin_time;self.clock.last=timestamp
        active=np.flatnonzero(self.threshold<float(strength));confidence=_confidence(geometry|sky,c['boundary_dilation'])
        offsets=np.linspace(-c['exposure_s']/2,c['exposure_s']/2,c['exposure_samples'])
        particle=np.zeros((h,w),np.float64);visible=np.zeros(len(self._ids),bool);projected=visible.copy();covered=visible.copy();strata=np.zeros(3,int)
        for offset in offsets:
            pose=T if twist is None else se3_exp(twist*offset)@T
            sample_depth=_conservative_depth(reproject_depth(geometry_depth,self.K,T,pose));sample=np.zeros((h,w),np.float64)
            # Sky is rotation-only at infinity; finite reprojected foreground still occludes it.
            H=self.K@pose[:3,:3]@T[:3,:3].T@np.linalg.inv(self.K)
            sample_sky=cv2.warpPerspective(sky.astype(np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)>0
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
                known=sample_depth[region]>0;sky_visible=sample_sky[region]&~known
                usable=(weights>0)&(known|sky_visible)
                take=(weights>0)&((known&(z-radius<sample_depth[region]-.01))|sky_visible)
                covered[i]|=bool(usable.any());visible[i]|=bool(take.any())
                area=math.pi*np.linalg.det(self.K[:2,:2])*(radius/z)**2*self.ratio[i]
                alpha=np.clip(.65*c['flake_opacity']*area*weights/max(norm,1e-12),0,.7)*take
                sample[region]=1-(1-sample[region])*(1-alpha)
            particle+=sample/len(offsets)
        particle=np.minimum(particle,.8)*confidence;particle[unknown]=0
        linear=linearize(rgb);A=linearize(np.asarray(self.light['airlight_srgb'])*255)
        local=cv2.GaussianBlur(linear,(0,0),18)
        particle_light=np.clip(.35*A+.65*local+.025,0,.75)
        curve=float(np.interp(strength,[0,.1,.25,.5,.8,1],[0,0,.0375,.225,.625,1]))
        beta=self.scene_haze_beta_max*curve
        haze_depth=np.where(geometry,depth,np.where(sky,90.,0.))
        trans=np.exp(-beta*haze_depth);mixed=(linear*(1-particle[...,None])+particle_light*particle[...,None])*trans[...,None]+A*(1-trans[...,None])
        out=rgb.copy()if strength==0 else encode(mixed);out[unknown]=rgb[unknown]
        return dict(rgb=out,masks=dict(particle_alpha=particle.astype(np.float32),haze_transmission=trans.astype(np.float32)),qa=dict(
            method='scene_lit_persistent_3D_snow_soft_depth_PSF',active_flakes=len(active),visible_flakes=int(visible.sum()),
            projected_flakes=int(projected.sum()),occluded_flakes=int((covered&~visible).sum()),respawn_count=len(births),
            initialized_this_frame=first,particle_pixel_fraction=float((particle>.05).mean()),mean_particle_alpha=float(particle.mean()),
            mean_haze_opacity=float((1-trans).mean()),haze_beta=beta,scene_haze_beta_max=self.scene_haze_beta_max,mean_transmission=float(trans.mean()),
            verified_sky_fraction=float(sky.mean()),unknown_geometry_fraction=float(unknown.mean()),
            sky_mask_policy='explicit_verified_mask_only_no_heuristic_promotion',sky_haze_distance_is_assumed=True,
            particle_radiance_mean=particle_light.mean(axis=(0,1)).tolist(),airlight_srgb=self.light['airlight_srgb'],
            soft_PSF_cutoff_sigma=4.,camera_motion_approximation='constant_left_twist'if twist is not None else'center_only_missing_twist',
            exposure_samples=len(offsets),calibrated_weather_rate=False,continuous_temporal_qualified=False))
