# Copyright 2026 The Google Research Authors.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Provided AS IS, without warranties or conditions of any kind.
# Modified: NumPy/OpenCV port; fixed comparison parameters; BDD solar alignment
# and temporal tracking. See docs/WU_DAYTIME.md and licenses/Apache-2.0.txt.
"""NumPy/OpenCV port of Wu ICCV2021 flare synthesis operators, with BDD alignment.

Derived equations: google-research/flare_removal/python/{synthesis,utils}.py,
commit83f1e7b59e762648c953de751d31f75720683622, Apache-2.0.
Official patterns: Wu et al., CC BY4.0. Original RGB geometry is preserved.
"""
import cv2
import numpy as np

EPS=1e-7
LEVELS=dict(weak=.5,middle=1.5,strong=4.5)


def read_pattern(path):
    array=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if array is None or array.ndim!=3 or array.shape[2]!=3 or array.dtype not in(np.uint8,np.uint16):raise ValueError('Expected official RGB PNG')
    return cv2.cvtColor(array,cv2.COLOR_BGR2RGB).astype(np.float32)/np.iinfo(array.dtype).max


def remove_background(im):
    low=im.min(axis=(0,1),keepdims=True);high=im.max(axis=(0,1),keepdims=True)
    return(im-low)*high/(high-low+EPS)


def white_balance(im):
    means=im.mean(axis=(0,1),keepdims=True)
    return float(means.max())*im/(means+EPS)


def blur(im,sigma=1.):
    x=np.arange(-10,11,dtype=np.float32);kernel=np.exp(-x*x/(2*sigma*sigma));kernel/=kernel.sum()
    return cv2.sepFilter2D(im,-1,kernel,kernel,borderType=cv2.BORDER_CONSTANT)


def matrix_for(origin,point,scale,angle=0.,reflection_centers=None):
    origin=np.asarray(origin,np.float64);point=np.asarray(point,np.float64)
    if reflection_centers is not None:
        template_center,scene_center=map(lambda x:np.asarray(x,np.float64),reflection_centers)
        a,b=origin-template_center,point-scene_center
        if np.linalg.norm(a)<8:raise ValueError('Captured flare source too close to optical center for two-anchor alignment')
        scale=float(np.linalg.norm(b)/np.linalg.norm(a))
        if not .2<=scale<=3.:raise ValueError('Captured pattern/source angle mismatch')
        angle=float(np.degrees(np.arctan2(b[1],b[0])-np.arctan2(a[1],a[0])))
        c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        linear=scale*np.array([[c,-s],[s,c]])
        matrix=np.c_[linear,scene_center-linear@template_center]
    else:
        matrix=cv2.getRotationMatrix2D(tuple(origin),angle,scale)
        matrix[:,2]+=point-origin
    error=float(np.linalg.norm(matrix@np.r_[origin,1]-point))
    if error>.001:raise ValueError('Pattern source alignment differs')
    return matrix.astype(np.float32),error


class WuFlare:
    def __init__(self,path,origin,kind='simulated',gamma=2.,rgb_gains=(1.,1.,1.),angle=0.,sigma=1.):
        self.rgb=read_pattern(path);self.gamma=float(gamma);self.origin=list(map(float,origin));self.kind=kind
        self.template=remove_background(self.rgb**self.gamma)
        self.rgb_gains=np.asarray(rgb_gains,np.float32);self.angle=float(angle);self.sigma=float(sigma)
        if not 1.8<=self.gamma<=2.2 or self.rgb_gains.shape!=(3,)or(self.rgb_gains<0).any():raise ValueError('Invalid fixed Wu parameters')

    def field(self,shape,source):
        h,w=shape[:2]
        if not source['active']:return np.zeros((h,w,3),np.float32),dict(effect_applied=False,reason='no_current_supported_sun',sources=[])
        point=source['xy'];scale=.95*h/self.template.shape[0]
        centers=([self.template.shape[1]/2,self.template.shape[0]/2],[w/2,h/2])if self.kind=='captured'else None
        matrix,error=matrix_for(self.origin,point,scale,self.angle,centers)
        warped=cv2.warpAffine(self.template,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        # Official order: affine -> clip -> white-balance -> RGB gain -> 21px blur.
        normalized=white_balance(np.clip(warped,0,1))
        field=blur(normalized*self.rgb_gains[None,None,:],self.sigma)
        if not np.isfinite(field).all()or(field<0).any():raise ValueError('Invalid flare field')
        return field,dict(effect_applied=True,source_xy=source['xy'],pattern_origin=self.origin,
            affine=matrix.tolist(),alignment_error_pixels=error,kind=self.kind,gamma=self.gamma,
            rgb_gains=self.rgb_gains.tolist(),blur_sigma=self.sigma,
            algorithm='Wu_official_patterns_DC_WB_linear_addition_NumPy_port',
            extension='native_sun_alignment_and_temporal_tracking; native_geometry_and_exposure_fixed')


def compose(rgb,field,strength,gamma=2.):
    if not np.isfinite(field).all()or(field<0).any()or not np.isfinite(strength)or strength<0:raise ValueError('Invalid additive flare')
    if strength==0 or not np.any(field):return rgb.copy()
    # Official RGB gain/blur/DC stage is clipped before the linear addition.
    flare=np.clip(field*strength,0,1)
    scene=(rgb.astype(np.float32)/255)**gamma
    combined=np.clip((scene+flare)**(1/gamma),0,1)
    # TensorFlow convert_image_dtype(float,uint8,saturate=True) uses255.5.
    return np.clip(combined*255.5,0,255).astype(np.uint8)


def find_peak(rgb,prediction,window=90):
    h,w=rgb.shape[:2];x,y=prediction
    x0,x1=max(0,int(x)-window),min(w,int(x)+window+1);y0,y1=max(0,int(y)-window),min(int(.65*h),int(y)+window+1)
    if x1<=x0 or y1<=y0:return None
    patch=rgb[y0:y1,x0:x1].astype(np.float32)/255
    lum=patch@np.array([.2126,.7152,.0722],np.float32)
    mask=((lum>.975)&(patch.min(axis=2)>.90)).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8);choices=[]
    for k in range(1,n):
        a,b,bw,bh,area=map(int,stats[k])
        if area<3 or max(bw,bh)/max(1,min(bw,bh))>5:continue
        point=centers[k]+[x0,y0];distance=float(np.linalg.norm(point-prediction))
        if distance>70:continue
        choices.append((distance-.03*np.sqrt(area),dict(xy=point.tolist(),area=area,active=True,current_peak_supported=True)))
    return min(choices,key=lambda x:x[0])[1]if choices else None


def track_sun(images,reference_index,seed):
    rows=[None]*len(images);reference=find_peak(images[reference_index],np.asarray(seed,np.float32))
    if reference is None:raise ValueError('Reviewed sun seed has no current saturated peak')
    rows[reference_index]=dict(reference,tracking='reviewed_reference_peak',forward_backward_error_pixels=None)
    for direction in [-1,1]:
        last=np.asarray(reference['xy'],np.float32);previous=cv2.cvtColor(images[reference_index],cv2.COLOR_RGB2GRAY)
        indices=range(reference_index+direction,len(images)if direction>0 else-1,direction)
        for i in indices:
            gray=cv2.cvtColor(images[i],cv2.COLOR_RGB2GRAY);predicted=last.copy();fb=None
            h,w=gray.shape;x,y=np.rint(last).astype(int);mask=np.zeros_like(gray)
            mask[max(0,y-100):min(h,y+101),max(0,x-100):min(w,x+101)]=255
            points=cv2.goodFeaturesToTrack(previous,30,.01,5,mask=mask)
            if points is not None and len(points)>=3:
                nxt,ok,_=cv2.calcOpticalFlowPyrLK(previous,gray,points,None)
                if nxt is not None:
                    back,valid,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None)
                    if back is not None:
                        error=np.linalg.norm(back-points,axis=2).ravel();keep=ok.ravel().astype(bool)&valid.ravel().astype(bool)&(error<2.)
                        if keep.sum()>=3:
                            shift=np.median((nxt-points).reshape(-1,2)[keep],axis=0)
                            if np.linalg.norm(shift)<100:predicted+=shift;fb=float(np.median(error[keep]))
            observed=find_peak(images[i],predicted)
            rows[i]=dict(observed or dict(active=False,xy=predicted.tolist(),current_peak_supported=False),
                tracking='local_flow_and_current_solar_peak'if observed else'no_supported_current_sun',forward_backward_error_pixels=fb)
            if observed:last=np.asarray(observed['xy'],np.float32)
            previous=gray
    return rows
