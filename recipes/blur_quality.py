"""Image sharpness diagnostics, including optional offline object boxes."""
import numpy as np
import cv2
from baseline_render import srgb_to_linear


def sharpness_loss(native,blurred,confidence=None,boxes=None):
    h,w=native.shape[:2]
    def gradient(rgb):
        y=srgb_to_linear(rgb.astype(np.float32)/255)@np.array([.2126,.7152,.0722],np.float32)
        return np.hypot(cv2.Sobel(y,cv2.CV_32F,1,0)/8,cv2.Sobel(y,cv2.CV_32F,0,1)/8)
    before,after=gradient(native),gradient(blurred);trusted=np.ones((h,w),bool)if confidence is None else confidence.copy()
    trusted[:2]=False;trusted[-2:]=False;trusted[:,:2]=False;trusted[:,-2:]=False
    values=before[trusted];threshold=max(1e-5,float(np.quantile(values,.7)))if values.size else 1e-5
    edges=trusted&(before>=threshold)
    def score(mask):
        take=mask&edges;n=int(take.sum())
        return dict(edge_pixels=n,edge_sharpness_loss=float(1-after[take].mean()/(before[take].mean()+1e-12))if n else None)
    global_score=score(np.ones((h,w),bool));objects=[]
    for j,box in enumerate([]if boxes is None else boxes):
        v=np.asarray(box,float)
        if v.shape!=(4,)or not np.isfinite(v).all()or not(v[2]>v[0]and v[3]>v[1]):raise ValueError('Offline boxes must be finite xyxy')
        x0,y0,x1,y1=np.rint(v).astype(int);region=np.zeros((h,w),bool)
        region[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=True
        objects.append(dict(object_index=j,bbox_xyxy=v.tolist(),**score(region)))
    return dict(**global_score,object_sharpness=objects,object_boxes_available=boxes is not None,
        object_metrics_are_offline_QA_only=True,edge_threshold_linear=threshold)
