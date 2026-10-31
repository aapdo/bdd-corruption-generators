"""Alignment checks and depth resampling that never invents boundary distances."""
import numpy as np
import cv2


def rgb_image(rgb):
    if not isinstance(rgb,np.ndarray)or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('RGB must be uint8 HxWx3')
    return rgb


def aligned_depth(depth,raw_shape,output_shape=(360,640)):
    d=np.asarray(depth)
    if d.ndim!=2 or d.shape not in {tuple(raw_shape[:2]),tuple(output_shape)}:
        raise ValueError('Depth must align with input or output RGB')
    if not np.issubdtype(d.dtype,np.number)or np.iscomplexobj(d)or np.any(np.isfinite(d)&(d<0)):
        raise ValueError('Nonnegative real camera-z depth required; nonfinite/zero means unknown')
    # Nearest avoids mixtures of foreground and background depth. This does not prove registration.
    return cv2.resize(d.astype(np.float32),(output_shape[1],output_shape[0]),interpolation=cv2.INTER_NEAREST)


def aligned_mask(mask,raw_shape,output_shape=(360,640)):
    m=np.asarray(mask)
    if m.dtype!=np.bool_ or m.ndim!=2 or m.shape not in {tuple(raw_shape[:2]),tuple(output_shape)}:
        raise ValueError('Verified sky mask must be aligned boolean HxW')
    return cv2.resize(m.astype(np.uint8),(output_shape[1],output_shape[0]),interpolation=cv2.INTER_NEAREST)>0


def flow_inputs(flow,confidence,shape):
    if not isinstance(flow,np.ndarray)or flow.shape!=(*shape,2)or not np.issubdtype(flow.dtype,np.floating)or not np.isfinite(flow).all():
        raise ValueError('Flow must be finite floating HxWx2 in output pixels')
    if not isinstance(confidence,np.ndarray)or confidence.shape!=shape or confidence.dtype!=np.bool_:
        raise ValueError('Flow confidence must be aligned boolean HxW')
    return flow.astype(np.float32),confidence
