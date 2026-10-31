"""Preserve the scalar inverse-depth path for nonphysical interpolation values."""
import numpy as np
from .flare7kstar import relative_depth

def reconstruct_depths(images,inverses,single_inverse):
    if len(images)!=len(inverses):raise ValueError('Image/depth count differs')
    result=[];refined=0
    for image,inverse in zip(images,inverses):
        if not np.isfinite(inverse).all():raise ValueError('Non-finite model inverse depth')
        if len(images)>1 and np.min(inverse)<=0:
            inverse=single_inverse(image);refined+=1
        result.append(relative_depth(inverse))
    return result,refined
