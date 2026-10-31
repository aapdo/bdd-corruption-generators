"""Optional controlled road-film approximation, disabled in the corruption sweep."""
import numpy as np
from baseline_render import srgb_to_linear,linear_to_uint8


class WetnessState:
    def __init__(self,saturation_mm=.2,evaporation_half_life_s=3600.):
        if not all(np.isfinite(x)and x>0 for x in [saturation_mm,evaporation_half_life_s]):raise ValueError('Positive wetness time scales required')
        self.saturation_mm=float(saturation_mm);self.half_life=float(evaporation_half_life_s);self.state=0.;self.last=None

    def advance(self,timestamp_s,rate_mm_h):
        if not np.isfinite(timestamp_s)or not np.isfinite(rate_mm_h)or rate_mm_h<0:raise ValueError('Finite time and nonnegative rate required')
        if self.last is not None:
            dt=timestamp_s-self.last
            if dt<=0:raise ValueError('Wetness timestamps must increase')
            dry=self.state*np.exp(-np.log(2)*dt/self.half_life)
            rain_depth=rate_mm_h*dt/3600
            self.state=float(1-(1-dry)*np.exp(-rain_depth/self.saturation_mm))
        self.last=float(timestamp_s)
        return self.state

    def compose(self,rgb,road_mask,airlight_srgb,road_mask_verified=False):
        if not road_mask_verified:raise ValueError('Optional wetness requires an explicitly verified road mask')
        if road_mask.shape!=rgb.shape[:2]or road_mask.dtype!=np.bool_:raise ValueError('Aligned boolean road mask required')
        source=srgb_to_linear(rgb.astype(np.float32)/255);A=srgb_to_linear(np.asarray(airlight_srgb,np.float32))
        film=self.state*road_mask[...,None]
        result=source*(1-.08*film)+.025*film*A
        return linear_to_uint8(result),dict(mode='CONTROLLED',wetness_state=self.state,enabled=True,
            method='optional_road_film_appearance_approximation',surface_optics_calibrated=False,
            default_sweep_enabled=False,training_release_ready=False)
