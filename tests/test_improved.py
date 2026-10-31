"""CPU checks for the new radiometry, blur measurement and camera conventions."""
import copy
from xml.etree.ElementTree import parse
import numpy as np
import pytest
from appearance import lowlight_linear,lighting_profile,fog_scene
from motion import target_blur,rigid_flow
from glare import sources_for_frame,render_glare
from rain_adapter import correct_stream,vector,compose_rain
from snow_scene import SceneSnowRenderer
from render_ops import snow_levels
from common import se3_exp


def fixture():
    h,w=64,96;yy,xx=np.mgrid[:h,:w];v=(40+((xx//4+yy//4)%2)*120).astype(np.uint8)
    rgb=np.repeat(v[...,None],3,-1);d=np.full((h,w),30,np.float32)
    K=np.array([[90.,0,w/2],[0,90.,h/2],[0,0,1.]])
    return rgb,d,K,lighting_profile([(rgb,d)])


def test_lowlight_no_bayer_and_photometry():
    rgb,d,K,light=fixture();means=[];snr=[]
    for ev in [-.5,-.75,-1.,-1.5,-2.,-3.,-4.]:
        y,q=lowlight_linear(rgb,ev,'v',10);again,_=lowlight_linear(rgb,ev,'v',10)
        assert np.array_equal(y,again) and q['no_Bayer_or_demosaic']
        clean,c=lowlight_linear(rgb,ev,'v',10,noise=False)
        assert np.array_equal(clean[...,0],clean[...,1]) and np.array_equal(clean[...,1],clean[...,2])
        assert q['exposure_multiplier']==2**ev
        means.append(q['mean_linear_signal']);snr.append(q['estimated_signal_to_noise'])
    assert np.all(np.diff(means)<0) and np.all(np.diff(snr)<0)
    assert np.array_equal(lowlight_linear(rgb,0,'v',10)[0],rgb)


@pytest.mark.parametrize('flow_speed',[0.,2.,100.])
def test_blur_reaches_actual_pixel_targets(flow_speed):
    rgb,d,K,light=fixture();flow=np.zeros((*d.shape,2),np.float32);flow[...,0]=flow_speed;valid=np.ones(d.shape,bool)
    for target in [1.,2.,3.,4.,6.,8.,12.]:
        y,q=target_blur(rgb,flow,valid,target,'v',K)
        assert y.shape==rgb.shape and y.dtype==np.uint8
        assert abs(q['actual_p90_pixels']-target)<.03 and q['max_blur_pixels']<=20.001
        assert q['samples']%2==1 and q['samples']>=9
        if flow_speed==0:assert q['mode']=='controlled_camera_shake' and q['camera_shake_is_observed'] is False
        if q['mode']=='observed_motion':assert q['effective_exposure_ms']<=60.0001


def test_glare_hood_is_not_emitter_and_no_black_center():
    rgb,d,K,light=fixture();rgb[-5:,20:30]=[255,0,0]
    sources=sources_for_frame(rgb,light,None)
    assert sources[0]['type']=='controlled_off_frame_sun' and not sources[0]['verified']
    assert sources[0]['xy'][0]<0 or sources[0]['xy'][0]>rgb.shape[1]
    old=0.
    for g in [.1,.25,.4,.5,.75,1.,1.5]:
        y,q=render_glare(rgb,sources,g)
        assert np.all(y>=rgb) and q['sources']==sources and q['mode']=='CONTROLLED'
        assert q['mean_added_linear']>old;old=q['mean_added_linear']


def test_fog_fixed_lighting_monotonic():
    rgb,d,K,light=fixture();trans=[]
    for beta in [.00375,.005,.0075,.01,.015,.03,.06]:
        y,q=fog_scene(rgb,d,beta,light);trans.append(q['mean_transmission'])
        assert q['airlight_srgb']==light['airlight_srgb']
    assert np.all(np.diff(trans)<0)


def test_rain_coordinate_exposure_and_occlusion(tmp_path):
    xml='<simulation><i id="25" rs="1"><p pid="1" wp1="[0;0;-10]" wp2="[0;-.03;-9.97]" wd1=".001" wd2=".001" ip1="[640;360]" ip2="[640;357.29188]" iw1=".09" iw2=".09027"/></i></simulation>'
    original=tmp_path/'raw.xml';original.write_text(xml)
    # Forward camera speed10m/s was already present; replacing by the same twist is identity.
    same=tmp_path/'same.xml';motion=[dict(twist=np.array([0.,0.,-10.,0.,0.,0.]))]
    correct_stream(original,same,motion,[36.])
    p=list(parse(same).getroot()[0])[0]
    assert np.allclose(vector(p.attrib['wp2']),[0.,-.03,-9.97],atol=1e-7)
    lateral=tmp_path/'lateral.xml';motion[0]['twist']=np.array([1.,0.,0.,0.,0.,0.])
    correct_stream(original,lateral,motion,[36.])
    p=list(parse(lateral).getroot()[0])[0]
    assert np.allclose(vector(p.attrib['wp2']),[.003,-.03,-10],atol=1e-7)
    blocked=tmp_path/'blocked.xml'
    q=correct_stream(original,blocked,motion,[36.],depths=[np.ones((360,640),np.float32)])
    assert len(parse(blocked).getroot()[0])==0 and q['removed_invalid']==1
    assert not q['full_upstream_6DoF_resimulation']


def test_rain_radiometry_keeps_black_free_identity_at_zero_rate():
    rgb,d,K,light=fixture();raw=rgb[...,::-1].astype(np.float32)/255
    y,q=compose_rain(rgb,d,raw,raw,np.zeros(d.shape),light,0)
    assert np.array_equal(y,rgb) and q['upstream_global_mean_shift_removed']


def test_snow_shared_population_scene_light_and_occlusion():
    rgb,d,K,light=fixture();config=dict(particle_count=60,exposure_samples=3)
    renderer=SceneSnowRenderer(7301,d.shape,K,light,config=config)
    values=snow_levels(renderer,rgb,d,np.eye(4),0.,[.1,.25,.4,.5,.65,.8,1.],np.zeros(6))
    assert len({v['shared_state_sha256']for v in values})==1
    counts=[v['value']['qa']['active_flakes']for v in values]
    assert np.all(np.diff(counts)>=0)
    assert all(v['value']['qa']['soft_PSF_cutoff_sigma']==4 for v in values)
    assert all(not v['value']['qa']['continuous_temporal_qualified']for v in values)


def test_rigid_projection_uses_left_camera_twist():
    rgb,d,K,light=fixture();T=se3_exp(np.array([1.,0,0,0,0,0]));flow=rigid_flow(T,d,K)
    assert np.allclose(flow[...,0],3.,atol=1e-5) and np.allclose(flow[...,1],0,atol=1e-5)


def test_unconfident_flow_cannot_select_observed_motion():
    rgb,d,K,light=fixture();flow=np.full((*d.shape,2),100,np.float32)
    y,q=target_blur(rgb,flow,np.zeros(d.shape,bool),1.,'v',K)
    assert q['mode']=='controlled_camera_shake' and abs(q['actual_p90_pixels']-1)<.03
    assert q['observed_support_pixels']==0


def test_invalid_rain_depth_preserves_original():
    rgb,d,K,light=fixture();d[0,:3]=[np.nan,-1,np.inf];raw=rgb[...,::-1].astype(np.float32)/255
    y,q=compose_rain(rgb,d,raw,raw,np.zeros(d.shape),light,25)
    assert np.array_equal(y[0,:3],rgb[0,:3]) and q['invalid_depth_fraction']>0


def test_controlled_sun_color_is_linear():
    from baseline_render import srgb_to_linear
    rgb,d,K,light=fixture();light['airlight_srgb']=[.5,.5,.5]
    sources=sources_for_frame(rgb,light)
    assert sources[0]['color_space']=='linear_RGB'
    assert np.allclose(sources[0]['color'],srgb_to_linear(np.array([.5,.5,.5])))
