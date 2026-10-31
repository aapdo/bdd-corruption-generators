"""Regression cases from the source audit, including 100-frame synthetic sequences."""
import copy
import json
from pathlib import Path
from xml.etree.ElementTree import parse
import numpy as np
import pytest
from test_improved import fixture
from snow_scene import SceneSnowRenderer
from appearance import lighting_profile,lowlight_linear,fog_scene
from emitter import LampTracker
from glare import sources_for_frame,render_glare
from motion import TemporalExposureRenderer,target_blur,exposure_plan
from rain_adapter import correct_stream,validate_motion_for_release,compose_rain
from input_validation import aligned_depth,flow_inputs
from sample_io import write_sample,read_completed
from quality import advance_quality,read_released


def snow(haze,depth=None,sky=None):
    rgb,d,K,light=fixture();d=d if depth is None else depth
    renderer=SceneSnowRenderer(7301,d.shape,K,light,config=dict(particle_count=1,exposure_samples=1),scene_haze_beta_max=haze)
    renderer._initialize(np.eye(4),0.);renderer._base[:]=[0,0,2];renderer.threshold[:]=0
    return renderer.render(rgb,d,np.eye(4),1.,0.,camera_twist=np.zeros(6),sky_mask=sky)


def test_scene_haze_config_is_effective_and_legacy_is_rejected():
    a,b=snow(0),snow(.008)
    assert np.all(a['masks']['haze_transmission']==1)
    assert np.all(b['masks']['haze_transmission']<a['masks']['haze_transmission'])
    rgb,d,K,light=fixture()
    with pytest.raises(ValueError,match='legacy haze_beta'):
        SceneSnowRenderer(1,d.shape,K,light,config=dict(haze_beta=.1))


def test_verified_sky_receives_snow_and_unknown_object_does_not():
    rgb,d,K,light=fixture();d[:]=np.inf;sky=np.ones(d.shape,bool)
    a=snow(0,d,sky);b=snow(0,d,np.zeros_like(sky))
    assert a['masks']['particle_alpha'].max()>0 and b['masks']['particle_alpha'].max()==0
    assert np.array_equal(b['rgb'],rgb)


def test_snow_100_frames_deterministic_state_and_ids():
    rgb,d,K,light=fixture();config=dict(particle_count=40,exposure_samples=1)
    a=SceneSnowRenderer(9,d.shape,K,light,config=config);b=SceneSnowRenderer(9,d.shape,K,light,config=config)
    for i in range(100):
        x=a.render(rgb,d,np.eye(4),.5,i*.2,camera_twist=np.zeros(6))
        y=b.render(rgb,d,np.eye(4),.5,i*.2,camera_twist=np.zeros(6))
        assert np.array_equal(x['rgb'],y['rgb'])and a.state_fingerprint()==b.state_fingerprint()
        assert len(set(a._ids))==len(a._ids) and x['qa']['initialized_this_frame']==(i==0)


def emitter(color):
    rgb=np.full((64,96,3),10,np.uint8);rgb[20:25,44:49]=color
    return rgb,dict(source_type='headlight',bbox=[42,18,51,27],verified_by_user=True,track_id='light001')


@pytest.mark.parametrize('color,kind',[([255,255,255],'headlight'),([255,220,40],'streetlight'),([255,0,0],'tail_light')])
def test_typed_emitter_requires_identity_photo_and_tracking(color,kind):
    rgb,record=emitter(color);record['source_type']=kind;tracker=LampTracker()
    first=tracker.observe(rgb,0.,record);assert first['passed']is False
    record['tracking']=tracker.observe(rgb,.2,record);assert record['tracking']['passed']
    light=dict(scene_regime='night',controlled_sun_eligible=False)
    sources=sources_for_frame(rgb,light,record);assert len(sources)==1 and sources[0]['verified']
    record['verified_by_user']=False
    assert not sources_for_frame(rgb,light,record)[0]['verified']


def test_bbox_alone_and_red_paint_do_not_verify_light():
    rgb,record=emitter([255,0,0]);record.update(source_type='tail_light',verified_by_user=False,tracking={})
    sources=sources_for_frame(rgb,{},record);assert not sources[0]['verified']
    with pytest.raises(ValueError,match='source_type'):sources_for_frame(rgb,{},dict(accepted=True,bbox=record['bbox']))


def test_night_without_light_is_ineligible_even_if_sun_flag_is_wrong():
    rgb=np.zeros((64,96,3),np.uint8)
    light=dict(scene_regime='night',controlled_sun_eligible=True,airlight_srgb=[.5]*3,sky_light_xy=[0,0])
    sources=sources_for_frame(rgb,light,allow_controlled_sun=True);assert sources==[]
    image,q=render_glare(rgb,sources,1.);assert image is None and q['mode']=='INELIGIBLE'


def test_sun_fallback_is_opt_in_and_direction_fixed():
    rgb,d,K,light=fixture();light.update(scene_regime='day',controlled_sun_eligible=True,sun_eligibility_evidence='synthetic_daylight')
    assert sources_for_frame(rgb,light)==[]
    a=sources_for_frame(rgb,light,allow_controlled_sun=True);b=sources_for_frame(np.flip(rgb,axis=1).copy(),light,allow_controlled_sun=True)
    assert a[0]['xy']==b[0]['xy'] and not a[0]['verified']


def motion_record():
    return dict(xml_index=0,frame_id=100,timestamp_s=20.,twist=[0,0,0,0,0,0],
        qa=dict(validated_motion=True,fallback_assumption=None,twist_capped=False,static_support_pixels=1200,median_reprojection_pixels=.8))


def xml_file(tmp_path,n=1):
    p=tmp_path/'in.xml';p.write_text('<simulation>'+''.join('<i id="'+str(i)+'" t="'+str(i*200000000)+'" d="3000000" rs="0"/>'for i in range(n))+'</simulation>');return p


@pytest.mark.parametrize('change',[{'validated_motion':False},{'fallback_assumption':'stationary'},{'twist_capped':True},{'median_reprojection_pixels':2.}])
def test_release_motion_rejects_unqualified_records(change):
    m=motion_record();m['qa'].update(change)
    with pytest.raises(ValueError):validate_motion_for_release(m)


def test_required_xml_exposures_all_corrected_extra_unused_allowed(tmp_path):
    p=xml_file(tmp_path,2);m=motion_record();out=tmp_path/'out.xml'
    with pytest.raises(ValueError,match='Every required'):correct_stream(p,out,[m],[0])
    assert not out.exists()
    q=correct_stream(p,out,[m],[0],render_exposure_indices=[0])
    assert q['corrected_indices']==[0]and q['unused_XML_exposure_indices']==[1]
    assert q['source_to_XML_mapping'][0]['frame_id']==100 and not q['calibrated_metric_scale']


def test_motion_binding_and_timestamp_are_required(tmp_path):
    p=xml_file(tmp_path);m=motion_record();del m['xml_index']
    with pytest.raises(ValueError,match='binding'):correct_stream(p,tmp_path/'out.xml',[m],[0])


def test_rain_mask_metrics_and_misalignment(tmp_path):
    rgb,d,K,light=fixture();raw=rgb[...,::-1].astype(np.float32)/255;rain=raw.copy();rain[:,20:25]+=.05
    mask=np.zeros(d.shape,np.float32);mask[:,20:25]=1
    out,q=compose_rain(rgb,d,rain,raw,mask,light,25)
    assert q['residual_inside_mask']>0 and q['residual_outside_mask']==0 and q['particle_left_density']>0
    with pytest.raises(ValueError,match='align'):compose_rain(rgb,d,rain,raw,mask[:,:-1],light,25)


def test_temporal_fixed_exposure_100_frames_no_mode_switch_or_fresh_shake():
    rgb,d,K,light=fixture();renderer=TemporalExposureRenderer(K,20,'clip',shake_position_amplitude_px=2)
    valid=np.ones(d.shape,bool);p90=[]
    for i in range(100):
        flow=np.zeros((*d.shape,2),np.float32);flow[...,0]=0 if i<50 else 20
        out,q=renderer.render(rgb,flow,valid,i*.02)
        assert q['effective_exposure_ms']==20 and q['mode']=='controlled_camera_shake'and not q['automatic_shake_fallback']
        p90.append(q['actual_p90_pixels'])
    assert np.median(p90[50:])>np.median(p90[:50])


def test_actual_image_edge_and_object_sharpness_loss():
    rgb,d,K,light=fixture();flow=np.zeros((*d.shape,2),np.float32);flow[...,0]=100
    out,q=target_blur(rgb,flow,np.ones(d.shape,bool),8,'v',K,boxes=[[10,10,80,55]])
    assert q['sharpness']['edge_sharpness_loss']>.2
    assert q['sharpness']['object_sharpness'][0]['edge_sharpness_loss']>.2


def test_dark_fog_airlight_has_no_point48_floor_and_unknown_is_preserved():
    rgb,d,K,light=fixture();rgb[:]=8;profile=lighting_profile([(rgb,d)])
    assert max(profile['airlight_srgb'])<.1 and profile['scene_regime']=='night'and not profile['controlled_sun_eligible']
    d[0,0]=np.nan;y,q=fog_scene(rgb,d,.015,profile)
    assert np.array_equal(y[0,0],rgb[0,0])and q['valid_depth_fraction']<1


@pytest.mark.parametrize('model',['exposure_only','independent','correlated','fixed_pattern'])
def test_lowlight_noise_ablation_is_repeatable_without_color_grid(model):
    rgb,d,K,light=fixture();a,q=lowlight_linear(rgb,-1.5,'clip',1,noise_model=model);b,_=lowlight_linear(rgb,-1.5,'clip',1,noise_model=model)
    assert np.array_equal(a,b) and q['noise_model']==model and q['no_Bayer_or_demosaic']
    if model=='exposure_only':assert np.array_equal(a[...,0],a[...,1])
    if model=='fixed_pattern':
        c,p=lowlight_linear(rgb,-1.5,'clip',2,noise_model=model)
        assert q['fixed_pattern_camera_seed']==p['fixed_pattern_camera_seed']and q['noise_seed']!=p['noise_seed']


def test_bad_depth_and_flow_registration_rejected():
    with pytest.raises(ValueError,match='align'):aligned_depth(np.ones((100,100)),(720,1280,3))
    d=np.full((720,1280),20,np.float32);d[:,:640]=2
    aligned=aligned_depth(d,(720,1280,3));assert set(np.unique(aligned))=={2,20}
    with pytest.raises(ValueError,match='confidence'):flow_inputs(np.zeros((64,96,2),np.float32),np.ones((64,96),np.float32),(64,96))


def test_complete_manifest_required_and_corruption_detected(tmp_path):
    rgb,d,K,light=fixture();out=tmp_path/'sample.jpg';ready=write_sample(out,rgb,dict(type='Snow'))
    sample=read_completed(ready['path']);assert sample['generation_state']=='GENERATED'and not sample['training_release_ready']
    with pytest.raises(ValueError,match='not RELEASED'):read_released(ready['path'])
    out.write_bytes(out.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):read_completed(ready['path'])


def test_failed_pair_commit_is_not_visible_to_loader(tmp_path,monkeypatch):
    import sample_io
    rgb,d,K,light=fixture();out=tmp_path/'sample.jpg';original=sample_io.os.link;count=0
    def fail_metadata(source,target):
        nonlocal count
        count+=1
        if count==2:raise OSError('synthetic interrupted metadata commit')
        return original(source,target)
    monkeypatch.setattr(sample_io.os,'link',fail_metadata)
    with pytest.raises(OSError):write_sample(out,rgb,{})
    assert out.exists()and not out.with_suffix('.READY.json').exists()


def test_quality_pass_cannot_be_inferred_from_generation(tmp_path):
    rgb,d,K,light=fixture();out=tmp_path/'sample.jpg';ready=write_sample(out,rgb,{})
    with pytest.raises(ValueError,match='state transition'):advance_quality(ready['path'],'RELEASED',{},tmp_path/'release.json')
    with pytest.raises(ValueError,match='Explicit quality'):advance_quality(ready['path'],'STATIC_PASS',{},tmp_path/'static.json')


def test_forged_prior_stage_without_reports_cannot_release(tmp_path):
    rgb,d,K,light=fixture();ready=write_sample(tmp_path/'sample.jpg',rgb,{})
    prior=tmp_path/'fake_temporal.json';prior.write_text(json.dumps(dict(quality_state='TEMPORAL_PASS',sample_ready=ready,evidence=[])))
    with pytest.raises(ValueError,match='Prior quality stages'):advance_quality(prior,'RELEASED',{},tmp_path/'release.json')


def test_optional_wetness_accumulates_instead_of_instant_full_wet():
    from wetness import WetnessState
    a=WetnessState();assert a.advance(0,25)==0
    first=a.advance(.2,25);later=a.advance(2.,25)
    assert 0<first<later<1
    rgb,d,K,light=fixture()
    with pytest.raises(ValueError,match='verified road'):a.compose(rgb,np.ones(d.shape,bool),light['airlight_srgb'])


def test_cli_rejects_misaligned_depth_without_writing_sample(tmp_path):
    import subprocess,sys
    from PIL import Image
    image=tmp_path/'native.png';Image.fromarray(np.full((360,640,3),128,np.uint8)).save(image)
    depth=tmp_path/'depth.npy';np.save(depth,np.full((100,100),20,np.float32));out=tmp_path/'fog.jpg'
    result=subprocess.run([sys.executable,'-B',str(Path(__file__).parents[1]/'generate.py'),'--type','Fog','--image',str(image),'--depth',str(depth),'--level','.015','--output',str(out)],capture_output=True,text=True,timeout=15)
    assert result.returncode and 'align'in result.stderr and not out.exists()


def test_cli_valid_output_has_complete_hash_bound_manifest(tmp_path):
    import subprocess,sys
    from PIL import Image
    image=tmp_path/'native.png';Image.fromarray(np.full((360,640,3),128,np.uint8)).save(image);out=tmp_path/'dark.jpg'
    result=subprocess.run([sys.executable,'-B',str(Path(__file__).parents[1]/'generate.py'),'--type','LowLight','--image',str(image),'--level','-1.5','--output',str(out)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    sample=read_completed(out.with_suffix('.READY.json'));assert sample['noise_model']=='independent'and not sample['training_release_ready']


def test_rain_multiple_drops_keep_the_correct_exposure_depth(tmp_path):
    drop='<p pid="1" wp1="[0;0;-10]" wp2="[0;-.03;-10]" wd1=".001" wd2=".001" ip1="[640;360]" ip2="[640;357.3]" iw1=".09" iw2=".09"/>'
    source=tmp_path/'two.xml';source.write_text('<simulation><i id="25" rs="2">'+drop+drop+'</i></simulation>')
    out=tmp_path/'out.xml';q=correct_stream(source,out,[motion_record()],[0],depths=[np.full((360,640),30,np.float32)])
    assert len(parse(out).getroot()[0])==2 and q['particle_counts']==[2]


@pytest.mark.parametrize('key,value',[('frame_id',None),('timestamp_s',None),('timestamp_s','20'),('timestamp_s',True)])
def test_strict_rain_rejects_null_or_non_numeric_bindings(tmp_path,key,value):
    source=xml_file(tmp_path);m=motion_record();m[key]=value
    with pytest.raises(ValueError):correct_stream(source,tmp_path/'out.xml',[m],[0])
    assert not(tmp_path/'out.xml').exists()


def test_fog_does_not_promote_heuristic_sky_depth():
    rgb,d,K,light=fixture();d[:]=30
    out,q=fog_scene(rgb,d,.015,light)
    assert abs(q['mean_transmission']-np.exp(-.015*30))<1e-6 and q['verified_sky_fraction']==0


def test_glare_tracking_evidence_cannot_move_to_another_lamp():
    rgb,record=emitter([255,255,255]);rgb[20:25,14:19]=[255,255,255]
    tracker=LampTracker();tracker.observe(rgb,0.,record);record['tracking']=tracker.observe(rgb,.2,record)
    assert sources_for_frame(rgb,{},record)[0]['verified']
    record['bbox']=[12,18,21,27]
    assert not sources_for_frame(rgb,{},record)[0]['verified']


def test_rain_backend_checks_actual_render_binding():
    from rain_binding import validate_render_binding
    inputs=dict(samples=[dict(image_id=10,timestamp_s=2.)]);qa=dict(render_exposure_indices=[0],corrected_indices=[0],
        source_to_XML_mapping=[dict(xml_index=0,frame_id=10,timestamp_s=2.)],mode='NUMERICAL_MOTION_PROFILE_PASS',corrected_XML_sha256='abc')
    assert validate_render_binding(dict(SE3_correction=qa,sha256='abc'),inputs)['corrected_exposures']==1
    qa['source_to_XML_mapping'][0]['frame_id']=11
    with pytest.raises(ValueError,match='different source'):validate_render_binding(dict(SE3_correction=qa,sha256='abc'),inputs)
    with pytest.raises(ValueError,match='correction report'):validate_render_binding({},inputs)
