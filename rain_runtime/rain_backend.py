"""Isolated upstream namespace: this directory must contain no common.py."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import sys
import types

SOURCE = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((SOURCE/'rain_settings.json').read_text())
BASE = Path(os.environ['BDD_RAIN_RUNTIME'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, sort_keys=True, allow_nan=False)


def options(rate, speeds):
    from common.db import _settings_defaults
    if rate not in CONTRACT['levels']['Rain'] or not speeds or any(not 0 <= x <= 144 for x in speeds):
        raise ValueError('Unfrozen rate or invalid estimated speed')
    result = dict(_settings_defaults)
    result.update(cam_hz=5, cam_CCD_WH=[1280, 720], cam_CCD_pixsize=CONTRACT['rain_pixel_um'],
        cam_WH=[1280, 720], cam_focal=4., cam_gain=20, cam_f_number=6., cam_focus_plane=6., cam_exposure=3,
        cam_pos=CONTRACT['rain_camera_position'], cam_lookat=CONTRACT['rain_camera_lookat'], cam_up=[0., 1., 0.],
        sim_hz=1000, sim_mode='steps', sim_duration=(len(speeds)+1)/5,
        sim_steps=dict(cam_motion=speeds, rain_fallrate=[rate]*len(speeds)),
        sequences={}, depth_scale=1, render_scale=2)
    return result


def simulate(output, rate, inputs):
    from tools.simulation import WeatherSimulation
    root = output/'simulations'/str(rate); root.mkdir(parents=True)
    settings = options(rate, inputs['speeds_kmh'])
    class SeededSimulation(WeatherSimulation):
        def interact(self, wait_for, send_str):
            if wait_for == 'Set the seed for random generator':
                send_str = str(CONTRACT['rain_simulator_seed'])
            return super().interact(wait_for, send_str)
    simulator = SeededSimulation(0, str(root/'particles'), settings, dict(weather='rain', fallrate=rate),
        redo=False, deactivate_window_mode=True, bin_folder=str(BASE/'binary'))
    # Replace upstream's seed0 input with an explicit positive constant.
    simulator.run()
    paths = sorted((root/'particles').rglob('*camera0.xml'))
    if len(paths) != 1: raise ValueError('One closed camera stream required')
    from xml.etree.ElementTree import iterparse
    exposures = 0; particle_records = 0; left = right = 0
    for event, e in iterparse(paths[0], events=['end']):
        if 'ip1' in e.attrib:
            particle_records += 1
            xy = [float(x) for x in e.attrib['ip1'][1:-1].split(';')]
            if 0 <= xy[0] < 1280 and 0 <= xy[1] < 720:
                if xy[0] < 640: left += 1
                else: right += 1
        if e.tag == 'i': exposures += 1
        e.clear()
    if exposures < len(inputs['samples']) or particle_records == 0: raise ValueError('Incomplete particle stream')
    save(root/'SIMULATED.json', dict(path=str(paths[0]), sha256=sha(paths[0]), settings=settings,
        seed_input=CONTRACT['rain_simulator_seed'], seed_input_shared_across_rates=True, bitwise_repetition_verified=False,
        exposures=exposures, particle_records=particle_records, visible_left=left, visible_right=right,
        left_fraction=left/max(1, left+right), upstream_particle_simulator=True))


def render(output, rate, inputs):
    import cv2
    import numpy as np
    from common import db
    from common.generator import Generator
    cv2.setNumThreads(1)
    sim = json.loads((output/'simulations'/str(rate)/'SIMULATED.json').read_text())
    if sha(sim['path']) != sim['sha256']: raise ValueError('Particle stream changed')
    settings = options(rate, inputs['speeds_kmh']); video = inputs['video']; n = len(inputs['samples'])
    db.dbs['bdd_static_sweep'] = types.SimpleNamespace(settings=lambda: settings)
    args = types.SimpleNamespace(conflict_strategy='skip', rendering_strategy=None,
        output=str(output/'rendered'/str(rate)), dataset='bdd_static_sweep', dataset_root=str(output/'input'),
        images={video: str(output/'input'/video/'rgb')}, sequences=[video],
        depth={video: str(output/'input'/video/'depth')}, particles={video: [sim['path']]},
        weather=[dict(weather='rain', fallrate=rate)], texture=str(output/'rainstreakdb/env_light_database/size32'),
        norm_coeff=str(output/'rainstreakdb/env_light_database/txt/normalized_env_max.txt'), save_envmap=True,
        settings=settings, calib={video: None}, noise_scale=0., noise_std=0., opacity_attenuation=1.,
        frame_start=0, frame_end=n, frame_step=1, frames=None, verbose=False)
    Generator(args).run()
    folder = output/'rendered'/str(rate)/'bdd_static_sweep'/video/'rain'/(str(rate)+'mm')
    paths = sorted((folder/'rainy_image').glob('*.png'))
    if [p.name for p in paths] != [f'{i:04d}.png' for i in range(n)]: raise ValueError('Incomplete render')
    result = []
    for path, sample in zip(paths, inputs['samples']):
        image = cv2.imread(str(path))
        if image is None or image.shape != (360, 640, 3): raise ValueError('Rain shape differs')
        result.append(dict(image_id=sample['image_id'], path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    save(output/('RENDER_'+str(rate)+'.json'), dict(samples=result, simulation=sim,
        upstream_PBR_renderer=True, exposure_ms=3, full_video_temporal_qualified=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['simulate', 'render'])
    p.add_argument('output', type=Path); p.add_argument('rate', type=int); a = p.parse_args()
    inputs = json.loads((a.output/'RAIN_INPUTS.json').read_text())
    if a.stage == 'simulate': simulate(a.output, a.rate, inputs)
    else: render(a.output, a.rate, inputs)
