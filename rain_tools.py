"""Prepare private upstream layer exports, correct official XML, or compose layers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('export-patch');a.add_argument('--upstream',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('correct');a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--motion',type=Path,required=True);a.add_argument('--depths',type=Path,help='Optional N x H x W camera-z NPY')
    a.add_argument('--diagnostic',action='store_true',help='Explicitly allow unqualified motion; never marks data RELEASED')
    a=sub.add_parser('compose');a.add_argument('--image',type=Path,required=True);a.add_argument('--depth',type=Path,required=True)
    a.add_argument('--layers',type=Path,required=True);a.add_argument('--rate',type=float,required=True);a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refuse existing output')
    if a.mode=='export-patch':
        original=a.upstream/'common/generator.py';text=original.read_text()
        expected=json.loads((Path(__file__).parent/'EXTERNAL_ASSETS.json').read_text())['generator_py_sha256']
        if hashlib.sha256(original.read_bytes()).hexdigest()!=expected:raise ValueError('Upstream generator revision differs')
        first='rainy_bg = FOG.fog_rain_layer(bg, depth)';second='plt.imsave(out_rainy_path, np.clip(rainy_bg_copy[..., ::-1], 0, 1))'
        if text.count(first)!=1 or text.count(second)!=1:raise ValueError('Unexpected upstream source')
        text=text.replace(first,first+'\n                    improved_fogged = rainy_bg.copy()')
        text=text.replace(second,"np.savez_compressed(out_rainy_path + '.layers.npz', rainy=rainy_bg, fogged=improved_fogged, mask=rainy_mask)\n                    "+second)
        shutil.copytree(a.upstream,a.output);(a.output/'common/generator.py').write_text(text);return
    sys.path.insert(0,str(Path(__file__).parent/'recipes'))
    import numpy as np
    from rain_adapter import correct_stream,compose_rain
    if a.mode=='correct':
        data=json.loads(a.motion.read_text());depths=np.load(a.depths,allow_pickle=False)if a.depths else None
        q=correct_stream(a.input,a.output,data['motions'],data['base_speeds_kmh'],depths=depths,
            render_exposure_indices=data.get('render_exposure_indices'),diagnostic=a.diagnostic)
    else:
        import cv2
        from PIL import Image
        from appearance import lighting_profile
        from input_validation import aligned_depth
        from sample_io import write_sample
        raw=np.asarray(Image.open(a.image).convert('RGB'));rgb=cv2.resize(raw,(640,360),interpolation=cv2.INTER_CUBIC)
        d=aligned_depth(np.load(a.depth,allow_pickle=False),raw.shape)
        with np.load(a.layers,allow_pickle=False)as layers:out,q=compose_rain(rgb,d,layers['rainy'],layers['fogged'],layers['mask'],lighting_profile([(rgb,d)]),a.rate)
        q.update(training_release_ready=False,quality_state='GENERATED');write_sample(a.output,out,q);return
    with a.output.with_suffix('.json').open('x')as f:json.dump(q,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
