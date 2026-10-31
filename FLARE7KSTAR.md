# Flare7K* PGS for full BDD MOT videos

This is a paper reimplementation of the PGS synthesis component from
[Harmonizing Light and Darkness (2024)](https://arxiv.org/abs/2404.00313),
with explicit video additions. At implementation time the
[author repository](https://github.com/qulishen/Harmonizing-Light-and-Darkness)
contained a README and license, without synthesis source. This does not reproduce
their AFM or removal-model training, or assert reproduction of their reported scores.

The 2026-11-01 full-data run targets all 2,000 BDD100K MOT videos / 397,688 native
1280x720, 5fps frames, generating three JPEG95 views per frame (1,193,064 images).
It generates first and records triage afterwards. The earlier source eligibility
classifier remains available as a separate recipe and does not prefilter this run.
No OpenAI or other model API is called. DPT runs locally without training.

## Method and explicit choices

- Use all 5,000 official Flare7K scattering `Compound_Flare` / `Light_Source` pairs.
  Pick one pair per video and make 1–3 random affine copies with matching masks.
- Local Intel `dpt-hybrid-midas`, revision `11eaf7a1cf4bd70740697dbc216f98980c0aeb03`.
  `pytorch_model.bin` SHA256:
  `b6c4d44f9d96ca3fa76dd3bbb153989a60b4ad5526559f3c598562a368d687ec`.
  Invert its relative inverse depth, flooring at the positive 1st percentile.
  The paper does not identify an exact DPT checkpoint. Distances are not metres.
- Use the rightmost published Eq7, `(mean(d)/d_i)^2*cos(theta_i)`, FOV proxy 60°.
  The paper also writes `E_i/mean(E)` as equal, which is generally not equivalent.
  This implementation fixes the former definition, including off-axis falloff
  for a single source. No calibrated lens/radiance claim is made.
- Gamma 2.2 approximation: linearize RGB, add the transformed pattern, clip and
  encode. Weak/middle/strong radiance multipliers are 0.25 / 0.5 / 1.0, using the
  same geometry and pattern. They are candidates, not final training severities.
- Video additions: stable per-video seed, background KLT + partial-affine
  transport, source-depth and BAM EMA(alpha=0.2). These are not 3D light tracking.
  Failed motion keeps the last cumulative transform and is explicitly recorded.
- This creates new synthetic light sources. It does not claim to have detected
  original headlights, taillights or reflectors. The paper targets nighttime;
  application to daytime BDD frames has a domain gap requiring visual review.

Keep all generated candidates. Record weak effect, saturation, motion failures,
field changes and daytime brightness as triage indicators. No automatic metric
is a human visual pass or approval for training. Never replace a failed generated
image with Native. Preserve split membership, error records and partial outputs.

## Offline use

Use Python 3.10 with NumPy, Pillow, OpenCV, PyTorch 2.3.1, torchvision 0.18.1 and
Transformers 4.44.2. Supply Intel's model files locally; this script does not
download weights or call an inference service. The model files are available from
[Intel](https://huggingface.co/Intel/dpt-hybrid-midas).

Prepare the external bank with `MANIFEST.json` containing `patterns` (5,000
entries). Each has `id`, `compound` and `light_source`; each image reference has
`file`, `bytes`, `sha256`. Paths are relative to the bank root. Bank data remains
under the [original Flare7K license](https://github.com/ykdai/Flare7K).
The S-Lab research dataset terms are separate from this repository's code license.

Create a manifest with a `video_id` and time-ordered `frames: [{"path": ...}]`.

```bash
python generate_flare7kstar.py --manifest clip.json --patterns /external/flare-bank \
  --model /external/dpt-hybrid-midas --output outputs/clip --device cuda
python -m unittest discover -s tests -p test_flare7kstar.py
```

In managed environments GPU allocation belongs to the site's scheduler. The
project run uses source/config hashes, one-GPU qualification before bulk execution,
two sequential archive lanes, single-execution locks and per-attempt process/log
records. There are no automatic model retries or periodic agent monitors.
The private project wrapper additionally writes per-video TARs, member offsets,
source/variant SHA256s and five Native/weak/middle/strong comparison timestamps.

CPU tests cover inverse-depth interpretation, BAM inverse-square and cosine
factors, zero-strength identity, severity monotonicity, deterministic replay and
conversion of flow translation to native coordinates. These are implementation
checks; they do not establish that every generated video looks realistic.
