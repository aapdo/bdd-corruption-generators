# Automatic local source classification and multi-source tracking

For moving vehicle lamps selected alongside environmental lights, see
[vehicle-associated source selection and tracking](FLARE_VEHICLE_SOURCES.md).

For a CPU replay that broadens the same sources and measures vehicle contrast,
see [the exploratory vehicle-impact comparison](GLARE_VEHICLE_IMPACT.md).

This experimental BDD extension removes the manually checked source coordinates
used by the earlier anchored pilot. It uses a locally loaded CLIP ViT-B/32 for
candidate classification, local optical flow for individual tracking, and the
source-aligned scattering layer for rendering. No model inference API or image
upload is used. Public model-file downloads are a separate preparation step.

The pinned model is `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; `pytorch_model.bin` SHA256 is
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
Use the existing PyTorch2.3.1 / Transformers4.44.2 runtime with offline local
loading. Models, original images and flare patterns are external assets.

The pipeline is:

1. Detect compact bright/red peaks with local contrast and form local LK
   forward/backward tracks with current-frame photometric support.
2. For stable tracks, score three temporally spaced observations. Compare
   centered tight/context crops against positive and negative descriptions.
   Also erase the central peak in a third crop: the center must contribute to
   the positive score, so surrounding nighttime context alone is insufficient.
3. Keep tracks supported by at least two of the three representative decisions.
   Recheck candidate semantics in every supported frame, with disclosed
   hysteresis, confirmation and loss behavior.
4. Automatically choose at most three simultaneous sources per frame, with
   selection continuity. The union of track IDs across a clip can exceed three.
5. Preserve each existing source center and add the same nonnegative flare
   field at weak/middle/strong gains0.25/0.5/1.0. The relative DPT/BAM and bounded
   rendering choices from [the anchored extension](FLARE_ANCHORED.md) remain.

Positive concepts include streetlights, headlights, taillights, traffic signals,
the sun, strong specular glints and illuminated signs. Negative descriptions
include painted road markings, vehicle paint, flat white surfaces, sky gaps,
ordinary windows/signs, asphalt and unlit reflectors. Strong reflections are
eligible; a reflector's object category alone does not establish emission.
The precise prompt bank, thresholds and rejection reasons are in
`recipes/flare_auto_sources.py`.

Cosine similarities and their margins are relative model scores, not calibrated
probabilities or verified light-source identities. Small sources and reflections
can remain ambiguous. The six-clip test is an offline pilot with whole-clip
persistence, not a claim of causal online operation or generalization accuracy.
No-source frames are explicit effect-off cases. There is no manual/central
fallback and no substitution of failed generations with the original image.

Integration follows the previous anchored example. Replace reference admission
with automatic classification:

```python
from recipes.flare_sources import SourceTracker
from recipes.flare_auto_sources import LocalClipSources, automatic_tracks

tracker = SourceTracker()
for rgb in frames:
    tracker.step(rgb)
classifier = LocalClipSources(local_clip_asset_receipt, scheduler_gpu_request)
observations, summary, decisions = automatic_tracks(tracker, frames, classifier)
classifier.close()
# Pass observation['active_sources'] to AnchoredFlare.field as before.
```

`local_clip_asset_receipt` supplies `path`, pinned `revision`, and a `files`
mapping of filename to SHA256 and byte count. The constructor verifies them.
The GPU request binds one visible scheduler-owned GPU. Generation in managed
environments follows that site's scheduler; this library does not allocate GPUs.

```bash
python -m unittest discover -s tests -p test_flare_auto_sources.py
```

These CPU integration tests use mock scores to check multi-source limits,
semantic rejection and lack of fallback. Actual semantic accuracy requires
separate local-model and real-image inspection. Automatic decisions and rejected
candidates should remain available alongside original/severity comparisons.
