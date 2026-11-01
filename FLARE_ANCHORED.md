# Source-aligned Flare7K* BDD pilot extension

This is an experimental extension of the paper-inspired PGS implementation.
It places the scattering layer around existing image brightness peaks and tracks
each source locally. It is not the Flare7K* authors' official implementation.
No remote model or API is used. Pattern data and depth assets remain external.

`recipes.flare_sources.SourceTracker` detects compact bright/red peaks, checks
local contrast, predicts motion with local forward/backward LK, and associates
current peaks. Confirmation requires three consecutive observations. Lost or
unsupported sources receive no effect. Black images have no central fallback.
Bright pixels can also be road paint, signs, or vehicle paint: photometry is not
semantic source identification.

The four-video pilot therefore uses source positions visually checked in a
reference frame. `finalize(reference_anchors, reference_frame)` admits only tracks
connected to these positions. Empty anchors are explicit negative controls.
This is an offline experiment with a reference frame; it does not establish
fully automatic or causal online source selection.

`recipes.flare_anchored.AnchoredFlare` subtracts the matching annotated light
layer nonnegatively and protects the observed source center smoothly. The
source's location, radius, color and brightness control the transformed pattern.
No original scene pixels are removed. The same field supplies severity gains
0.25/0.5/1.0. Relative DPT distance and the previously declared BAM definition
are reused; the pilot additionally bounds BAM gain at 3 and pattern scale at
0.04–0.8. These are disclosed extension parameters, not calibrated radiometry.

Minimal integration, with local RGB frames, a paired external bank, and aligned
positive relative-distance maps supplied by local inference:

```python
from recipes.flare_sources import SourceTracker
from recipes.flare_anchored import AnchoredFlare
from recipes.flare7kstar import compose

tracker = SourceTracker()
for rgb in frames:
    tracker.step(rgb)
observations, track_summary = tracker.finalize(reference_anchors, reference_frame)
renderer = AnchoredFlare(video_id, bank, pattern_root)
for rgb, distance, observation in zip(frames, distances, observations):
    field, metadata = renderer.field(rgb, distance, observation['active_sources'])
    variants = {name: compose(rgb, field, strength) for name, strength in
                [('weak', 0.25), ('middle', 0.5), ('strong', 1.0)]}
```

Run the focused CPU checks with:

```bash
python -m unittest discover -s tests -p test_flare_sources.py
```

Tests cover independently moving source IDs, occlusion/reconfirmation, current
photometric support, checked-reference admission, no-source behavior, source
alignment/core preservation, and additive severity ordering. Real-video quality
and automatic source selection remain separate validation tasks.
