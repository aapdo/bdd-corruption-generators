# Source-anchored glare: exploratory vehicle-impact comparison

This CPU-only extension reuses completed automatic source tracks and the recorded
relative-depth brightness factors. It broadens the source-centered pattern and
adds a smooth broad scattering envelope so nearby vehicles can lose contrast.
It does not use vehicle boxes to position lights or select per-frame intensity.
No model inference API or new local model inference is needed for this replay.

The added envelope is a Moffat profile with power1.6, linear peak0.12, and radius
equal to14 times the measured source radius, bounded to100–240 pixels at720p.
The existing source-core protection, tracked color, brightness and fade remain.
Pattern scale is multiplied by2.25 and bounded to1.6. The same spatial field is
rendered at gains0.25/0.7/1.5. These are exploratory comparison values, not final
severities or calibrated physical lens parameters. The envelope and enlargement
are project additions; they are not equations from the Flare7K* paper.

```python
from recipes.flare_vehicle_impact import ImpactFlare, LEVELS, audit_boxes
from recipes.flare7kstar import compose

effect = ImpactFlare(video_id, pattern_manifest, external_pattern_root)
for rgb, row, boxes in chronological_frames:
    field, metadata = effect.field(rgb, row)
    for level, strength in LEVELS.items():
        rendered = compose(rgb, field, strength)
        impact = audit_boxes(rgb, rendered, boxes)
```

`row` is the completed anchored-pilot frame record. It includes
`tracking.active_sources` and `metadata.pattern_id`/`metadata.sources`.
Each active source supplies its track ID, observed center, color, radius and
fade. Its matching metadata supplies the smoothed radius, pattern scale,
relative depth and `used_bam`. Replay every frame chronologically so color
smoothing remains identical to the producer; do not render a later frame alone.
Pattern selection and pixel centers are checked against the original metadata.
Frames without a supported source explicitly keep the effect off.

Vehicle audit boxes supply `id`, `category`, `xyxy`, and `original_occluded`.
The audit reports changed area at8/255, mean pixel change, normalized contrast
ratio and new saturation. These are visibility proxies, not detector AP, true
occlusion percentages, or annotation-validity decisions. Original annotations
and heavily affected examples are preserved for review. No test-set labels or
detector failures are used to choose this pilot's sources or gains.

```bash
python -m unittest discover -s tests -p test_flare_vehicle_impact.py
```

The three CPU tests check replay compatibility with the original renderer,
monotone gains, broader influence and effect-off behavior. The pilot uses four
night clips with100 frames each. Human quality review, downstream detector
performance and training release remain separate decisions.
