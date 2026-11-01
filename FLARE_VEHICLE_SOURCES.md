# Vehicle-associated lamps and environmental lights

The earlier automatic pilot selected three sources by a CLIP margin and track
continuity. Large persistent overhead lights dominated, while moving small
vehicle lamps often failed the ten-observation gate. This revision locates
vehicles with a local pretrained detector, tracks lamps relative to the moving
vehicle, and gives vehicle lamps and environmental lights separate source slots.

All inference is local: torchvision Faster R-CNN ResNet50 FPN COCO_V1, the pinned
local CLIP model, then DPT. No model inference API or image upload is used.
The detector uses car/bus/truck detections at score0.4 and batch2 at native720p.
Its official public weight is
`https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`.
Verify the official hash prefix, record the full SHA256 and use a strict CPU
weight load before managed GPU execution. Models and patterns are external assets.

1. Associate vehicle detections using within-vehicle optical flow and predicted
   box overlap. Original annotation boxes are not used for generation.
2. Find compact white/red peaks within each detected vehicle. Check local
   brightness, area, plausible lamp position and optional opposite-side pairing.
   Plate-center, roof and wheel regions are excluded by the disclosed rules.
3. Verify whole detected-vehicle crops with the same local CLIP. Compare vehicle
   descriptions to fences, railings, shopfronts, buildings and empty roadside
   context. At least two of three temporal crops must pass cosine0.20 and
   positive-minus-negative margin0.005. This reduces detector false vehicles.
4. Associate peaks in normalized vehicle coordinates. Missing current peaks
   immediately stop the effect; reappearance requires confirmation. A track
   needs three observations and two consecutive current observations.
5. Score three temporal representatives with the local CLIP classifier.
   At least two need photometric/geometry/semantic support. These permissive
   pilot thresholds are explicit in `recipes/flare_vehicle_sources.py`.
6. Keep up to four vehicle lamps, balanced across vehicles, plus two supported
   environmental sources. Rank within groups by observed contrast, area and
   peak, with modest continuity. CLIP margins control admission, not brightness.

```python
from recipes.flare_vehicle_sources import LocalVehicles, automatic_vehicle_sources
from recipes.flare_vehicle_context import VehicleAwareClip
from recipes.flare_vehicle_anchored import AnchoredFlare
from recipes.flare_vehicle_impact import LEVELS
from recipes.flare7kstar import compose

# detector_ready supplies local path and full weight sha256.
detector = LocalVehicles(detector_ready, scheduler_gpu_request)
detected = []
for start in range(0, len(frames), 2):
    detected.extend(detector.predict(frames[start:start+2]))
detector.close()
local_clip_classifier = VehicleAwareClip(local_clip_asset_receipt, scheduler_gpu_request)
rows, summary, decisions = automatic_vehicle_sources(
    frames, detected, previous_frame_records, local_clip_classifier)
local_clip_classifier.close()
# Load local DPT next and render source-aligned frames as in FLARE_ANCHORED.md.
```

`previous_frame_records` supplies the original environmental candidate evidence
from the preceding automatic pilot. Each item includes `tracking.sources` and
`tracking.semantic_candidates`. Candidates inside current detected vehicles and
the lower hood region are removed from the environmental pool. This revision
does not invent missing environmental semantics or manually insert light points.

Rendering retains the exploratory expanded Flare7K surrounding layer and broad
Moffat scattering envelope described in [the impact comparison](GLARE_VEHICLE_IMPACT.md),
with gains0.25/0.7/1.5. It is a BDD extension, not official Flare7K* author code
or a calibrated physical lens simulation. The same spatial field is used at
all three gains. Preserve original/source overlays and every rejected track.

```bash
python -m unittest discover -s tests -p test_flare_vehicle_sources.py
```

Seven CPU tests check side-lamp geometry/plate rejection, moving vehicle-relative
identity, disappearance/reconfirmation, separate source quotas and complete
decision-record JSON serialization, fence-context rejection and retention of
photometrically supported lamps despite coarse text-type disagreement.
Coordinates use standard Python integers. White lamp candidates use normalized
height0.28–0.72 to reduce roof reflections and wheel highlights. A nearest text
such as traffic-light or sign alone cannot override supported vehicle context,
compact current source evidence and plausible lamp geometry.
An initial pilot failed at export because NumPy integers were left in boxes;
the corrected revision preserves that failure and reuses its600 verified
detector predictions. Actual
detector recall, lamp identity, reflections and visual realism need image review;
the tests do not establish these accuracies. The pilot covers the same six
train/dev clips with100 frames each. Labels are used only after rendering for
the descriptive vehicle-impact audit. Training and full-data release are separate.
