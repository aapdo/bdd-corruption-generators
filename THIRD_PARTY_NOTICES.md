# Source and asset terms

The current Fog, flow integration, geometry and persistent Snow code comes from the project implementation. Linear-RGB LowLight and compact-PSF/controlled-sun Glare are the October31 replacements. They do not redistribute the earlier Google Unprocessing port or Flare7K images.

Rain uses external astra-vision/rain-rendering and its legacy particle simulator. Renderer revision: 0a9d8ca739d71d4b09f61567c6f6cea94a79868e. Particle binary revision: 6a130aa7294dd980940416e8e0b21fef7112efaa. The Rain Streak Database belongs to its original authors. URLs and archive SHA256 are in EXTERNAL_ASSETS.json. Obtain and follow each upstream source/asset's own terms; neither the binary, database nor upstream renderer is bundled here. Local endpoint/radiometry adaptations are described in README.md. This repository does not claim complete reproduction of a published pipeline.

NumPy, OpenCV and Pillow remain governed by their respective upstream licenses. BDD100K footage/annotations and depth model weights are not included. Public availability of this repository does not grant redistribution rights over those assets. No additional license grant is inferred for independently owned material.

# Flare7K* PGS addition (2026-11-01)

`recipes/flare7kstar.py` reimplements the PGS equations of
[Qu et al., Harmonizing Light and Darkness (2024)](https://arxiv.org/abs/2404.00313).
The authors' repository did not publish synthesis source at implementation time.
The background-affine transport and temporal smoothing are video additions.
Official Flare7K scattering patterns and Intel DPT-Hybrid model files are external
assets; none are redistributed in this code repository. Flare7K data retains
the S-Lab License 1.0 research terms. Intel's model card identifies Apache 2.0.

# Wu daylight synthesis addition (2026-11-01)

`recipes/flare_wu_day.py` ports the synthesis operators of Wu et al., ICCV 2021,
Google Research `flare_removal/python/{synthesis,utils}.py`, pinned commit
83f1e7b59e762648c953de751d31f75720683622. Copyright 2026 The Google Research
Authors, Apache License 2.0; see licenses/Apache-2.0.txt. Modified to NumPy/OpenCV,
fixed severity comparisons, visible-sun alignment and temporal tracking.
Original captured/simulated PNG assets are external, Wu et al., CC BY 4.0, and
are not redistributed in this repository. See docs/WU_DAYTIME.md.
