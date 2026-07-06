# D4D Zivid → rectified-left transform chain

## Problem

Zivid structured-light geometry (`depth_images/*.npy`, `pointcloud/*.ply`, metres) is
expressed in the **Zivid color-camera optical frame** (`zivid_optical_frame`, 2448×2048).
Keyframe stereo GT needs it in the **rectified stereo-left** frame (894×714). The two are
rigidly related at each acquisition instant through the **Polaris Spectra optical tracker**.

## Available TF edges (per session, `tf/*.json`)

Each JSON: `{timestamp, parent_frame, child_frame, transform:{translation xyz (m),
rotation xyzw}}`. ROS convention: transform maps **child → parent**
(`p_parent = T · p_child`; translation = child origin expressed in parent).

| publisher prefix | parent | child | ~count |
|---|---|---|---|
| `polaris_spectra_to_camera_optical` | polaris_spectra | camera_optical (endoscope) | 759 |
| `polaris_to_zivid_optical_frame` | polaris | zivid_optical_frame | 1007 |
| `polaris_to_MiRe45` | polaris | MiRe45 | 1089 |
| `polaris_spectra_to_polaris_spectra_MiRe45` | polaris_spectra | (polaris_spectra_)MiRe45 | 804 |
| `polaris_spectra_to_polaris_spectra_MiRe44` | polaris_spectra | MiRe44 | 426 |

Key observation: the **endoscope is tracked in `polaris_spectra`** while the **Zivid is
tracked in `polaris`** — two distinct tracker coordinate definitions. They must be bridged.
The **MiRe45 marker is observed from BOTH** (`polaris_to_MiRe45` and
`polaris_spectra_to_..._MiRe45`), providing the bridge.

## Hypotheses tested (empirically, not assumed)

**H0 — polaris ≡ polaris_spectra (no bridge):**
`T_cam←ziv = inv(T_ps←cam) · T_pol←ziv`.
→ Points land at **z ∈ [−0.69, −0.45] m (behind the camera)**. **REJECTED.**

**H1 — MiRe45 bridge (validated):**
```
T_cam←zivid = inv(T_ps←cam) · T_ps←MiRe45 · inv(T_polaris←MiRe45) · T_polaris←zivid
```
→ z ∈ [0.064, 0.184] m (in front, endoscope working distance), 28–34 % of the Zivid cloud
projects inside the narrow endoscope FOV, median disparity ≈ 30 px. **Anatomical /
instrument alignment** of the projected Zivid colour onto the rectified-left endoscope image
is correct (see `diagnostics/OV_bridge.png`, `CHAIN_sidebyside.png`): both surgical
instruments and the tissue folds line up. **ACCEPTED.**

**H1-inverse** (`inv` of H1): z ∈ [0.20, 0.64] m, 97 % in-bounds but at Zivid working
distance — a degenerate far projection that does not match endoscope geometry. **REJECTED.**

## Validated chain (used by `d4d_keyframe_gt.py`)

```
p_camera_optical = inv(T_ps←cam) · T_ps←MiRe45 · inv(T_polaris←MiRe45) · T_polaris←zivid · p_zivid
p_rect_left      = R_left · p_camera_optical
pixel            = P_left_rect · p_rect_left            (z-buffered, nearest wins)
disparity        = fx · baseline / Z,   fx = P_left_rect[0,0] = 798.32 px,
                                          baseline = −P_right_rect[0,3]/fx = 4.235 mm
```

All four poses are interpolated (translation lerp + rotation slerp) to the Zivid scan
timestamp `t_scan`. Measured pose interpolation offsets at anchors are **< 10 ms** and the
nearest stereo frame is **2–9 ms** from `t_scan` (see `timestamp_analysis.csv`) — the camera
is effectively static over that interval.

## Provenance

Hand-eye calibration files are referenced in each TF (`applied_calibration`):
`8700339__TO__camera_optical__…yaml` (endoscope) and
`8700449__TO__zivid_optical_frame__…yaml` (Zivid), both to their MiRe markers.

## Residual caveat

The bridge assumes `polaris_spectra_MiRe45` and `MiRe45` are the same physical marker
(consistent with the geometry validating). This is an empirical, not documented, identity —
recorded in `blockers.md`. It is corroborated by correct anatomical reprojection, which would
fail under a wrong marker identity.
