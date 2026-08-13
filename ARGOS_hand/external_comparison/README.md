# External comparison boundary

This directory pins official, clean upstream checkouts and keeps their code isolated from ARGOS. The only data boundary is an NPZ with `rgb_left`, `rgb_right` (`float32 [T,3,H,W]`), `raw_disparity` (`float32 [T,1,H,W]`, positive-left), `raw_valid` (`bool`), and exact string `frame_ids`; its JSON sidecar hashes every array. `numpy.load(..., allow_pickle=False)` is mandatory.

Methods are deliberately labelled by their actual temporal access: `nvds_plus_forward_clip4` is a causal four-frame clip (not H4/online); `nvds_plus_bidirectional_offline` and `bidastabilizer_bidirectional_offline` require future frames and are noncausal. BiDA's upstream stabilizer internally negates its input and returns a negative signed result; the runnable worker records raw/refined provenance and an official-wrapper equivalence artifact. Resizing scales disparity by output-width/input-width.

Preflight a generated bridge input without selecting any GPU:

```bash
python external_comparison/run_external_evaluation.py --method bidastabilizer_bidirectional_offline --input /path/sequence.npz --output /path/out.npz --preflight
```

Run without `--preflight` launches a subprocess with `CUDA_VISIBLE_DEVICES=''`. Set `ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1` explicitly for GPU work; physical GPU 0 is rejected. BiDA is `READY` for its TEST_ONLY smoke: verified original checkpoints are converted offline into strict tensor-only derived states. NVDS+ remains `BLOCKED` because no stereo/raw-disparity checkpoint is published. No publishable execution manifest is approved yet.

NVDS (not NVDS+) DPT has a separate, unitless monocular diagnostic path. It runs the pinned official bidirectional script directly on the first 64 left RGB frames of SCARED-C `dataset_2_keyframe_2`, at `480x384`, and writes only normalized uint16 inverse-depth plus upstream OPW diagnostics. It never invokes the disparity bridge or H4 evaluator:

```bash
ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1 /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python \
  external_comparison/run_nvds_dpt_d2_smoke.py
```

The BiDA RAFTStereo robust pair is now pinned and converted once from verified
original bytes into `checkpoints/derived/` tensor-only state dictionaries. The
worker verifies both source and derived hashes, then strict-loads the derived
files with `weights_only=True`; it uses official RAFT-Stereo `corr_implementation=reg`,
32 iterations, padding/cropping, and official BiDA `forward_batch(kernel_size=50)`.
For the non-publishable D2 smoke only, first seed/freeze the 64-frame boundary,
then select physical GPU 1 explicitly (it is logical `cuda:0` in the worker):

```bash
python external_comparison/export_scared_d2_smoke.py --seed /path/seed.npz
ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1 python external_comparison/run_external_evaluation.py \
  --method bidastabilizer_bidirectional_offline --input /path/seed.npz --raw-output /path/raw.npz \
  --output /path/refined.npz --python /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
python external_comparison/export_scared_d2_smoke.py --bridge /path/raw.npz --evaluation /path/d2_smoke.evaluation.npz
ARGOS_EXTERNAL_CUDA_VISIBLE_DEVICES=1 python external_comparison/workers/bidastabilizer.py \
  --input /path/seed.npz --equivalence /path/bidastabilizer.equivalence.json
```

This is a `SMOKE_DIAGNOSTIC`/`TEST_ONLY` artifact: the raw and refined results
come from the same RAFT inference, it has no H4 or publication claim, and its
diagnostic support mask is not the full flow-conditioned D2 paper support.

`package_source_run.py` makes a source run only after a competitor output has passed the bridge contract and an allowlisted evaluation artifact supplies fixed GT/protocol masks. `evaluation_artifacts.lock.json` pins both artifact bytes, its input hash, publication status, and dataset/split/backbone/sequence metadata; `--evaluation` accepts only its artifact ID. `TEST_ONLY` fixtures compile but are labelled as such. A `PUBLISHABLE` artifact additionally requires `--execution-manifest ID`: the JSON and its allowlist hash must attest to the exact method/protocol, upstream commit, READY checkpoint SHA256, bridge input and RGB snapshot hashes, exact frame IDs, and output prediction SHA256. No publishable execution manifest is currently allowlisted. It imports `original_h4`'s frozen `evaluate_scared_bundle` rather than duplicating metrics, then writes the exact SCARED report/manifest layout consumed by `--compile-from`. Its method manifest retains noncausal access; it does not claim causal reset or H4 equivalence. This supports SCARED D2/D7 only, because the frozen compiler has separate layouts for the other datasets. The local `original_h4` dependency must remain versioned and available at `../original_h4`; this boundary deliberately does not vendor or replace it.

`envs/*.environment.yml` are best-effort environment descriptions, not reproducible lockfiles, until fully frozen channel/package artifacts are recorded.

```bash
python external_comparison/package_source_run.py --source-root /path/to/source-runs --method nvds_plus_forward_clip4 --input /path/input.npz --prediction /path/prediction.npz --evaluation TEST_FIXTURE
```

Publication is only through the gate below. It rejects `TEST_ONLY` source runs, resolves each execution-manifest ID through `execution_manifests.lock.json`, verifies the referenced manifest file hash and exact method/protocol/upstream/checkpoint/input/output bindings, and requires the lock's checkpoint to be `READY` before starting the frozen compiler.

```bash
python external_comparison/compile_external_results.py --source-root /path/to/source-runs --datasets scared-d2 --output /path/to/compiled
```

The frozen compiler may be called directly only as a structural `TEST_ONLY` fixture check; that command never produces publishable results:

```bash
python original_h4/model_design/comparison/run_definitive_evaluation.py --compile-from /path/to/source-runs --datasets scared-d2 --output /path/to/compiled
```
