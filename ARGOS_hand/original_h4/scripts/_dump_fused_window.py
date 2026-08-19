"""Fused-disparity thumbnails for the overview figure, same frames as the raw ones."""
import sys, numpy as np, torch
from pathlib import Path
ROOT=Path("/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4"); ARGOS=ROOT.parents[1]
B=ARGOS/"ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust/d2_full/dataset_2_keyframe_4"
for p in (str(ROOT),str(ROOT/"scripts"),str(ARGOS/"ARGOS_FREEZED/src"),str(ARGOS/"ARGOS-V2/scripts")):
    if p not in sys.path: sys.path.insert(0,p)
from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
from model_design.comparison.run_comparison import drive, load_factory
dev=torch.device("cuda:0")
ad=load_factory("model_design.comparison.ablation_h4:factory_a2")(device="cuda:0")
fm=SEARAFTFlowAdapter(device=dev)
raw=np.load(B/"raw.npz",allow_pickle=False)
# warm-up well before the window so the recurrent state is in steady state
S,E=1088,1105
left=[torch.from_numpy(raw["rgb_left"][i:i+1].copy()).float().to(dev) for i in range(S,E)]
right=[torch.from_numpy(raw["rgb_right"][i:i+1].copy()).float().to(dev) for i in range(S,E)]
frames=[{"index":k,"raw":torch.from_numpy(raw["raw_disparity"][i:i+1].copy()).float().to(dev),
         "raw_valid":torch.from_numpy(raw["raw_valid"][i:i+1].copy()).to(dev),
         "rgb":left[k],"right_rgb":right[k]} for k,i in enumerate(range(S,E))]
def flow(c,p):
    a,b=c["index"],p["index"]; return fm.infer(left[a],left[b]), fm.infer(left[b],left[a])
out=dict(drive(ad,frames,flow))
np.savez_compressed("/dtu/p1/leopam/ARGOS/ARGOS_hand/paper/figure_assets/fused_window.npz",
    fused=np.stack([out[k]["disparity"][0,0].detach().cpu().numpy() for k in range(len(frames))]),
    start=S)
print("OK", len(frames), "frames driven,", S, "->", E-1)
