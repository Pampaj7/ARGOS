#!/usr/bin/env python3
"""D4D sparse-keyframe stereo GT from Zivid structured-light anchors.

Builds rectified-left metric depth + disparity ground truth ONLY at the Zivid
acquisition instants (keyframes). NO dense per-frame GT is fabricated (the scene is
non-rigid; see reports/.../blockers.md).

Validated transform chain (Polaris optical tracker, MiRe45 marker bridge):

    T_cam<-zivid =  inv(T_ps<-cam) . T_ps<-MiRe45 . inv(T_polaris<-MiRe45) . T_polaris<-zivid

  * camera_optical (endoscope) is tracked in frame `polaris_spectra`
  * zivid_optical_frame is tracked in frame `polaris`
  * the two tracker frames are bridged by the MiRe45 marker, seen from both
  * empirically validated by anatomical/instrument alignment of the projected Zivid
    cloud onto the rectified-left endoscope image (photometric check below).

Zivid depth (.npy, metres, optical-axis Z, NaN=invalid) is registered to the Zivid
color camera; backprojection with the color K reproduces the .ply exactly and carries
per-point SNR.

Per anchor outputs:
  left_rectified.png right_rectified.png gt_depth_left.npy gt_disparity_left.npy
  valid_mask.png snr_mask.npy metadata.json diagnostics/
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

RAW_ROOT = Path("/dtu/p1/leopam/ARGOS/dataset/D4D/raw/extracted")
OUT_ROOT = Path("/dtu/p1/leopam/ARGOS/dataset/D4D/processed/keyframe_stereo_gt")


# ----------------------------- calibration --------------------------------
def load_cam(yaml_path: Path) -> dict:
    d = yaml.safe_load(yaml_path.read_text())
    def mat(key, alt, shape):
        v = d.get(key, {})
        v = v.get("data") if isinstance(v, dict) else d.get(alt)
        return np.array(v, float).reshape(shape)
    return {
        "K": mat("camera_matrix", "K", (3, 3)),
        "D": np.array((d.get("distortion_coefficients", {}) or {}).get("data", d.get("D")), float).ravel(),
        "R": mat("rectification_matrix", "R", (3, 3)),
        "P": mat("projection_matrix", "P", (3, 4)),
        "W": int(d["image_width"]), "H": int(d["image_height"]),
        "frame": d.get("camera_name"),
    }


def rectify_maps(cam: dict):
    return cv2.initUndistortRectifyMap(cam["K"], cam["D"], cam["R"], cam["P"],
                                       (cam["W"], cam["H"]), cv2.CV_32FC1)


# ----------------------------- transforms ---------------------------------
def load_tf_series(tf_dir: Path, prefix: str):
    out = []
    for p in glob.glob(str(tf_dir / f"{prefix}_*.json")):
        d = json.loads(Path(p).read_text())
        tr = d["transform"]
        out.append((float(d["timestamp"]),
                    np.array([tr["translation"][k] for k in "xyz"]),
                    np.array([tr["rotation"][k] for k in "xyzw"])))
    out.sort(key=lambda x: x[0])
    return out


def interp_pose(tfs, t: float):
    ts = np.array([x[0] for x in tfs])
    i = np.searchsorted(ts, t)
    i0 = max(0, min(i - 1, len(ts) - 1)); i1 = max(0, min(i, len(ts) - 1))
    a = 0.0 if i0 == i1 else float((t - ts[i0]) / (ts[i1] - ts[i0]))
    trans = (1 - a) * tfs[i0][1] + a * tfs[i1][1]
    rot = Slerp([0, 1], Rotation.from_quat([tfs[i0][2], tfs[i1][2]]))([a])[0]
    T = np.eye(4); T[:3, :3] = rot.as_matrix(); T[:3, 3] = trans
    gap = float(ts[i1] - ts[i0]) if i1 != i0 else 0.0
    off = float(min(abs(t - ts[i0]), abs(t - ts[i1])))
    return T, off, gap


# Per-specimen tf conventions (validated by reprojection alignment; do NOT force one).
#  * mire45_bridge (specimen_1): camera in polaris_spectra, zivid in polaris, bridged via MiRe45 marker.
#  * direct_ps    (specimen_2): both camera and zivid published in polaris_spectra -> no bridge.
#  * direct_polaris (fallback): both published in polaris.
CONV_PREFIXES = {
    "mire45_bridge": ["polaris_spectra_to_camera_optical", "polaris_spectra_to_polaris_spectra_MiRe45",
                      "polaris_to_MiRe45", "polaris_to_zivid_optical_frame"],
    "direct_ps": ["polaris_spectra_to_camera_optical", "polaris_spectra_to_zivid_optical_frame"],
    "direct_polaris": ["polaris_to_camera_optical", "polaris_to_zivid_optical_frame"],
}


def detect_convention(tf_dir: Path) -> str | None:
    have = lambda p: bool(glob.glob(str(tf_dir / f"{p}_*.json")))
    for conv, prefixes in CONV_PREFIXES.items():
        if all(have(p) for p in prefixes):
            return conv
    return None


def load_session_tf(tf_dir: Path) -> dict:
    """Detect the tf convention and load its required series ONCE per session."""
    conv = detect_convention(tf_dir)
    if conv is None:
        raise ValueError("no known tf convention (missing camera/zivid tracker frames)")
    series = {p: load_tf_series(tf_dir, p) for p in CONV_PREFIXES[conv]}
    empty = [p for p, s in series.items() if not s]
    if empty:
        raise ValueError(f"empty tf series: {empty}")
    series["_convention"] = conv
    return series


def chain_cam_from_zivid(series: dict, t: float):
    """Zivid->camera_optical transform for the detected convention. Returns (T, diagnostics)."""
    conv = series["_convention"]
    if conv == "mire45_bridge":
        T_ps_cam, o1, g1 = interp_pose(series["polaris_spectra_to_camera_optical"], t)
        T_ps_M45, o2, g2 = interp_pose(series["polaris_spectra_to_polaris_spectra_MiRe45"], t)
        T_pol_M45, o3, g3 = interp_pose(series["polaris_to_MiRe45"], t)
        T_pol_ziv, o4, g4 = interp_pose(series["polaris_to_zivid_optical_frame"], t)
        T = np.linalg.inv(T_ps_cam) @ T_ps_M45 @ np.linalg.inv(T_pol_M45) @ T_pol_ziv
        gaps = [g1, g2, g3, g4]; offs = {"cam": o1 * 1e3, "ps_MiRe45": o2 * 1e3, "pol_MiRe45": o3 * 1e3, "zivid": o4 * 1e3}
    else:  # direct_ps / direct_polaris
        cam_pre, ziv_pre = CONV_PREFIXES[conv]
        T_cam, o1, g1 = interp_pose(series[cam_pre], t)
        T_ziv, o2, g2 = interp_pose(series[ziv_pre], t)
        T = np.linalg.inv(T_cam) @ T_ziv
        gaps = [g1, g2]; offs = {"cam": o1 * 1e3, "zivid": o2 * 1e3}
    return T, {"convention": conv, "pose_offsets_ms": offs, "max_interp_gap_ms": max(gaps) * 1e3}


# ----------------------------- geometry -----------------------------------
def read_ts(name: str) -> float:
    a, b = Path(name).stem.split("_"); return float(f"{a}.{b}")


def backproject_zivid(depth: np.ndarray, snr: np.ndarray, color: np.ndarray, Kz: np.ndarray):
    H, W = depth.shape
    fy, fx, cx, cy = Kz[1, 1], Kz[0, 0], Kz[0, 2], Kz[1, 2]
    v, u = np.mgrid[0:H, 0:W]
    m = np.isfinite(depth) & (depth > 0)
    Z = depth[m]
    pts = np.stack([(u[m] - cx) / fx * Z, (v[m] - cy) / fy * Z, Z], 1)
    return pts, snr[m], color[m[..., None].repeat(3, 2)].reshape(-1, 3) if color is not None else None


def project_zbuffer(pts_cam, snr, color, R_rect, P_rect, W, H):
    """Project camera-frame points into rectified-left image, z-buffered (nearest wins)."""
    Pr = (R_rect @ pts_cam.T).T
    front = Pr[:, 2] > 1e-6
    Pr, snr = Pr[front], snr[front]
    color = color[front] if color is not None else None
    uvw = (P_rect[:3, :3] @ Pr.T).T
    u = uvw[:, 0] / uvw[:, 2]; v = uvw[:, 1] / uvw[:, 2]; z = Pr[:, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, snr = u[inb].astype(int), v[inb].astype(int), z[inb], snr[inb]
    color = color[inb] if color is not None else None
    depth = np.full((H, W), np.nan, np.float32); snr_map = np.zeros((H, W), np.float32)
    col_map = np.zeros((H, W, 3), np.uint8) if color is not None else None
    order = np.argsort(-z)  # far first -> nearest overwrites
    u, v, z, snr = u[order], v[order], z[order], snr[order]
    depth[v, u] = z; snr_map[v, u] = snr
    if color is not None:
        col_map[v, u] = color[order]
    return depth, snr_map, col_map, int(inb.sum()), int(front.sum())


# ----------------------------- one anchor ---------------------------------
def build_anchor(session: Path, zivid_stem: str, out_dir: Path, make_diag=True, tf_series: dict | None = None) -> dict:
    ci = session / "camera_info"
    left, right = load_cam(ci / "left.yaml"), load_cam(ci / "right.yaml")
    Kz = np.array(yaml.safe_load((ci / "color_camera_info.yaml").read_text())["K"]).reshape(3, 3)
    fx = float(left["P"][0, 0]); baseline_m = float(-right["P"][0, 3] / right["P"][0, 0]) if right["P"][0, 0] else float("nan")
    # baseline from right P: Tx = -fx*B -> B = -Tx/fx
    baseline_m = float(-right["P"][0, 3] / fx)
    W, H = left["W"], left["H"]

    t_scan = read_ts(zivid_stem + ".npy")
    depth = np.load(session / "depth_images" / f"{zivid_stem}.npy")
    snr = np.load(session / "snr_images" / f"{zivid_stem}.npy")
    zcolor = cv2.cvtColor(cv2.imread(str(session / "color_images" / f"{zivid_stem}.png")), cv2.COLOR_BGR2RGB)
    pts_z, snr_pts, col_pts = backproject_zivid(depth, snr, zcolor, Kz)

    series = tf_series if tf_series is not None else load_session_tf(session / "tf")
    T_cam_ziv, tdiag = chain_cam_from_zivid(series, t_scan)
    pts_cam = (T_cam_ziv @ np.concatenate([pts_z, np.ones((len(pts_z), 1))], 1).T).T[:, :3]
    gt_depth, snr_map, col_map, n_in, n_front = project_zbuffer(
        pts_cam, snr_pts, col_pts, left["R"], left["P"], W, H)

    valid = np.isfinite(gt_depth)
    gt_disp = np.where(valid, fx * baseline_m / np.maximum(gt_depth, 1e-9), np.nan).astype(np.float32)

    # nearest stereo frame
    stereo = sorted(glob.glob(str(session / "left_images" / "*.png")))
    st_ts = np.array([read_ts(Path(p).name) for p in stereo])
    si = int(np.argmin(np.abs(st_ts - t_scan)))
    stereo_off_ms = float(abs(st_ts[si] - t_scan) * 1e3)
    lname = Path(stereo[si]).name
    limg = cv2.imread(str(session / "left_images" / lname))
    rimg = cv2.imread(str(session / "right_images" / lname))
    lmapx, lmapy = rectify_maps(left); rmapx, rmapy = rectify_maps(right)
    rectL = cv2.remap(limg, lmapx, lmapy, cv2.INTER_LINEAR)
    rectR = cv2.remap(rimg, rmapx, rmapy, cv2.INTER_LINEAR) if rimg is not None else None

    # ---- quantitative validation ----
    # photometric alignment: projected Zivid color vs rectified-left at overlap
    vv = valid & (col_map.sum(2) > 0)
    photo_mae = float(np.abs(col_map[vv].astype(float) - cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)[vv].astype(float)).mean()) if vv.any() else float("nan")
    # round-trip depth<->disparity
    rt = float(np.nanmax(np.abs(gt_depth[valid] - fx * baseline_m / gt_disp[valid]))) if valid.any() else float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "left_rectified.png"), rectL)
    if rectR is not None:
        cv2.imwrite(str(out_dir / "right_rectified.png"), rectR)
    np.save(out_dir / "gt_depth_left.npy", gt_depth)
    np.save(out_dir / "gt_disparity_left.npy", gt_disp)
    cv2.imwrite(str(out_dir / "valid_mask.png"), (valid * 255).astype(np.uint8))
    np.save(out_dir / "snr_mask.npy", snr_map)

    calib_hash = hashlib.md5((ci / "left.yaml").read_bytes() + (ci / "right.yaml").read_bytes()).hexdigest()[:12]
    meta = {
        "zivid_stem": zivid_stem, "zivid_timestamp": t_scan,
        "stereo_frame": lname, "stereo_timestamp": float(st_ts[si]),
        "stereo_zivid_offset_ms": stereo_off_ms,
        "transform_chain": "inv(T_ps<-cam).T_ps<-MiRe45.inv(T_polaris<-MiRe45).T_polaris<-zivid",
        **tdiag,
        "fx_px": fx, "baseline_m": baseline_m, "baseline_mm": baseline_m * 1e3,
        "resolution": [W, H], "units": "depth_m, disparity_px, positive_left_reference",
        "n_zivid_points": int(len(pts_z)), "n_front": n_front, "n_projected_inbounds": n_in,
        "valid_pixels": int(valid.sum()), "valid_coverage_pct": round(100 * valid.mean(), 3),
        "disp_min": float(np.nanmin(gt_disp)) if valid.any() else None,
        "disp_max": float(np.nanmax(gt_disp)) if valid.any() else None,
        "depth_min_m": float(np.nanmin(gt_depth)) if valid.any() else None,
        "depth_max_m": float(np.nanmax(gt_depth)) if valid.any() else None,
        "photometric_rgb_mae": photo_mae,
        "depth_disp_roundtrip_max": rt,
        "calibration_md5": calib_hash,
        "source_depth": str(session / "depth_images" / f"{zivid_stem}.npy"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    if make_diag:
        dd = out_dir / "diagnostics"; dd.mkdir(exist_ok=True)
        dvis = np.zeros((H, W, 3), np.uint8)
        vv2 = valid
        if vv2.any():
            dn = np.clip((gt_disp - np.nanpercentile(gt_disp, 1)) / (np.nanpercentile(gt_disp, 99) - np.nanpercentile(gt_disp, 1) + 1e-9), 0, 1)
            dvis = cv2.applyColorMap((np.nan_to_num(dn) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            dvis[~vv2] = 0
        blend = cv2.addWeighted(rectL, 0.5, col_map[..., ::-1], 0.5, 0)
        cv2.imwrite(str(dd / "rectL_zivid_disp_blend.png"), np.concatenate([rectL, col_map[..., ::-1], dvis, blend], 1))
    return meta


# ----------------------------- driver -------------------------------------
def session_root(specimen: str) -> Path:
    """Handle both extraction layouts: specimen/specimen/<session> (specimen_1) and
    specimen/<session> (specimen_2+)."""
    dbl = RAW_ROOT / specimen / specimen
    if dbl.is_dir() and any(p.is_dir() and p.name.startswith("20") for p in dbl.glob("*")):
        return dbl
    return RAW_ROOT / specimen


def list_sessions(specimen: str) -> list[Path]:
    root = session_root(specimen)
    if not root.is_dir():
        return []
    return [s for s in sorted(root.glob("*")) if s.is_dir() and (s / "left_images").exists()]


def process_session(specimen: str, session: Path, out_root: Path, make_diag: bool, resume: bool) -> list[dict]:
    """Process all anchors of one session; tf loaded once. Returns per-anchor rows."""
    rows = []
    try:
        series = load_session_tf(session / "tf")
    except Exception as e:
        for a in enumerate_anchors(session):
            rows.append({"specimen": specimen, "session": session.name, "clip": a["clip"], "anchor": a["anchor"],
                         "status_convert": "rejected", "reject_reason": f"tf:{e}"})
        return rows
    for a in enumerate_anchors(session):
        out_dir = out_root / specimen / session.name / (a["clip"] or "clip") / f"{a['anchor']}_anchor"
        if resume and (out_dir / "metadata.json").exists():
            try:
                meta = json.loads((out_dir / "metadata.json").read_text())
                meta.update({"specimen": specimen, "session": session.name, "clip": a["clip"],
                             "anchor": a["anchor"], "out_dir": str(out_dir), "status_convert": "resumed"})
                rows.append(meta); continue
            except Exception:
                pass
        try:
            meta = build_anchor(session, a["zivid_stem"], out_dir, make_diag=make_diag, tf_series=series)
            meta.update({"specimen": specimen, "session": session.name, "clip": a["clip"], "anchor": a["anchor"],
                         "out_dir": str(out_dir), "status_convert": "ok"})
            rows.append(meta)
        except Exception as e:
            rows.append({"specimen": specimen, "session": session.name, "clip": a["clip"], "anchor": a["anchor"],
                         "status_convert": "rejected", "reject_reason": str(e)[:120]})
    return rows


def enumerate_anchors(session: Path):
    cj = session / "clips.json"
    if not cj.exists():
        return []
    clips = json.loads(cj.read_text()).get("clips", [])
    anchors = []
    for c in clips:
        for kind in ("start", "end"):
            geom = c.get(f"{kind}_geometry")
            if geom:
                stem = Path(geom).stem
                if (session / "depth_images" / f"{stem}.npy").exists():
                    anchors.append({"clip": c.get("name"), "anchor": kind, "zivid_stem": stem})
    return anchors


def _worker(argt):
    specimen, session, out_root, make_diag, resume = argt
    return process_session(specimen, session, out_root, make_diag, resume)


def main() -> int:
    import csv
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, help="one session dir (smoke)")
    ap.add_argument("--specimen", default=None, help="single specimen (legacy)")
    ap.add_argument("--specimens", default=None, help="comma list e.g. specimen_1,specimen_2 (default: all extracted)")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--report-root", type=Path, default=None)
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip anchors with existing metadata.json")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # build (specimen, session) work list
    jobs = []
    if args.session:
        jobs = [(args.session.parent.name, args.session)]
    else:
        if args.specimens:
            specs = args.specimens.split(",")
        elif args.specimen:
            specs = [args.specimen]
        else:
            specs = sorted({p.name for p in RAW_ROOT.glob("specimen_*") if p.is_dir()})
        for sp in specs:
            for sess in list_sessions(sp):
                jobs.append((sp, sess))

    if args.dry_run:
        n_anchor = sum(len(enumerate_anchors(s)) for _, s in jobs)
        print(json.dumps({"specimens": sorted({j[0] for j in jobs}), "sessions": len(jobs),
                          "candidate_anchors": n_anchor, "resume": args.resume, "workers": args.workers}, indent=2))
        return 0

    work = [(sp, sess, args.out, not args.no_diag, args.resume) for sp, sess in jobs]
    rows = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for res in pool.imap_unordered(_worker, work):
                rows.extend(res)
                done = {r["specimen"] for r in res}
                print(f"[session done] {list(done)} {res[0]['session'] if res else ''} (+{len(res)} anchors)")
    else:
        for w in work:
            res = _worker(w); rows.extend(res)
            print(f"[session done] {w[0]}/{w[1].name} (+{len(res)} anchors)")

    args.out.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r.get("status_convert") in ("ok", "resumed")]
    keys = sorted({k for r in ok for k in r if not isinstance(r.get(k), dict)}) if ok else []
    with (args.out / "keyframe_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(ok)
    # rejects (chain/tf failures) recorded separately
    rej = [r for r in rows if r.get("status_convert") == "rejected"]
    if rej:
        rk = ["specimen", "session", "clip", "anchor", "reject_reason"]
        with (args.out / "conversion_rejected.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rk, extrasaction="ignore"); w.writeheader(); w.writerows(rej)
    print(f"\n{len(ok)} anchors -> {args.out/'keyframe_manifest.csv'} ({len(rej)} rejected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
