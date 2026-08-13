"""Official ARGOS v2 metrics.

Inputs are ``HW``, ``THW`` or ``BTHW`` (batch=sequence, time=frame).  Public
reports contain only JSON values; ``MetricState`` keeps the private samples.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
import json
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = F = None
__all__ = ['MetricConfig',
    'MetricState',
    'EvaluationState',
    'build_eval_mask',
    'disparity_to_depth',
    'warp_with_flow',
    'compute_spatial_metrics',
    'compute_temporal_metrics',
    'compute_refinement_safety',
    'compute_selective_metrics',
    'evaluate_argos_prediction',
    'paired_bootstrap_ci']

@dataclass(frozen=True)
class MetricConfig:
    """ARGOS thresholds and optional stereo calibration (``Z_mm=fx*b/d``)."""
    fx_px: Optional[float] = None
    baseline_mm: Optional[float] = None
    bad_px: tuple[float, ...] = (1.0, 3.0, 5.0)
    bad_mm: tuple[float, ...] = (2.0, 5.0, 10.0)
    invalid_penalty_px: float = 1000.0
    invalid_penalty_mm: float = 10000.0
    deadband_px: float = 0.0
    deadband_mm: float = 0.0
    catastrophic_px: float = 0.0
    catastrophic_mm: float = 0.0
    temporal_horizons: tuple[int, ...] = (1, 2, 4, 8)
    flow_confidence_threshold: float = 0.0
    risk_budgets: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)
    risk_magnitude_budgets_px: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    risk_magnitude_budgets_mm: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0)
    coverage_targets: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    ece_bins: int = 10

    def __post_init__(self) -> None:
        if (self.fx_px is None) != (self.baseline_mm is None):
            raise ValueError('fx_px and baseline_mm must be supplied together')
        if (
            self.fx_px is not None
            and (
                not np.isfinite(self.fx_px)
                or not np.isfinite(self.baseline_mm)
                or self.fx_px <= 0
                or self.baseline_mm <= 0
            )
        ):
            raise ValueError('calibration must be positive')
        nums = self.bad_px + self.bad_mm + self.risk_magnitude_budgets_px + self.risk_magnitude_budgets_mm
        if any((not np.isfinite(x) or x < 0 for x in nums)):
            raise ValueError('thresholds must be finite and nonnegative')
        if (
            any((not np.isfinite(x) or x < 0 for x in (self.invalid_penalty_px,
                self.invalid_penalty_mm,
                self.deadband_px,
                self.deadband_mm,
                self.catastrophic_px,
                self.catastrophic_mm,
                self.flow_confidence_threshold))) or self.invalid_penalty_px <= 0 or self.invalid_penalty_mm <= 0
        ):
            raise ValueError('penalties must be positive and controls finite/nonnegative')
        if any((not np.isfinite(x) or not 0 <= x <= 1 for x in self.risk_budgets + self.coverage_targets)):
            raise ValueError('risk budgets/coverage must be in [0,1]')
        if any((not isinstance(x, (int, np.integer)) or x <= 0 for x in self.temporal_horizons)):
            raise ValueError('temporal horizons must be positive integers')
        if self.ece_bins < 1:
            raise ValueError('ece_bins must be positive')

@dataclass
class MetricState:
    """Private per-pixel/per-frame samples, tagged by B,T,unit and method."""
    pixel: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    frame: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    tags: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

@dataclass
class EvaluationState:
    """Private state returned only by ``return_state=True``."""
    spatial: MetricState = field(default_factory=MetricState)
    temporal: MetricState = field(default_factory=MetricState)
    safety: MetricState = field(default_factory=MetricState)

def _np(x: Any) -> np.ndarray:
    """Detach Torch safely before NumPy reductions."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        for name in ('resolve_conj', 'resolve_neg'):
            if hasattr(x, name):
                x = getattr(x, name)()
        return x.numpy()
    return np.asarray(x)

def _bthw(x: Any, name: str) -> np.ndarray:
    a = _np(x).astype(float, copy=False)
    if a.ndim == 2:
        return a[None, None]
    if a.ndim == 3:
        return a[None]
    if a.ndim == 4:
        return a
    raise ValueError(f'{name} must be HW, THW, or BTHW')

def _mask(x: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = _np(x)
    if raw.ndim == 2 and raw.shape == shape[-2:]:
        return np.broadcast_to(raw.astype(bool), shape)
    a = _bthw(raw, name).astype(bool, copy=False)
    if a.shape != shape:
        raise ValueError(f'{name} must match BTHW (only HW broadcasts)')
    return a


def _eval_support(gt_valid: Any, protocol_mask: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return fixed GT/protocol support without inspecting predictions."""
    return _mask(gt_valid, shape, 'gt_valid') & _mask(protocol_mask, shape, 'protocol_mask')

def _finite(x: Any) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)

def _depth_mm(x: Any, depth_input_unit: str) -> np.ndarray:
    if depth_input_unit not in ('m', 'mm'):
        raise ValueError("depth_input_unit must be 'm' or 'mm'")
    return _np(x).astype(float, copy=False) * (1000.0 if depth_input_unit == 'm' else 1.0)

def _scalar(x: Any, n: int, frames: int, seq: int) -> dict[str, Any]:
    return {'value': _finite(x), 'support_count': int(n), 'frame_count': int(frames), 'sequence_count': int(seq)}

def _name(prefix: str, threshold: float) -> str:
    return f'{prefix}{(int(threshold) if float(threshold).is_integer() else threshold)}'

def _errors(pred: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    penalty: float) -> tuple[np.ndarray,
    np.ndarray,
    np.ndarray]:
    valid = np.isfinite(pred) & (pred > 0)
    err = np.where(valid, np.abs(pred - target), penalty)
    return (err[support], valid[support], valid)

def _require_valid_target(target: np.ndarray, support: np.ndarray) -> None:
    if np.any(support & ~(np.isfinite(target) & (target > 0))):
        raise ValueError('gt_valid includes invalid target')

def _summary(error: np.ndarray,
    valid: np.ndarray,
    thresholds: Sequence[float],
    prefix: str,
    frames: int,
    seq: int,
    *,
    depth_target: Optional[np.ndarray]=None,
    prediction: Optional[np.ndarray]=None) -> dict[str,
    Any]:
    n = int(error.size)
    stat = lambda f: _scalar(f(error) if n else None, n, frames, seq)
    out = {'MAE': stat(np.mean),
        'Median': stat(np.median),
        'RMSE': stat(lambda x: np.sqrt(np.mean(x * x))),
        'P90': stat(lambda x: np.percentile(x, 90)),
        'P95': stat(lambda x: np.percentile(x, 95)),
        'P99': stat(lambda x: np.percentile(x, 99)),
        'Max': stat(np.max),
        'InvalidRate': _scalar((~valid).mean() if n else None, n, frames, seq)}
    out.update({_name(prefix,
        q): _scalar((~valid | (error > q)).mean() if n else None,
        n,
        frames,
        seq) for q in thresholds})
    if depth_target is not None and prediction is not None:
        z = depth_target
        good = valid & np.isfinite(z) & (z > 0)
        ratio = np.full(n, np.inf)
        ratio[good] = np.maximum(prediction[good] / z[good], z[good] / prediction[good])
        out['AbsRel'] = _scalar(np.mean(error / z) if n else None, n, frames, seq)
        out['SqRel'] = _scalar(np.mean(error * error / z) if n else None, n, frames, seq)
        for i in (1, 2, 3):
            out[f'Delta{i}'] = _scalar(np.mean(ratio < 1.25 ** i) if n else None, n, frames, seq)
        (out['MAE_Z'], out['RMSE_Z']) = (out['MAE'], out['RMSE'])
    return out

def build_eval_mask(gt_valid: Any, protocol_mask: Any) -> np.ndarray:
    """Return fixed ``M_eval=M_GT∩M_protocol``; it never inspects predictions."""
    g = _bthw(gt_valid, 'gt_valid').astype(bool)
    return _eval_support(g, protocol_mask, g.shape)

def disparity_to_depth(disparity_px: Any, fx_px: float, baseline_mm: float) -> np.ndarray:
    """Convert px disparity to mm depth by ``fx_px*baseline_mm/d``; nonpositive is NaN."""
    if fx_px <= 0 or baseline_mm <= 0:
        raise ValueError('positive calibration required')
    d = _np(disparity_px).astype(float, copy=False)
    z = np.full(d.shape, np.nan)
    good = np.isfinite(d) & (d > 0)
    z[good] = fx_px * baseline_mm / d[good]
    return z

def _aggregate(errors: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
    thresholds: Sequence[float],
    prefix: str) -> dict[str,
    Any]:
    """Exact micro and equal-sequence macro aggregates for every spatial summary metric."""
    (B, T) = support.shape[:2]
    n = int(support.sum())
    functions = {'MAE': np.mean,
        'Median': np.median,
        'RMSE': lambda x: np.sqrt(np.mean(x * x)),
        'P90': lambda x: np.percentile(x, 90),
        'P95': lambda x: np.percentile(x, 95),
        'P99': lambda x: np.percentile(x, 99),
        'Max': np.max}
    functions['InvalidRate'] = lambda x, v: (~v).mean()
    for q in thresholds:
        functions[_name(prefix, q)] = lambda x, v, q=q: (~v | (x > q)).mean()

    def one(name: str, fn: Any) -> dict[str, Any]:
        per = []
        for b in range(B):
            (x, v) = (errors[b][support[b]], valid[b][support[b]])
            per.append(float(fn(x,
                v) if name not in ('MAE',
                'Median',
                'RMSE',
                'P90',
                'P95',
                'P99',
                'Max') else fn(x)) if x.size else np.nan)
        good = np.asarray(per)[np.isfinite(per)]
        (x, v) = (errors[support], valid[support])
        micro = fn(x,
            v) if x.size and name not in ('MAE',
            'Median',
            'RMSE',
            'P90',
            'P95',
            'P99',
            'Max') else fn(x) if x.size else None
        higher_is_better = name.startswith('Delta')
        return {'primary_aggregate': 'macro_sequence',
            'macro_sequence': _finite(good.mean()) if good.size else None,
            'micro_pixel': _finite(micro),
            'median_sequence': _finite(np.median(good)) if good.size else None,
            'P95_sequence': _finite(np.percentile(good, 95)) if good.size else None,
            'worst_sequence': _finite(
                good.min()
                if higher_is_better and good.size
                else good.max() if good.size else None
            ),

            'higher_is_better': higher_is_better,
            'support_count': n,
            'frame_count': int(B * T),
            'sequence_count': int(B)}
    return {name: one(name, fn) for (name, fn) in functions.items()}

def compute_spatial_metrics(prediction: Any,
    target: Any,
    gt_valid: Any,
    protocol_mask: Any,
    config: MetricConfig,
    *,
    prediction_depth_mm: Any=None,
    target_depth_mm: Any=None,
    boundary_mask: Any=None,
    unit: str='disparity_px',
    depth_input_unit: str='mm',
    return_state: bool=False) -> Any:
    """Spatial EPE (px) and/or depth error (mm), lower-is-better, on fixed GT support."""
    if unit not in ('disparity_px', 'depth_mm'):
        raise ValueError('unit must be disparity_px or depth_mm')
    (p,
        t) = (_bthw(_depth_mm(prediction, depth_input_unit) if unit == 'depth_mm' else prediction, 'prediction'),
        _bthw(_depth_mm(target, depth_input_unit) if unit == 'depth_mm' else target, 'target'))
    if p.shape != t.shape:
        raise ValueError('prediction/target shape mismatch')
    m = _eval_support(gt_valid, protocol_mask, p.shape)
    _require_valid_target(t, m)
    (B, T) = p.shape[:2]
    st = MetricState()
    report: dict[str, Any] = {'support_count': int(m.sum()), 'aggregate': {}}
    if unit == 'disparity_px':
        (e, valid, _) = _errors(p, t, m, config.invalid_penalty_px)
        st.pixel['disparity_px/prediction'] = e
        st.tags['disparity_px/prediction'] = {'B': B, 'T': T, 'unit': 'px', 'method': 'prediction'}
        px = _summary(e, valid, config.bad_px, 'Bad', B * T, B)
        (px['EPE'], px['MedianEPE'], px['MaxError']) = (px['MAE'], px['Median'], px['Max'])
        report['disparity_px'] = {'prediction': px}
        full = np.where(np.isfinite(p) & (p > 0), np.abs(p - t), config.invalid_penalty_px)
        report['aggregate']['disparity_px'] = _aggregate(full, np.isfinite(p) & (p > 0), m, config.bad_px, 'Bad')
        report['aggregate']['disparity_px']['EPE'] = report['aggregate']['disparity_px']['MAE']
    if boundary_mask is not None:
        bm = _mask(boundary_mask, p.shape, 'boundary_mask')
        report['boundary'] = {}
        for (label, region) in (('boundary', m & bm), ('non_boundary', m & ~bm)):
            (x,
                v,
                _) = _errors(p,
                t,
                region,
                config.invalid_penalty_mm if unit == 'depth_mm' else config.invalid_penalty_px)
            entry = _summary(x,
                v,
                config.bad_mm if unit == 'depth_mm' else config.bad_px,
                'BadMM' if unit == 'depth_mm' else 'Bad',
                B * T,
                B)
            alias = 'Boundary' if label == 'boundary' else 'NonBoundary'
            entry[f'{alias}MAE'] = entry['MAE']
            for q in config.bad_mm if unit == 'depth_mm' else config.bad_px:
                metric_name = _name('BadMM' if unit == 'depth_mm' else 'Bad', q)
                entry[f'{alias}{metric_name}'] = entry[metric_name]
            report['boundary'][label] = entry
    (z, zt) = (p, t) if unit == 'depth_mm' else (prediction_depth_mm, target_depth_mm)
    if z is None and config.fx_px is not None:
        (z,
            zt) = (disparity_to_depth(p, config.fx_px, config.baseline_mm),
            disparity_to_depth(t, config.fx_px, config.baseline_mm))
    if z is not None or zt is not None:
        if z is None or zt is None:
            raise ValueError('direct depth needs prediction and target')
        (z, zt) = (_bthw(z, 'prediction_depth_mm'), _bthw(zt, 'target_depth_mm'))
        if z.shape != p.shape or zt.shape != p.shape:
            raise ValueError('depth shape mismatch')
        _require_valid_target(zt, m)
        (ze, zv, _) = _errors(z, zt, m, config.invalid_penalty_mm)
        st.pixel['depth_mm/prediction'] = ze
        st.tags['depth_mm/prediction'] = {'B': B, 'T': T, 'unit': 'mm', 'method': 'prediction'}
        report['depth_mm'] = {'prediction': _summary(ze,
            zv,
            config.bad_mm,
            'BadMM',
            B * T,
            B,
            depth_target=zt[m],
            prediction=z[m])}
        full = np.where(np.isfinite(z) & (z > 0), np.abs(z - zt), config.invalid_penalty_mm)
        zvfull = np.isfinite(z) & (z > 0)
        agg = _aggregate(full, zvfull, m, config.bad_mm, 'BadMM')

        def rel_aggregate(values: np.ndarray, higher_is_better: bool=False) -> dict[str, Any]:
            per = np.asarray([values[b][m[b]].mean() if m[b].any() else np.nan for b in range(B)])
            good = per[np.isfinite(per)]
            allv = values[m]
            worst = good.min() if higher_is_better and good.size else good.max() if good.size else None
            return {'primary_aggregate': 'macro_sequence',
                'macro_sequence': _finite(good.mean()) if good.size else None,
                'micro_pixel': _finite(allv.mean()) if allv.size else None,
                'median_sequence': _finite(np.median(good)) if good.size else None,
                'P95_sequence': _finite(np.percentile(good, 95)) if good.size else None,
                'worst_sequence': _finite(worst),
                'higher_is_better': higher_is_better,
                'support_count': int(m.sum()),
                'frame_count': B * T,
                'sequence_count': B}
        safe_zt = np.where(np.isfinite(zt) & (zt > 0), zt, 1.0)
        ratio = np.where(zvfull, np.maximum(z / safe_zt, safe_zt / z), np.inf)
        agg.update({'AbsRel': rel_aggregate(full / safe_zt),
            'SqRel': rel_aggregate(full * full / safe_zt),
            'Delta1': rel_aggregate((ratio < 1.25).astype(float), True),
            'Delta2': rel_aggregate((ratio < 1.25 ** 2).astype(float), True),
            'Delta3': rel_aggregate((ratio < 1.25 ** 3).astype(float), True)})
        (agg['MAE_Z'], agg['RMSE_Z']) = (agg['MAE'], agg['RMSE'])
        report['aggregate']['depth_mm'] = agg
    return (report, st) if return_state else report

def _safety(raw: np.ndarray,
    refined: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    config: MetricConfig,
    unit: str) -> tuple[dict[str, Any],
    MetricState]:
    (penalty,
        dead,
        catastrophic,
        thresholds) = (config.invalid_penalty_px,
        config.deadband_px,
        config.catastrophic_px,
        config.bad_px) if unit == 'px' else (config.invalid_penalty_mm,
        config.deadband_mm,
        config.catastrophic_mm,
        config.bad_mm)
    (e0, v0, _) = _errors(raw, target, support, penalty)
    (e1, v1, _) = _errors(refined, target, support, penalty)
    (n, B, T) = (e0.size, support.shape[0], support.shape[1])
    delta = e1 - e0
    val = lambda x, count=n: _scalar(x if count else None, count, B * T, B)
    out = {'unit': unit,
        'HPlus': val(np.maximum(delta, 0).mean() if n else None),
        'BPlus': val(np.maximum(-delta, 0).mean() if n else None),
        'HUR': val((delta > dead).mean() if n else None),
        'BUR': val((delta < -dead).mean() if n else None),
        'Neutral': val((np.abs(delta) <= dead).mean() if n else None),
        'MAEDelta': val(delta.mean() if n else None),
        'thresholds': {}}
    for q in thresholds:
        (b0, b1) = (~v0 | (e0 > q), ~v1 | (e1 > q))
        good = ~b0
        out['thresholds'][str(q)] = {'NewBad': val((good & b1).mean() if n else None),
            'RecoveredBad': val((b0 & ~b1).mean() if n else None),
            'BadDelta': val(b1.mean() - b0.mean() if n else None),
            'IdentityPreservation': val((~b1[good]).mean() if good.any() else None, int(good.sum()))}
    full0 = np.where(np.isfinite(raw) & (raw > 0), np.abs(raw - target), penalty)
    full1 = np.where(np.isfinite(refined) & (refined > 0), np.abs(refined - target), penalty)
    fd = np.array([full1[b,
        t][support[b, t]].mean() - full0[b,
        t][support[b, t]].mean() for b in range(B) for t in range(T) if support[b,
        t].any()])
    out['PixelDeltaDistribution'] = {'Mean': val(delta.mean() if n else None),
        'Median': val(np.median(delta) if n else None),
        'P95': val(np.percentile(delta, 95) if n else None),
        'P99': val(np.percentile(delta, 99) if n else None)}
    out['FrameDegradation'] = {k: _scalar(x,
        len(fd),
        len(fd),
        B) for (k,
        x) in (('Mean', fd.mean() if len(fd) else None),
        ('Median', np.median(fd) if len(fd) else None),
        ('P95', np.percentile(fd, 95) if len(fd) else None),
        ('P99', np.percentile(fd, 99) if len(fd) else None),
        ('Worst', fd.max() if len(fd) else None),
        ('PositiveFraction', (fd > 0).mean() if len(fd) else None),
        ('CatastrophicFraction', (fd > catastrophic).mean() if len(fd) else None))}
    st = MetricState({f'{unit}/raw': e0, f'{unit}/refined': e1},
        {f'{unit}/delta_frame': fd},
        {f'{unit}/delta': {'B': B, 'T': T, 'unit': unit, 'method': 'refined-raw'}})
    full_delta = full1 - full0

    def aggregate(values: np.ndarray, fn: Any=np.mean, higher_is_better: bool=False) -> dict[str, Any]:
        per = np.array([fn(values[b][support[b]]) if support[b].any() else np.nan for b in range(B)], float)
        good = per[np.isfinite(per)]
        allv = values[support]
        return {'primary_aggregate': 'macro_sequence',
            'macro_sequence': _finite(good.mean()) if good.size else None,
            'micro_pixel': _finite(fn(allv)) if allv.size else None,
            'median_sequence': _finite(np.median(good)) if good.size else None,
            'P95_sequence': _finite(np.percentile(good, 95)) if good.size else None,
            'worst_sequence': _finite(
                good.min()
                if higher_is_better and good.size
                else good.max() if good.size else None
            ),

            'higher_is_better': higher_is_better,
            'support_count': int(support.sum()),
            'frame_count': B * T,
            'sequence_count': B}
    frame_values = np.full((B, T), np.nan)
    for b in range(B):
        for j in range(T):
            if support[b, j].any():
                frame_values[b, j] = full_delta[b, j][support[b, j]].mean()
    fgood = frame_values[np.isfinite(frame_values)]

    def frame_aggregate(fn: Any=np.mean) -> dict[str, Any]:
        per = np.asarray(
            [
                fn(frame_values[b, np.isfinite(frame_values[b])])
                if np.isfinite(frame_values[b]).any()
                else np.nan
                for b in range(B)
            ],
            float,
        )
        good = per[np.isfinite(per)]
        return {'primary_aggregate': 'macro_sequence',
            'macro_sequence': _finite(good.mean()) if good.size else None,
            'micro_pixel': _finite(fn(fgood)) if fgood.size else None,
            'median_sequence': _finite(np.median(good)) if good.size else None,
            'P95_sequence': _finite(np.percentile(good, 95)) if good.size else None,
            'worst_sequence': _finite(good.max()) if good.size else None,
            'higher_is_better': False,
            'support_count': int(fgood.size),
            'frame_count': int(fgood.size),
            'sequence_count': B}
    threshold_aggregate = {}
    fullv0 = np.isfinite(raw) & (raw > 0)
    fullv1 = np.isfinite(refined) & (refined > 0)
    for q in thresholds:
        b0 = ~fullv0 | (full0 > q)
        b1 = ~fullv1 | (full1 > q)
        good = ~b0

        def identity_aggregate() -> dict[str, Any]:
            den = good & support
            kept = ~b1 & den
            per = np.asarray([kept[b].sum() / den[b].sum() if den[b].any() else np.nan for b in range(B)])
            g = per[np.isfinite(per)]
            frame_count = int(sum((den[b, j].any() for b in range(B) for j in range(T))))
            return {'primary_aggregate': 'macro_sequence',
                'macro_sequence': _finite(g.mean()) if g.size else None,
                'micro_pixel': _finite(kept.sum() / den.sum()) if den.any() else None,
                'median_sequence': _finite(np.median(g)) if g.size else None,
                'P95_sequence': _finite(np.percentile(g, 95)) if g.size else None,
                'worst_sequence': _finite(g.min()) if g.size else None,
                'higher_is_better': True,
                'support_count': int(den.sum()),
                'frame_count': frame_count,
                'sequence_count': int(g.size)}
        threshold_aggregate[str(q)] = {'NewBad': aggregate((good & b1).astype(float)),
            'RecoveredBad': aggregate((b0 & ~b1).astype(float), higher_is_better=True),
            'BadDelta': aggregate(b1.astype(float) - b0.astype(float)),
            'IdentityPreservation': identity_aggregate()}
    out['aggregate'] = {'HPlus': aggregate(np.maximum(full_delta, 0)),
        'BPlus': aggregate(np.maximum(-full_delta, 0), higher_is_better=True),
        'MAEDelta': aggregate(full_delta),
        'HUR': aggregate((full_delta > dead).astype(float)),
        'BUR': aggregate((full_delta < -dead).astype(float), higher_is_better=True),
        'Neutral': aggregate((np.abs(full_delta) <= dead).astype(float), higher_is_better=True),
        'thresholds': threshold_aggregate,
        'FrameDegradation': {'Mean': frame_aggregate(),
            'Median': frame_aggregate(np.median),
            'P95': frame_aggregate(lambda x: np.percentile(x,
            95)),
            'P99': frame_aggregate(lambda x: np.percentile(x,
            99)),
            'Worst': frame_aggregate(np.max),
            'PositiveFraction': frame_aggregate(lambda x: (x > 0).mean()),
            'CatastrophicFraction': frame_aggregate(lambda x: (x > catastrophic).mean())}}
    return (out, st)

def compute_refinement_safety(raw: Any,
    refined: Any,
    target: Any,
    gt_valid: Any,
    protocol_mask: Any,
    config: MetricConfig,
    *,
    gate: Any=None,
    unit: str='px',
    depth_input_unit: str='mm',
    return_state: bool=False) -> Any:
    """Report px/mm ``max(E_refined-E_raw, 0)`` on fixed support; lower harm is better."""
    if unit not in ('px', 'mm'):
        raise ValueError('unit must be px or mm')
    (r,
        a,
        t) = (_bthw(_depth_mm(raw, depth_input_unit) if unit == 'mm' else raw, 'raw'),
        _bthw(_depth_mm(refined, depth_input_unit) if unit == 'mm' else refined, 'refined'),
        _bthw(_depth_mm(target, depth_input_unit) if unit == 'mm' else target, 'target'))
    if r.shape != a.shape or r.shape != t.shape:
        raise ValueError('raw/refined/target mismatch')
    support = _eval_support(gt_valid, protocol_mask, r.shape)
    _require_valid_target(t, support)
    (out, st) = _safety(r, a, t, support, config, unit)
    if gate is not None:
        gm = _mask(gate, r.shape, 'gate')
        output = np.where(gm, a, r)
        out['output_vs_raw'] = _safety(r, output, t, support, config, unit)[0]
        out['gate'] = _gate_metrics(r, a, t, support, gm, config, unit)
    return (out, st) if return_state else out

def warp_with_flow(values_tk: Any, flow_t_to_tk: Any, mode: str='bilinear', align_corners: bool=True):
    """Torch pull warp ``v_tk(x+flow[dx,dy])`` in pixels; returns values and in-bounds mask."""
    if torch is None:
        raise RuntimeError('warp_with_flow requires torch for map alignment')
    v = values_tk if isinstance(values_tk, torch.Tensor) else torch.as_tensor(values_tk)
    f = flow_t_to_tk if isinstance(flow_t_to_tk, torch.Tensor) else torch.as_tensor(flow_t_to_tk, device=v.device)
    if f.device != v.device:
        f = f.to(v.device)
    original = v.shape
    vb = v[None, None] if v.ndim == 2 else v[None] if v.ndim == 3 else v
    fb = f[None, None] if f.ndim == 3 else f[None] if f.ndim == 4 else f
    if (
        vb.ndim != 4
        or fb.ndim != 5
        or fb.shape[:2] != vb.shape[:2]
        or (fb.shape[2] != 2)
        or (fb.shape[-2:] != vb.shape[-2:])
    ):
        raise ValueError('values HW/THW/BTHW and flow 2HW/T2HW/BT2HW required')
    (B, T, H, W) = vb.shape
    (yy,
        xx) = torch.meshgrid(torch.arange(H, device=v.device, dtype=v.dtype),
        torch.arange(W, device=v.device, dtype=v.dtype),
        indexing='ij')
    (x, y) = (xx + fb[..., 0, :, :], yy + fb[..., 1, :, :])
    gx = (2 * x / (W - 1) - 1 if W > 1 else torch.zeros_like(x)) if align_corners else (2 * x + 1) / W - 1
    gy = (2 * y / (H - 1) - 1 if H > 1 else torch.zeros_like(y)) if align_corners else (2 * y + 1) / H - 1
    with torch.no_grad():
        out = F.grid_sample(vb.reshape(B * T, 1, H, W),
            torch.stack((gx, gy), -1).reshape(B * T, H, W, 2),
            mode=mode,
            padding_mode='zeros',
            align_corners=align_corners).reshape(B,
            T,
            H,
            W)
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    return (out.reshape(original), inside.reshape(original))

def _temporal_summary(samples: list[np.ndarray],
    invalid: list[np.ndarray],
    B: int,
    frames: int,
    penalty: float) -> dict[str,
    Any]:
    x = np.concatenate(samples) if samples else np.empty(0)
    bad = ~np.isfinite(x) | (np.concatenate(invalid) if invalid else np.zeros(x.size, bool))
    return _summary(np.where(bad, penalty, x), ~bad, (), '', frames, B)

def _temporal_aggregate(samples: list[tuple[int, int, np.ndarray]], B: int, penalty: float) -> dict[str, Any]:
    """Aggregate temporal pixels from their original sequence/frame-pair support."""
    names = {'MAE': np.mean,
        'Median': np.median,
        'RMSE': lambda x: np.sqrt(np.mean(x * x)),
        'P90': lambda x: np.percentile(x, 90),
        'P95': lambda x: np.percentile(x, 95),
        'P99': lambda x: np.percentile(x, 99),
        'Max': np.max}
    by_sequence = [[] for _ in range(B)]
    for (b, _, values) in samples:
        by_sequence[b].append(values)
    pooled_raw = (
        np.concatenate([np.concatenate(parts) for parts in by_sequence if parts])
        if any(by_sequence)
        else np.empty(0)
    )
    pooled = np.where(np.isfinite(pooled_raw), pooled_raw, penalty)
    support_count = int(pooled.size)
    frame_pairs = {(b, i) for (b, i, values) in samples if values.size}

    def one(name: str, fn: Any) -> dict[str, Any]:
        per = np.asarray([fn(np.where(np.isfinite(np.concatenate(parts)),
            np.concatenate(parts),
            penalty)) if parts else np.nan for parts in by_sequence],

            float)
        good = per[np.isfinite(per)]
        micro = fn(pooled) if pooled.size else None
        return {'primary_aggregate': 'macro_sequence',
            'macro_sequence': _finite(good.mean()) if good.size else None,
            'micro_pixel': _finite(micro),
            'median_sequence': _finite(np.median(good)) if good.size else None,
            'P95_sequence': _finite(np.percentile(good, 95)) if good.size else None,
            'worst_sequence': _finite(good.max()) if good.size else None,
            'higher_is_better': False,
            'support_count': support_count,
            'frame_pair_count': len(frame_pairs),
            'frame_count': len(frame_pairs),
            'sequence_count': int(sum((bool(parts) for parts in by_sequence)))}
    invalid_per = np.asarray(
        [(~np.isfinite(np.concatenate(parts))).mean() if parts else np.nan for parts in by_sequence],
        float,
    )
    good = invalid_per[np.isfinite(invalid_per)]
    invalid = ~np.isfinite(pooled_raw)
    out = {name: one(name, fn) for (name, fn) in names.items()}
    out['InvalidRate'] = {'primary_aggregate': 'macro_sequence',
        'macro_sequence': _finite(good.mean()) if good.size else None,
        'micro_pixel': _finite(invalid.mean()) if pooled.size else None,
        'median_sequence': _finite(np.median(good)) if good.size else None,
        'P95_sequence': _finite(np.percentile(good, 95)) if good.size else None,
        'worst_sequence': _finite(good.max()) if good.size else None,
        'higher_is_better': False,
        'support_count': support_count,
        'frame_pair_count': len(frame_pairs),
        'frame_count': len(frame_pairs),
        'sequence_count': int(sum((bool(parts) for parts in by_sequence)))}
    return out

def _alignment(k: int,
    B: int,
    T: int,
    H: int,
    W: int,
    alignment: Mapping[int, Mapping[str, Any]],
    pre: Optional[Mapping[str, Any]]) -> tuple[Optional[dict[str, np.ndarray]],
    str]:
    if pre is not None:
        bundle = pre.get(k, pre.get(str(k)))
        if bundle is None:
            return (None, 'none')
        if 'warped_raw_tk' not in bundle and 'warped_prediction_tk' not in bundle:
            raise ValueError('prewarped needs warped_raw_tk (or raw-only alias warped_prediction_tk)')
        for name in ('warp_valid', 'warped_gt_valid_tk', 'warped_protocol_mask_tk'):
            if name not in bundle:
                raise ValueError(f'prewarped needs {name}')
        d = {name: _bthw(value,
            name) for (name,
            value) in bundle.items() if name not in ('is_gt',
            'warp_valid',
            'warped_gt_valid_tk',
            'warped_protocol_mask_tk')}
        d['warp_valid'] = _mask(bundle['warp_valid'], (B, T - k, H, W), 'warp_valid')
        d['warped_gt_valid_tk'] = _mask(bundle['warped_gt_valid_tk'], (B, T - k, H, W), 'warped_gt_valid_tk')
        d['warped_protocol_mask_tk'] = _mask(bundle['warped_protocol_mask_tk'],
            (B, T - k, H, W),
            'warped_protocol_mask_tk')
        d['is_gt'] = bool(bundle.get('is_gt', False))
        shape = (B, T - k, H, W)
        if any((a.shape != shape for (name, a) in d.items() if name != 'is_gt')):
            raise ValueError('prewarped maps must be B,T-k,H,W')
        return (d, 'prewarped')
    item = alignment.get(k, alignment.get(str(k)))
    if item is None:
        return (None, 'none')
    (has_flow, has_coords) = ('flow_t_to_tk' in item, 'coords_t_to_tk' in item)
    if has_flow == has_coords:
        raise ValueError('alignment_by_horizon[k] needs exactly one flow or coords map')
    if 'corr_valid' not in item:
        raise ValueError('alignment_by_horizon[k] needs corr_valid')
    if torch is None:
        raise RuntimeError('map alignment requires torch; use prewarped maps without torch')
    flow = _np(item['flow_t_to_tk']) if has_flow else _np(item['coords_t_to_tk'])
    if has_coords:
        if flow.shape != (B, T - k, H, W, 2):
            raise ValueError('coords_t_to_tk must be B,T-k,H,W,2')
        (yy, xx) = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        flow = np.moveaxis(flow - np.stack((xx, yy), -1), -1, 2)
    if flow.shape != (B, T - k, 2, H, W):
        raise ValueError('flow_t_to_tk must be B,T-k,2,H,W')
    d = {'flow': flow, 'corr_valid': _mask(item['corr_valid'], (B, T - k, H, W), 'corr_valid')}
    for key in ('flow_valid', 'confidence', 'occluded'):
        if key in item:
            d[key] = _mask(item[key], (B, T - k, H, W), key) if key != 'confidence' else _bthw(item[key], key)
    d['is_gt'] = bool(item.get('is_gt', False))
    return (d, 'map')

def compute_temporal_metrics(prediction: Any,
    target: Any,
    gt_valid: Any,
    protocol_mask: Any,
    config: MetricConfig,
    *,
    unit: str,
    refined: Any=None,
    gate: Any=None,
    alignment_by_horizon: Optional[Mapping[int, Mapping[str, Any]]]=None,
    keyframe_mask: Any=None,
    prewarped: Optional[Mapping[str, Any]]=None,
    depth_input_unit: str='mm',
    return_state: bool=False) -> Any:
    """Report lower-is-better ``|Δprediction-Δtarget|`` px/mm errors on fixed support."""
    if unit not in ('px', 'mm'):
        raise ValueError('unit must be px or mm')
    (p,
        t) = (_bthw(_depth_mm(prediction, depth_input_unit) if unit == 'mm' else prediction, 'prediction'),
        _bthw(_depth_mm(target, depth_input_unit) if unit == 'mm' else target, 'target'))
    if p.shape != t.shape:
        raise ValueError('temporal prediction/target mismatch')
    if alignment_by_horizon is not None and prewarped is not None:
        raise ValueError('map and prewarped alignment are mutually exclusive')
    m = _eval_support(gt_valid, protocol_mask, p.shape)
    _require_valid_target(t, m)
    (B, T, H, W) = p.shape
    methods = {'raw': p}
    if refined is not None:
        a = _bthw(_depth_mm(refined, depth_input_unit) if unit == 'mm' else refined, 'refined')
        if a.shape != p.shape:
            raise ValueError('refined shape mismatch')
        methods['refined'] = a
        if gate is not None:
            methods['output'] = np.where(_mask(gate, p.shape, 'gate'), a, p)
    elif gate is not None:
        raise ValueError('gate requires refined')
    key = np.zeros((B, T), bool)
    key[:, 0] = True if keyframe_mask is None else False
    if keyframe_mask is not None:
        km = _np(keyframe_mask).astype(bool)
        if km.shape != (B, T):
            raise ValueError('keyframe_mask must be B,T')
        key = km
    alignment_by_horizon = alignment_by_horizon or {}
    out = {}
    state = MetricState()
    penalty = config.invalid_penalty_mm if unit == 'mm' else config.invalid_penalty_px
    for k in config.temporal_horizons:
        if k >= T:
            continue
        (al, kind) = _alignment(k, B, T, H, W, alignment_by_horizon, prewarped)
        if al is not None and kind == 'prewarped' and (unit == 'mm') and (depth_input_unit == 'm'):
            for (map_name, values) in tuple(al.items()):
                if (
                    map_name.startswith('warped_') and map_name not in ('warped_gt_valid_tk',
                        'warped_protocol_mask_tk')
                ):
                    al[map_name] = values * 1000.0
        one = {'diagnostic_grid_based': True,
            'flow_conditioned_proxy': bool(al is not None and (not al.get('is_gt', False))),
            'methods': {}}
        for (name, x) in methods.items():
            result = {}
            aggregate_samples: dict[str,
                list[tuple[int, int, np.ndarray]]] = {f'DTCE_grid_{unit}': [],
                f'Drift_{unit}': []}

            def add_samples(metric: str, values: np.ndarray, valid: np.ndarray, pair: int) -> np.ndarray:
                picked = values[valid]
                aggregate_samples.setdefault(metric,
                    []).extend(((b, pair, values[b][valid[b]]) for b in range(B) if valid[b].any()))
                return picked
            grid = []
            drift = []
            for i in range(T - k):
                sup = m[:, i] & m[:, i + k]
                grid_value = np.abs(x[:, i + k] - x[:, i] - (t[:, i + k] - t[:, i]))
                grid.append(add_samples(f'DTCE_grid_{unit}', grid_value, sup, i))
                anchors = np.where(key[:, i])[0]
                if anchors.size:
                    drift_value = np.abs(x[:, i + k] - t[:, i + k] - (x[:, i] - t[:, i]))
                    drift_valid = np.zeros_like(sup)
                    drift_valid[anchors] = sup[anchors]
                    drift.append(add_samples(f'Drift_{unit}', drift_value, drift_valid, i))
            gridname = f'DTCE_grid_{unit}'
            driftname = f'Drift_{unit}'
            result[gridname] = _temporal_summary(grid, [], B, B * (T - k), penalty)
            result[driftname] = _temporal_summary(drift, [], B, B * (T - k), penalty)
            state.pixel[f'{unit}/{name}/{k}/{gridname}'] = np.concatenate(grid) if grid else np.empty(0)
            state.pixel[f'{unit}/{name}/{k}/{driftname}'] = np.concatenate(drift) if drift else np.empty(0)
            state.tags[f'{unit}/{name}/{k}/{gridname}'] = {'B': B, 'T': T, 'unit': unit, 'method': name, 'horizon': k}
            pre_key = {'raw': 'warped_raw_tk', 'refined': 'warped_refined_tk', 'output': 'warped_output_tk'}[name]
            if name == 'raw' and pre_key not in (al or {}) and (al is not None) and ('warped_prediction_tk' in al):
                pre_key = 'warped_prediction_tk'
            if al is not None and (kind != 'prewarped' or pre_key in al):
                corr = []
                wdtce = []
                opw = []
                aggregate_samples['TEPE_corr_px' if unit == 'px' else 'W_DTCE_mm'] = []
                aggregate_samples[f'OPW_{unit}'] = []
                for i in range(T - k):
                    base = m[:, i]
                    if kind == 'prewarped':
                        wp = al[pre_key][:, i]
                        wt = al.get('warped_target_tk')
                        valid = base & al['warp_valid'][:,
                            i] & al['warped_gt_valid_tk'][:,
                            i] & al['warped_protocol_mask_tk'][:,
                            i]
                    else:
                        flow = torch.as_tensor(al['flow'][:, i], dtype=torch.float32)
                        (wp, inside) = warp_with_flow(torch.as_tensor(x[:, i + k], dtype=torch.float32), flow)
                        wp = _np(wp)
                        (warped_eval,
                            _) = warp_with_flow(torch.as_tensor(m[:, i + k], dtype=torch.float32),
                            flow,
                            mode='nearest')
                        valid = base & _np(warped_eval).astype(bool) & _np(inside).astype(bool) & al['corr_valid'][:,
                            i]
                        if 'flow_valid' in al:
                            valid &= al['flow_valid'][:, i]
                        if 'confidence' in al:
                            valid &= al['confidence'][:, i] >= config.flow_confidence_threshold
                        if 'occluded' in al:
                            valid &= ~al['occluded'][:, i]
                        (wt, _) = warp_with_flow(torch.as_tensor(t[:, i + k], dtype=torch.float32), flow)
                        wt = _np(wt)
                    opw_value = np.abs(wp - x[:, i])
                    opw.append(add_samples(f'OPW_{unit}', opw_value, valid, i))
                    if wt is not None:
                        target_delta = wt[:, i] - t[:, i] if kind == 'prewarped' and wt.ndim == 4 else wt - t[:, i]
                        wdtce_value = np.abs(wp - x[:, i] - target_delta)
                        wdtce.append(add_samples('TEPE_corr_px' if unit == 'px' else 'W_DTCE_mm',
                            wdtce_value,
                            valid,
                            i))
                        if unit == 'px':
                            corr.append(wdtce[-1])
                if unit == 'px':
                    result['TEPE_corr_px'] = _temporal_summary(corr, [], B, B * (T - k), penalty)
                else:
                    result['W_DTCE_mm'] = _temporal_summary(wdtce, [], B, B * (T - k), penalty)
                result[f'OPW_{unit}'] = _temporal_summary(opw, [], B, B * (T - k), penalty)
                for (metric,
                    values) in (('TEPE_corr_px' if unit == 'px' else 'W_DTCE_mm', corr if unit == 'px' else wdtce),
                    (f'OPW_{unit}', opw)):
                    state.pixel[f'{unit}/{name}/{k}/{metric}'] = np.concatenate(values) if values else np.empty(0)
                    state.tags[f'{unit}/{name}/{k}/{metric}'] = {'B': B,
                        'T': T,
                        'unit': unit,
                        'method': name,
                        'horizon': k}
            result['aggregate'] = {metric: _temporal_aggregate(samples,
                B,
                penalty) for (metric,
                samples) in aggregate_samples.items()}
            one['methods'][name] = result
        out[str(k)] = one
    return (out, state) if return_state else out

def _curve(score: np.ndarray, harm: np.ndarray, mag: np.ndarray, config: MetricConfig, unit: str) -> dict[str, Any]:
    n = len(score)
    if not n:
        return {'risk_coverage_curve': [],
            'AURC_harm': None,
            'AURC_magnitude': None,
            'oracle_AURC_harm': None,
            'oracle_AURC_magnitude': None,
            'excess_AURC_harm': None,
            'excess_AURC_magnitude': None}

    def run(order: np.ndarray, grouped: bool):
        ends = np.r_[np.flatnonzero(np.diff(score[order])) + 1, n] if grouped else np.arange(1, n + 1)
        cov = ends / n
        h = np.cumsum(harm[order])[ends - 1] / ends
        m = np.cumsum(mag[order])[ends - 1] / ends
        return (ends,
            cov,
            h,
            m,
            float(np.sum(np.diff(np.r_[0.0, cov]) * h)),
            float(np.sum(np.diff(np.r_[0.0, cov]) * m)))
    o = np.argsort(score, kind='stable')
    (ends, cov, h, m, ah, am) = run(o, True)
    (_, _, _, _, oh, _) = run(np.argsort(harm, kind='stable'), False)
    (_, _, _, _, _, omag) = run(np.argsort(mag, kind='stable'), False)
    curve = [{'coverage': float(c),
        'risk_harm': float(a),
        'risk_magnitude': float(b),
        'threshold': float(score[o[e - 1]])} for (e,
        c,
        a,
        b) in zip(ends,
        cov,
        h,
        m)]
    pick = lambda q: next((x for x in curve if x['coverage'] >= q), curve[-1])
    budgets = {str(q): max((x['coverage'] for x in curve if x['risk_harm'] <= q),
        default=0.0) for q in config.risk_budgets}
    mbudgets = {str(q): max((x['coverage'] for x in curve if x['risk_magnitude'] <= q),
        default=0.0) for q in (config.risk_magnitude_budgets_mm if unit == 'mm' else config.risk_magnitude_budgets_px)}
    return {'risk_coverage_curve': curve,
        'AURC': ah,
        'AURC_harm': ah,
        'AURC_magnitude': am,
        'oracle_AURC_harm': oh,
        'oracle_AURC_magnitude': omag,
        'excess_AURC_harm': ah - oh,
        'excess_AURC_magnitude': am - omag,
        'risk_at_coverage': {str(q): pick(q) for q in config.coverage_targets},
        'coverage_at_harm_budget': budgets,
        'coverage_at_magnitude_budget': mbudgets}

def _empty_selective(config: MetricConfig, unit: str) -> dict[str, Any]:
    return {'risk_coverage_curve': [],
        'AURC': None,
        'AURC_harm': None,
        'AURC_magnitude': None,
        'oracle_AURC_harm': None,
        'oracle_AURC_magnitude': None,
        'excess_AURC_harm': None,
        'excess_AURC_magnitude': None,
        'risk_at_coverage': {str(q): None for q in config.coverage_targets},
        'coverage_at_harm_budget': {str(q): None for q in config.risk_budgets},
        'coverage_at_magnitude_budget': {
            str(q): None
            for q in (
                config.risk_magnitude_budgets_mm
                if unit == 'mm'
                else config.risk_magnitude_budgets_px
            )
        },
        'AUROC': None,
        'AUPRC': None,
        'Brier': None,
        'NLL': None,
        'ECE': None,
        'Precision': None,
        'Recall': None,
        'F1': None,
        'support_count': 0}

def compute_selective_metrics(risk_score: Any,
    raw: Any,
    refined: Any,
    target: Any,
    gt_valid: Any,
    protocol_mask: Any,
    config: MetricConfig,
    *,
    harm_probability: Any=None,
    gate: Any=None,
    unit: str='px',
    depth_input_unit: str='mm') -> dict[str,
    Any]:
    """Evaluate whether higher risk ranks positive px/mm refined-minus-raw error on fixed support."""
    if unit not in ('px', 'mm'):
        raise ValueError('unit must be px or mm')
    scale = lambda x: _depth_mm(x, depth_input_unit) if unit == 'mm' else x
    (r,
        a,
        t,
        q) = (_bthw(scale(raw), 'raw'),
        _bthw(scale(refined), 'refined'),
        _bthw(scale(target), 'target'),
        _bthw(risk_score, 'risk_score'))
    if len({r.shape, a.shape, t.shape, q.shape}) != 1:
        raise ValueError('selective inputs mismatch')
    m = _eval_support(gt_valid, protocol_mask, r.shape)
    _require_valid_target(t, m)
    penalty = config.invalid_penalty_px if unit == 'px' else config.invalid_penalty_mm
    dead = config.deadband_px if unit == 'px' else config.deadband_mm
    (e0, _, _) = _errors(r, t, m, penalty)
    (e1, _, _) = _errors(a, t, m, penalty)
    score = q[m]
    if not score.size:
        return _empty_selective(config, unit)
    if np.any(~np.isfinite(score)):
        raise ValueError('risk_score must be finite on support')
    harm = e1 - e0 > dead
    mag = np.maximum(e1 - e0, 0)
    out = _curve(score, harm, mag, config, unit)
    n = len(score)
    ranks = np.empty(n)
    order = np.argsort(score, kind='stable')
    start = 0
    while start < n:
        end = start + 1
        while end < n and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    pos = int(harm.sum())
    out['AUROC'] = _finite((ranks[harm].sum() - pos * (pos + 1) / 2) / (pos * (n - pos))) if pos and n - pos else None
    desc = np.argsort(-score, kind='stable')
    y = harm[desc]
    ends = np.r_[np.flatnonzero(np.diff(score[desc])) + 1, n]
    rec = np.cumsum(y)[ends - 1] / max(pos, 1)
    prec = np.cumsum(y)[ends - 1] / ends
    out['AUPRC'] = _finite(np.sum(np.diff(np.r_[0.0, rec]) * prec)) if pos else None
    out.update({'Brier': None, 'NLL': None, 'ECE': None, 'support_count': n})
    if harm_probability is not None:
        probability = _bthw(harm_probability, 'harm_probability')
        if probability.shape != r.shape:
            raise ValueError('harm_probability shape mismatch')
        probability = probability[m]
        if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
            raise ValueError('harm_probability must be finite in [0,1] on support')
        clip = np.clip(probability, 1e-12, 1 - 1e-12)
        out['Brier'] = _finite(np.mean((probability - harm) ** 2))
        out['NLL'] = _finite(-np.mean(harm * np.log(clip) + ~harm * np.log(1 - clip)))
        ece = 0.0
        for i in range(config.ece_bins):
            (lo, hi) = (i / config.ece_bins, (i + 1) / config.ece_bins)
            z = (probability >= lo) & (probability < hi if i + 1 < config.ece_bins else probability <= hi)
            if z.any():
                ece += z.mean() * abs(probability[z].mean() - harm[z].mean())
        out['ECE'] = _finite(ece)
    accepted = np.ones(r.shape, bool) if gate is None else _mask(gate, r.shape, 'gate')
    reject = ~accepted[m]
    tp = (reject & harm).sum()
    precision = tp / reject.sum() if reject.any() else None
    recall = tp / pos if pos else None
    out.update({'Precision': _finite(precision),
        'Recall': _finite(recall),
        'F1': _finite(2 * precision * recall / (precision + recall)) if precision is not None
        and recall is not None
        and precision + recall else None})
    return out

def _gate_metrics(raw: np.ndarray,
    refined: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    gate: np.ndarray,
    config: MetricConfig,
    unit: str) -> dict[str,
    Any]:
    (penalty,
        dead) = (config.invalid_penalty_px,
        config.deadband_px) if unit == 'px' else (config.invalid_penalty_mm,
        config.deadband_mm)
    (e0, _, _) = _errors(raw, target, support, penalty)
    (e1, _, _) = _errors(refined, target, support, penalty)
    g = gate[support]
    d = e1 - e0
    n = len(d)
    val = lambda x: _scalar(x if n else None, n, support.shape[0] * support.shape[1], support.shape[0])
    return {'Coverage': val(g.mean() if n else None),
        'AcceptedHarm': val((d[g] > dead).mean() if g.any() else None),
        'RejectedBenefit': val((d[~g] < -dead).mean() if (~g).any() else None),
        'FalseUpdate': val((g & (d > dead)).mean() if n else None),
        'MissedRecovery': val((~g & (d < -dead)).mean() if n else None)}

def evaluate_argos_prediction(*,
    raw_disparity: Any=None,
    refined_disparity: Any=None,
    gt_disparity: Any=None,
    raw_depth: Any=None,
    refined_depth: Any=None,
    gt_depth: Any=None,
    gt_valid: Any,
    protocol_mask: Any,
    config: MetricConfig,
    boundary_mask: Any=None,
    gate: Any=None,
    risk: Any=None,
    risk_score: Any=None,
    harm_probability: Any=None,
    depth_input_unit: str='mm',
    alignment_by_horizon: Optional[Mapping[int, Mapping[str, Any]]]=None,
    keyframe_mask: Any=None,
    sequence_ids: Optional[Sequence[Any]]=None,
    frame_ids: Optional[Sequence[Any]]=None,
    return_state: bool=False,
    prewarped: Optional[Mapping[str, Any]]=None) -> Any:
    """Return JSON-safe disparity/depth reports on fixed support; macro sequence is primary."""
    if raw_disparity is None and raw_depth is None:
        raise ValueError('provide raw disparity and/or raw depth')
    if depth_input_unit not in ('m', 'mm'):
        raise ValueError("depth_input_unit must be 'm' or 'mm'")
    if risk is not None and risk_score is not None:
        raise ValueError('provide only one of risk and risk_score')
    if (raw_disparity is None) != (gt_disparity is None) or (raw_disparity is None) != (refined_disparity is None):
        raise ValueError('disparity needs raw/refined/gt together')
    if (raw_depth is None) != (gt_depth is None) or (raw_depth is None) != (refined_depth is None):
        raise ValueError('depth needs raw/refined/gt together')
    if (
        raw_disparity is not None and raw_depth is not None and (prewarped is not None) and (not isinstance(prewarped,
            Mapping) or not {'disparity_px',
            'depth_mm'}.issubset(prewarped))
    ):
        raise ValueError('when disparity and direct depth are both evaluated, prewarped must be unit-scoped')
    shape = _bthw(raw_disparity if raw_disparity is not None else raw_depth, 'raw').shape
    (B, T) = shape[:2]
    sequence_ids = list(range(B)) if sequence_ids is None else list(sequence_ids)
    frame_ids = list(range(T)) if frame_ids is None else list(frame_ids)
    if len(sequence_ids) != B or len(set(sequence_ids)) != B or len(frame_ids) != T or (len(set(frame_ids)) != T):
        raise ValueError('sequence_ids/frame_ids must be unique and match B/T')
    report = {'primary_aggregate': 'macro_sequence',
        'sequence_ids': sequence_ids,
        'frame_ids': frame_ids,
        'spatial': {},
        'safety': {},
        'gate': {},
        'selective': {},
        'temporal': {},
        'per_sequence': [{'sequence_id': x, 'metrics': {}} for x in sequence_ids],
        'per_frame': [{'sequence_id': sequence_ids[b],
            'frame_id': frame_ids[t],
            'metrics': {}} for b in range(B) for t in range(T)],

        'aggregate': {}}
    state = EvaluationState()

    def add(label: str,
        raw: Any,
        refined: Any,
        target: Any,
        unit: str,
        depth: bool=False,
        prewarped_depth_input_unit: str='mm') -> None:
        metric_unit = 'depth_mm' if depth else 'disparity_px'
        penalty = config.invalid_penalty_mm if depth else config.invalid_penalty_px
        (sp,
            ss) = compute_spatial_metrics(raw,
            target,
            gt_valid,
            protocol_mask,
            config,
            boundary_mask=boundary_mask,
            unit=metric_unit,
            return_state=True)
        sr = compute_spatial_metrics(refined,
            target,
            gt_valid,
            protocol_mask,
            config,
            boundary_mask=boundary_mask,
            unit=metric_unit)
        aggregate_key = 'depth_mm' if depth else 'disparity_px'
        report['spatial'][label] = {'raw': sp, 'refined': sr}
        report['aggregate'][label] = {'raw': sp['aggregate'].get(aggregate_key),
            'refined': sr['aggregate'].get(aggregate_key)}
        (saf,
            st) = _safety(_bthw(raw, 'raw'),
            _bthw(refined, 'refined'),
            _bthw(target, 'target'),
            _eval_support(gt_valid, protocol_mask, shape),
            config,
            'mm' if depth else 'px')
        report['safety'][label] = saf
        if gate is not None:
            gm = _mask(gate, shape, 'gate')
            output = np.where(gm, _bthw(refined, 'refined'), _bthw(raw, 'raw'))
            report['gate'][label] = _gate_metrics(_bthw(raw, 'raw'),
                _bthw(refined, 'refined'),
                _bthw(target, 'target'),
                _eval_support(gt_valid, protocol_mask, shape),
                gm,
                config,
                'mm' if depth else 'px')
            output_spatial = compute_spatial_metrics(output,
                target,
                gt_valid,
                protocol_mask,
                config,
                unit=metric_unit,
                boundary_mask=boundary_mask)
            report['spatial'][label]['output'] = output_spatial
            report['aggregate'][label]['output'] = output_spatial['aggregate'].get(aggregate_key)
            report['safety'][label]['output_vs_raw'] = _safety(_bthw(raw, 'raw'),
                output,
                _bthw(target, 'target'),
                _eval_support(gt_valid, protocol_mask, shape),
                config,
                'mm' if depth else 'px')[0]
        selected_risk = risk_score if risk_score is not None else risk
        if selected_risk is not None:
            report['selective'][label] = compute_selective_metrics(selected_risk,
                raw,
                refined,
                target,
                gt_valid,
                protocol_mask,
                config,
                harm_probability=harm_probability,
                gate=gate,
                unit='mm' if depth else 'px')
        unit_prewarped = prewarped.get(label) if isinstance(prewarped,
            Mapping) and label in prewarped else prewarped if not (depth and raw_depth is None) else None
        if depth and prewarped_depth_input_unit == 'm' and isinstance(unit_prewarped, Mapping):
            unit_prewarped = {horizon: {name: _depth_mm(value,
                'm') if name.startswith('warped_') and name not in ('warped_gt_valid_tk',
                'warped_protocol_mask_tk') else value for (name,
                value) in bundle.items()} for (horizon,

                bundle) in unit_prewarped.items()}
        (temporal,
            ts) = compute_temporal_metrics(raw,
            target,
            gt_valid,
            protocol_mask,
            config,
            unit='mm' if depth else 'px',
            refined=refined,
            gate=gate,
            alignment_by_horizon=alignment_by_horizon,
            keyframe_mask=keyframe_mask,
            prewarped=unit_prewarped,
            return_state=True)
        report['temporal'][label] = temporal
        state.temporal.pixel.update(ts.pixel)
        state.temporal.tags.update(ts.tags)
        full_support = _eval_support(gt_valid, protocol_mask, shape)
        (rr, aa, tt) = (_bthw(raw, 'raw'), _bthw(refined, 'refined'), _bthw(target, 'target'))
        er = np.where(np.isfinite(rr) & (rr > 0), np.abs(rr - tt), penalty)
        ea = np.where(np.isfinite(aa) & (aa > 0), np.abs(aa - tt), penalty)
        methods_for_detail = {'raw': rr, 'refined': aa}
        if gate is not None:
            methods_for_detail['output'] = np.where(_mask(gate, shape, 'gate'), aa, rr)
        thresholds = config.bad_mm if depth else config.bad_px
        prefix = 'BadMM' if depth else 'Bad'

        def detail(pred: np.ndarray, target_local: np.ndarray, q: np.ndarray, frames: int) -> dict[str, Any]:
            (ee, vv, _) = _errors(pred, target_local, q, penalty)
            return _summary(ee,
                vv,
                thresholds,
                prefix,
                frames,
                1,
                depth_target=target_local[q] if depth else None,
                prediction=pred[q] if depth else None)
        for b in range(B):
            q = full_support[b:b + 1]
            report['per_sequence'][b]['metrics'][label] = {name: detail(pred[b:b + 1],
                tt[b:b + 1],
                q,
                T) for (name,
                pred) in methods_for_detail.items()}
            report['per_sequence'][b]['metrics'][label]['safety'] = {
                'FrameDegradation': _finite(
                    ea[b][q[0]].mean() - er[b][q[0]].mean() if q.any() else None
                )
            }
            for j in range(T):
                qj = full_support[b:b + 1, j:j + 1]
                report['per_frame'][b * T + j]['metrics'][label] = {name: detail(pred[b:b + 1, j:j + 1],
                    tt[b:b + 1, j:j + 1],
                    qj,
                    1) for (name,
                    pred) in methods_for_detail.items()}
                report['per_frame'][b * T + j]['metrics'][label]['safety'] = {'FrameDegradation': _finite(ea[b,
                    j][qj[0, 0]].mean() - er[b,
                    j][qj[0, 0]].mean() if qj.any() else None)}
        state.spatial.pixel.update(ss.pixel)
        state.safety.pixel.update(st.pixel)
        state.safety.frame.update(st.frame)
    if raw_disparity is not None:
        add('disparity_px', raw_disparity, refined_disparity, gt_disparity, 'px')
    if raw_depth is not None:
        add('depth_mm',
            _depth_mm(raw_depth, depth_input_unit),
            _depth_mm(refined_depth, depth_input_unit),
            _depth_mm(gt_depth, depth_input_unit),
            'mm',
            True,
            depth_input_unit)
    if raw_depth is None and raw_disparity is not None and (config.fx_px is not None):
        add('depth_mm',
            disparity_to_depth(raw_disparity, config.fx_px, config.baseline_mm),
            disparity_to_depth(refined_disparity, config.fx_px, config.baseline_mm),
            disparity_to_depth(gt_disparity, config.fx_px, config.baseline_mm),
            'mm',
            True)
    json.dumps(report, allow_nan=False)
    return (report, state) if return_state else report

def paired_bootstrap_ci(baseline: Mapping[Any, float],
    candidate: Mapping[Any, float],
    *,
    n_resamples: int=10000,
    confidence: float=0.95,
    seed: int=0,
    lower_is_better: bool=True) -> dict[str,
    Any]:
    """Return a deterministic CI for finite candidate-minus-baseline sequence values."""
    if not baseline or set(baseline) != set(candidate):
        raise ValueError('baseline/candidate need identical non-empty IDs')
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError('invalid bootstrap parameters')
    ids = sorted(baseline, key=str)
    d = np.asarray([candidate[i] - baseline[i] for i in ids], float)
    if not np.isfinite(d).all():
        raise ValueError('bootstrap values must be finite')
    mean = float(d.mean())
    base = {'sequence_count': len(d),
        'difference_candidate_minus_baseline': mean,
        'confidence': float(confidence),
        'lower_is_better': bool(lower_is_better),
        'improvement': bool(mean < 0 if lower_is_better else mean > 0)}
    if len(d) < 2:
        return {**base, 'ci_lower': None, 'ci_upper': None, 'reason': 'at least two sequences required'}
    rng = np.random.default_rng(seed)
    samples = d[rng.integers(0, len(d), (n_resamples, len(d)))].mean(1)
    q = (1 - confidence) / 2
    return {**base,
        'ci_lower': float(np.quantile(samples, q)),
        'ci_upper': float(np.quantile(samples, 1 - q)),
        'reason': None}

def _self_test() -> dict[str, str]:
    """Fourteen executable public-contract checks; only unavailable warp backends skip."""
    ok = {}
    c = MetricConfig(invalid_penalty_px=99.0, temporal_horizons=(1,))
    gt = np.full((2, 3, 1, 2), 10.0)
    m = np.ones_like(gt, bool)
    spatial = compute_spatial_metrics(gt, gt, m, m, c)
    temporal = compute_temporal_metrics(gt, gt, m, m, c, unit='px')
    assert (
        spatial['disparity_px']['prediction']['MAE']['value'] == 0
        and temporal['1']['methods']['raw']['DTCE_grid_px']['MAE']['value'] == 0
    )
    ok['perfect_spatial_temporal'] = 'pass'
    bias = compute_temporal_metrics(gt + 2, gt, m, m, c, unit='px')
    flicker = gt.copy()
    flicker[:, 1] += 5
    flicker_metrics = compute_temporal_metrics(flicker, gt, m, m, c, unit='px')
    assert (
        bias['1']['methods']['raw']['DTCE_grid_px']['MAE']['value'] == 0
        and flicker_metrics['1']['methods']['raw']['DTCE_grid_px']['MAE']['value'] == 5
    )
    ok['constant_bias_and_flicker'] = 'pass'
    proposal_raw = gt.copy()
    proposal_refined = gt.copy()
    proposal_refined[0, 0, 0, 0] += 2
    proposal_refined[0, 1, 0, 0] += 2
    proposal_raw[1, 0, 0, 0] += 2
    safety = compute_refinement_safety(proposal_raw, proposal_refined, gt, m, m, c)
    threshold = safety['thresholds']['1.0']
    assert (
        threshold['NewBad']['value'] > 0
        and threshold['RecoveredBad']['value'] > 0
        and (threshold['BadDelta']['value'] == threshold['NewBad']['value'] - threshold['RecoveredBad']['value'])
    )
    ok['delta_bad_identity'] = 'pass'
    assert (
        safety['HPlus']['value'] > 0
        and safety['BPlus']['value'] > 0
        and (safety['MAEDelta']['value'] == safety['HPlus']['value'] - safety['BPlus']['value'])
    )
    ok['delta_mae_identity'] = 'pass'
    invalid = gt.copy()
    invalid[0, 0, 0, 0] = np.nan
    invalid[1, 1, 0, 1] = np.inf
    invalid_spatial = compute_spatial_metrics(invalid, gt, m, m, c)
    assert (
        invalid_spatial['support_count'] == int(m.sum())
        and invalid_spatial['aggregate']['disparity_px']['MAE']['support_count'] == int(m.sum())
    )
    ok['support_count_invariant'] = 'pass'
    empty = compute_selective_metrics(np.full_like(gt, 7.0), gt, gt, gt, np.zeros_like(m), m, c)
    assert (
        empty['support_count'] == 0 and empty['risk_coverage_curve'] == [] and all((empty[k] is None for k in ('AURC',
            'AUROC',
            'Brier',
            'NLL',
            'ECE',
            'F1')))
    )
    ok['empty_selective_support'] = 'pass'
    uneven = np.full((2, 2, 1, 11), 10.0)
    uneven[0, 1, 0, 0] = 20
    uneven_mask = np.ones_like(uneven, bool)
    uneven_mask[1, :, 0, 1:] = False
    aggregate = compute_temporal_metrics(uneven,
        np.full_like(uneven, 10.0),
        uneven_mask,
        uneven_mask,
        c,
        unit='px')['1']['methods']['raw']['aggregate']['DTCE_grid_px']['MAE']
    assert (
        aggregate['macro_sequence'] != aggregate['micro_pixel']
        and aggregate['support_count'] == 12
        and (aggregate['frame_pair_count'] == 2)
    )
    ok['temporal_macro_not_micro'] = 'pass'
    score = np.array([[[[-4.0, 9.0]]]])
    rr = np.full_like(score, 10.0)
    aa = np.array([[[[10.0, 12.0]]]])
    selective = compute_selective_metrics(score, rr, aa, rr, np.ones_like(rr), np.ones_like(rr), c)
    assert selective['AURC'] is not None and selective['Brier'] is None and (selective['AUROC'] == 1)
    ok['non_probability_risk_score'] = 'pass'
    gated = compute_refinement_safety(gt, gt + 2, gt, m, m, c, gate=np.zeros_like(m))
    assert (
        gated['HPlus']['value'] == 2
        and gated['output_vs_raw']['HPlus']['value'] == 0
        and (gated['gate']['Coverage']['value'] == 0)
    )
    ok['gate_effect'] = 'pass'
    raw = np.array([[[[11.0]]], [[[13.0]]]])
    target = np.full_like(raw, 10.0)
    one = np.ones_like(raw, bool)
    hib = compute_refinement_safety(raw, target, target, one, one, c)['aggregate']
    depth = compute_spatial_metrics(np.array([[[[10.0]]], [[[20.0]]]]),
        np.full((2, 1, 1, 1), 10.0),
        np.ones((2, 1, 1, 1), bool),
        np.ones((2, 1, 1, 1), bool),
        c,
        unit='depth_mm')['aggregate']['depth_mm']['Delta1']
    assert (
        hib['BPlus']['higher_is_better']
        and hib['BPlus']['worst_sequence'] == 1
        and depth['higher_is_better']
        and (depth['worst_sequence'] == 0)
    )
    ok['higher_is_better_worst'] = 'pass'
    depth_mm = np.full((1, 2, 1, 1), 1000.0)
    depth_m = depth_mm / 1000.0
    dm = np.ones_like(depth_mm, bool)
    mm = compute_spatial_metrics(depth_mm, depth_mm, dm, dm, c, unit='depth_mm')
    metres = compute_spatial_metrics(depth_m, depth_m, dm, dm, c, unit='depth_mm', depth_input_unit='m')
    high_mm = evaluate_argos_prediction(raw_depth=depth_mm,
        refined_depth=depth_mm,
        gt_depth=depth_mm,
        gt_valid=dm,
        protocol_mask=dm,
        config=c)
    high_m = evaluate_argos_prediction(raw_depth=depth_m,
        refined_depth=depth_m,
        gt_depth=depth_m,
        gt_valid=dm,
        protocol_mask=dm,
        config=c,
        depth_input_unit='m')
    pre_m = {1: {'warp_valid': dm[:, :1],
        'warped_gt_valid_tk': dm[:, :1],
        'warped_protocol_mask_tk': dm[:, :1],
        'warped_raw_tk': depth_m[:, :1] + 0.1,
        'warped_target_tk': depth_m[:, :1],
        'is_gt': True}}
    pre_mm = {1: {'warp_valid': dm[:, :1],
        'warped_gt_valid_tk': dm[:, :1],
        'warped_protocol_mask_tk': dm[:, :1],
        'warped_raw_tk': depth_mm[:, :1] + 100.0,
        'warped_target_tk': depth_mm[:, :1],
        'is_gt': True}}
    temporal_m = compute_temporal_metrics(depth_m,
        depth_m,
        dm,
        dm,
        c,
        unit='mm',
        depth_input_unit='m',
        prewarped=pre_m)
    temporal_mm = compute_temporal_metrics(depth_mm, depth_mm, dm, dm, c, unit='mm', prewarped=pre_mm)
    assert (
        mm['depth_mm'] == metres['depth_mm']
        and high_mm['spatial'] == high_m['spatial']
        and (
            temporal_m['1']['methods']['raw']['W_DTCE_mm']['MAE']['value']
            == temporal_mm['1']['methods']['raw']['W_DTCE_mm']['MAE']['value']
            == 100
        )
    )
    ok['depth_m_mm_equivalence'] = 'pass'
    report = evaluate_argos_prediction(raw_disparity=gt,
        refined_disparity=gt,
        gt_disparity=gt,
        gt_valid=m,
        protocol_mask=m,
        config=c,
        risk_score=np.full_like(gt, 4.0),
        harm_probability=np.full_like(gt, 0.5))
    invalid_target = gt.copy()
    invalid_target[0, 0, 0, 0] = np.nan
    for (fn,
        args) in ((compute_refinement_safety, (gt, gt, invalid_target, m, m, c)),
        (compute_temporal_metrics, (gt, invalid_target, m, m, c)),
        (compute_selective_metrics, (np.full_like(gt, 4.0), gt, gt, invalid_target, m, m, c))):
        try:
            fn(*args, unit='px')
        except ValueError:
            pass
        else:
            raise AssertionError('invalid target accepted on support')
    json.dumps(report, allow_nan=False)
    ok['strict_json'] = 'pass'
    if torch is None:
        try:
            warp_with_flow(np.ones((1, 1)), np.zeros((2, 1, 1)))
        except RuntimeError:
            pass
        else:
            raise AssertionError('warp requires torch')
        ok['warp_cpu'] = 'skip: torch absent'
        ok['cuda_parity'] = 'skip: torch absent'
    else:
        x = torch.arange(6.0, dtype=torch.float32).reshape(2, 3)
        (y, inside) = warp_with_flow(x, torch.zeros(2, 2, 3))
        assert torch.equal(x, y) and inside.all()
        ok['warp_cpu'] = 'pass'
        if torch.cuda.is_available():
            (yc, _) = warp_with_flow(x.cuda(), torch.zeros(2, 2, 3, device='cuda'))
            assert yc.is_cuda and torch.equal(yc.cpu(), y)
            ok['cuda_parity'] = 'pass'
        else:
            ok['cuda_parity'] = 'skip: cuda absent'
    assert (
        len(ok) == 14 and all((value == 'pass' or (name in {'warp_cpu',
            'cuda_parity'} and value.startswith('skip:')) for (name,
            value) in ok.items()))
    )
    return ok
if __name__ == '__main__':
    print(_self_test())
