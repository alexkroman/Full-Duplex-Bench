"""Robust aggregation helpers (outlier filtering + averaging) for
eval_general_before_after.py."""

import math
from typing import Any, Dict, List

import numpy as np


def _robust_filter_vals(vals: List[float], agg_cfg: Dict[str, Any]) -> List[float]:
    """Apply outlier filtering per agg_cfg; return filtered list (copy)."""
    mode = (agg_cfg or {}).get("mode", "none")
    vals = [float(v) for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return []
    if mode in (None, "none"):
        return vals
    min_n = agg_cfg.get("min_n", 3)
    if len(vals) < min_n:
        return vals

    if mode == "iqr":
        k = float(agg_cfg.get("iqr_k", 1.5))
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        return [v for v in vals if lo <= v <= hi]

    if mode == "mad":
        k = float(agg_cfg.get("mad_k", 3.5))
        med = np.median(vals)
        mad = np.median(np.abs(np.array(vals) - med))
        if mad <= 0:
            return vals
        # scaled MAD ~ std
        dev = np.abs(np.array(vals) - med) / (mad * 1.4826)
        return [v for v, d in zip(vals, dev) if d <= k]

    if mode == "zscore":
        zt = float(agg_cfg.get("z_thresh", 3.0))
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        if sd <= 0:
            return vals
        return [v for v in vals if abs((v - mu) / sd) <= zt]

    if mode == "winsor":
        lo_p, hi_p = agg_cfg.get("winsor_limits", (0.05, 0.05))
        lo = np.percentile(vals, lo_p * 100.0)
        hi = np.percentile(vals, 100.0 - hi_p * 100.0)
        return [min(max(v, lo), hi) for v in vals]

    if mode == "trim":
        p = float(agg_cfg.get("trim_prop", 0.05))
        lo = np.percentile(vals, p * 100.0)
        hi = np.percentile(vals, 100.0 - p * 100.0)
        return [v for v in vals if lo <= v <= hi]

    return vals


def _aggregate_results(
    results: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    agg_cfg = config.get("agg", {})

    def _agg_side(side: str) -> Dict[str, float]:
        keys = set()
        for r in results:
            keys.update(r.get(side, {}).keys())
        out = {}
        for k in keys:
            vals = []
            for r in results:
                v = r.get(side, {}).get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv):
                    continue
                vals.append(fv)
            clean = _robust_filter_vals(vals, agg_cfg)
            if clean:
                out[k] = float(sum(clean) / len(clean))
        return out

    return {"pre": _agg_side("pre"), "post": _agg_side("post")}
