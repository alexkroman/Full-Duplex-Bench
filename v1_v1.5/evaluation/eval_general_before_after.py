"""Eval outputs split at distractor_end (word-aligned + robust metrics).

Orchestrates per-segment and per-directory evaluation plus the CLI.
Supporting pieces live in sibling modules:
  - audio_utils.py    : audio loading / slicing / VAD trim
  - audio_metrics.py  : SQuIM, UTMOSv2, sudden cutoff, pitch, intensity
  - chunk_utils.py    : word-aligned chunk splitting / WPM
  - aggregation.py    : robust outlier filtering + averaging
"""

import os
import json
import math
import argparse
from typing import Dict, List, Any, Optional

from audio_utils import _load_audio, _mix_mono, _slice_wave, _trim_dispatch
from audio_metrics import (
    detect_sudden_cutoffs,
    _run_squim_objective,
    _run_utmosv2,
    _compute_pitch_stats_robust,
    _compute_intensity_stats_robust,
)
from chunk_utils import (
    _choose_split_time_word_aligned,
    _partition_chunks_word_aligned,
    _speech_stats,
    _wpm_speech_only,
)
from aggregation import _aggregate_results


# =============================================================
# Segment evaluation (core)
# =============================================================


def _eval_segment(
    wav,
    sr: int,
    chunks: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if wav is None or wav.numel() == 0:
        return out

    # trim silence
    wav_t, speech_ts = _trim_dispatch(wav, sr, config)
    if wav_t.numel() == 0:
        return out

    # SQuIM
    if config.get("squim", False):
        try:
            out.update(_run_squim_objective(_mix_mono(wav_t)))
        except Exception as e:
            print(f"[WARN] SQuIM failed: {e}")
            out.update(
                {"stoi": float("nan"), "pesq": float("nan"), "si_sdr": float("nan")}
            )

    # UTMOS
    if config.get("utmosv2", False):
        try:
            out["utmosv2"] = _run_utmosv2(_mix_mono(wav_t), sr)
        except Exception as e:
            print(f"[WARN] UTMOSv2 failed: {e}")
            out["utmosv2"] = float("nan")

    # WPM
    if config.get("speaking_rate", False) and chunks is not None:
        out["wpm"] = _wpm_speech_only(chunks)
        speech_s, _ = _speech_stats(chunks)
        out["speech_dur_s"] = float(speech_s)

    # Sudden cutoff
    if config.get("sudden_cutoff", False):
        cut_times = detect_sudden_cutoffs(_mix_mono(wav_t), sr)
        out["cutoff_count"] = float(len(cut_times))

    # Pitch / Intensity (robust, no-NaN)
    if config.get("pitch", False):
        pp = config.get("pitch_params", {})
        mp, sp = _compute_pitch_stats_robust(
            wav_t,
            sr,
            fmin=pp.get("freq_low", 50.0),
            fmax=pp.get("freq_high", 600.0),
            frame_time=pp.get("frame_time", 0.01),
            voiced_floor_hz=pp.get("voiced_floor_hz", None),
        )
        out["mean_pitch"] = mp
        out["std_pitch"] = sp

    if config.get("intensity", False):
        ip = config.get("intensity_params", {})
        mi, si = _compute_intensity_stats_robust(
            wav_t,
            sr,
            frame_time=ip.get("frame_time", 0.01),
        )
        out["mean_intensity"] = mi
        out["std_intensity"] = si

    return out


# =============================================================
# File-level evaluation (split aware, word aligned)
# =============================================================


def _safe_dict(val):
    """保證回傳 dict；遇到 None / 非 dict 統一轉成 {}。"""
    return val if isinstance(val, dict) else {}


def eval_general_split(
    config: Dict[str, Any],
    wav_path: str,
    output_json_path: str,
    metadata_path: str,
    sr: int = 16000,
) -> Dict[str, Any]:

    prev_json_path = os.path.join(os.path.dirname(wav_path), "general_split.json")
    existing: Dict[str, Any] = {}
    if os.path.isfile(prev_json_path):
        with open(prev_json_path) as f:
            try:
                with open(prev_json_path) as f:
                    loaded = json.load(f)
                    existing = _safe_dict(loaded)

            except Exception:
                existing = {}

    waveform, sr = _load_audio(wav_path, target_sr=sr)
    total_samps = waveform.shape[-1]
    total_dur_s = total_samps / sr

    # metadata
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    distractor_end = float(meta["timestamps"][1])

    # chunks
    with open(output_json_path, "r") as f:
        out_json = json.load(f)
    chunks = out_json.get("chunks", [])

    # choose split
    split_t = _choose_split_time_word_aligned(distractor_end, chunks)
    split_t = max(0.0, min(total_dur_s, split_t))

    # audio split
    pre_wav = _slice_wave(waveform, sr, 0.0, split_t)
    post_wav = _slice_wave(waveform, sr, split_t, total_dur_s)

    # chunk split
    pre_chunks, post_chunks = _partition_chunks_word_aligned(chunks, split_t)

    existing_pre = _safe_dict(existing.get("pre"))
    existing_post = _safe_dict(existing.get("post"))

    pre_conf = _prune_config(existing_pre, config)
    post_conf = _prune_config(existing_post, config)

    pre_out = _eval_segment(pre_wav, sr, pre_chunks, pre_conf)
    post_out = _eval_segment(post_wav, sr, post_chunks, post_conf)

    merged_pre = {**_safe_dict(existing.get("pre")), **pre_out}
    merged_post = {**_safe_dict(existing.get("post")), **post_out}

    clean_out = _safe_dict(existing.get("clean"))

    clean_wav = wav_path.replace("output.wav", "clean_output.wav")
    clean_js = output_json_path.replace("output.json", "clean_output.json")
    if os.path.isfile(clean_wav) and os.path.isfile(clean_js):
        with open(clean_js) as f:
            clean_chunks = json.load(f).get("chunks", [])
            wav_c, _ = _load_audio(clean_wav, target_sr=sr)
            clean_conf = _prune_config(clean_out, config)
            new_clean = _eval_segment(wav_c, sr, clean_chunks, clean_conf)

        clean_out = {**clean_out, **new_clean}

    result = {
        "pre": merged_pre,
        "post": merged_post,
        "split_t": float(split_t),
        "distractor_end": float(distractor_end),
        "pre_dur_s": float(split_t),
        "post_dur_s": float(max(0.0, total_dur_s - split_t)),
    }
    if clean_out:
        result["clean"] = clean_out

    print("Evaluated split results:")
    print(existing)
    print("---")
    print(result)
    return result


# =============================================================
# Directory evaluation helper
# =============================================================


def _collect_example_roots(data_dir: str) -> List[str]:
    roots = []
    for root, dirs, files in os.walk(data_dir):
        if {"output.json", "metadata.json", "output.wav"}.issubset(files):
            roots.append(root)
    return roots


_METRIC_KEYS = {
    "squim": ["stoi", "pesq", "si_sdr"],
    "utmosv2": ["utmosv2"],
    "speaking_rate": ["wpm", "speech_dur_s"],
    "sudden_cutoff": ["cutoff_count"],
    "pitch": ["mean_pitch", "std_pitch"],
    "intensity": ["mean_intensity", "std_intensity"],
}


def _metric_complete(seg: dict, keys: list[str]) -> bool:
    for k in keys:
        v = seg.get(k)
        if v is None:
            return False
        if isinstance(v, float) and math.isnan(v):
            return False
    return True


def _prune_config(seg: dict, conf: dict) -> dict:
    new_conf = dict(conf)
    for m, keys in _METRIC_KEYS.items():
        if m == "speaking_rate":
            print("[INFO] WPM will always be recalculated.")
            continue
        if conf.get(m, False) and _metric_complete(seg, keys):
            new_conf[m] = False
    return new_conf


def eval_general_all_split(
    config: Dict[str, Any],
    data_dir: str,
    aggregate: bool = False,
) -> Any:
    paths = _collect_example_roots(data_dir)
    results = []
    for p in paths:
        wav_path = os.path.join(p, "output.wav")
        out_json_path = os.path.join(p, "output.json")
        meta_path = os.path.join(p, "metadata.json")
        print(f"Evaluating {p} ...")
        out = eval_general_split(config, wav_path, out_json_path, meta_path)
        # save per-example
        out_file = os.path.join(p, "general_split.json")
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
        results.append({"path": p, **out})

    if not results:
        return [] if not aggregate else {"pre": {}, "post": {}}

    if not aggregate:
        return results

    # aggregated
    return _aggregate_results(results, config)


# =============================================================
# CLI
# =============================================================


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Eval outputs split at distractor_end (word-aligned + robust metrics)"
    )
    ap.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Single wav to score (use with --output_json & --meta)",
    )
    ap.add_argument("--output_json", type=str, default=None, help="Path to output.json")
    ap.add_argument("--meta", type=str, default=None, help="Path to metadata.json")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Batch mode: directory with subfolders",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Average results across files in batch mode",
    )

    # metric toggles
    ap.add_argument("--squim", action="store_true")
    ap.add_argument("--utmosv2", action="store_true")
    ap.add_argument("--speaking_rate", action="store_true")
    ap.add_argument("--sudden_cutoff", action="store_true")
    ap.add_argument("--pitch", action="store_true")
    ap.add_argument("--intensity", action="store_true")

    # pitch/intensity params
    ap.add_argument("--pitch_frame_time", type=float, default=0.01)
    ap.add_argument("--pitch_fmin", type=float, default=50.0)
    ap.add_argument("--pitch_fmax", type=float, default=600.0)
    ap.add_argument("--intensity_frame_time", type=float, default=0.01)

    # trim mode flags
    ap.add_argument(
        "--trim_mode",
        type=str,
        default="torchaudio",
        choices=["none", "torchaudio", "silero"],
        help="Silence trim strategy",
    )
    ap.add_argument("--silero_threshold", type=float, default=0.5)
    ap.add_argument("--silero_min_speech_ms", type=int, default=60)
    ap.add_argument("--silero_min_silence_ms", type=int, default=50)
    ap.add_argument("--silero_window_size", type=int, default=-1, help="-1=default")
    ap.add_argument(
        "--silero_collapse",
        type=str,
        default="concat",
        choices=["trim_edges", "concat"],
    )

    # aggregation / outlier
    ap.add_argument(
        "--agg_mode",
        type=str,
        default="none",
        choices=["none", "iqr", "mad", "zscore", "winsor", "trim"],
        help="Outlier filter mode for aggregation",
    )
    ap.add_argument("--agg_iqr_k", type=float, default=1.5)
    ap.add_argument("--agg_mad_k", type=float, default=3.5)
    ap.add_argument("--agg_z_thresh", type=float, default=3.0)
    ap.add_argument("--agg_winsor_lo", type=float, default=0.05)
    ap.add_argument("--agg_winsor_hi", type=float, default=0.05)
    ap.add_argument("--agg_trim_prop", type=float, default=0.05)
    ap.add_argument("--agg_min_n", type=int, default=3)

    return ap.parse_args()


def _args_to_config(args) -> Dict[str, Any]:
    return {
        "squim": args.squim,
        "utmosv2": args.utmosv2,
        "speaking_rate": args.speaking_rate,
        "sudden_cutoff": args.sudden_cutoff,
        "pitch": args.pitch,
        "intensity": args.intensity,
        "pitch_params": {
            "frame_time": args.pitch_frame_time,
            "freq_low": args.pitch_fmin,
            "freq_high": args.pitch_fmax,
        },
        "intensity_params": {
            "frame_time": args.intensity_frame_time,
        },
        "trim_mode": args.trim_mode,
        "vad_kwargs": {},  # for torchaudio
        "silero_vad": {
            "threshold": args.silero_threshold,
            "min_speech_ms": args.silero_min_speech_ms,
            "min_silence_ms": args.silero_min_silence_ms,
            "window_size_samples": (
                None if args.silero_window_size < 0 else args.silero_window_size
            ),
            "collapse": args.silero_collapse,
        },
        "agg": {
            "mode": args.agg_mode,
            "iqr_k": args.agg_iqr_k,
            "mad_k": args.agg_mad_k,
            "z_thresh": args.agg_z_thresh,
            "winsor_limits": (args.agg_winsor_lo, args.agg_winsor_hi),
            "trim_prop": args.agg_trim_prop,
            "min_n": args.agg_min_n,
        },
    }


def main():
    args = _parse_args()
    config = _args_to_config(args)

    if args.data_dir:
        res = eval_general_all_split(config, args.data_dir, aggregate=args.aggregate)
        print(json.dumps(res, indent=2))
        return

    if not (args.wav and args.output_json and args.meta):
        raise SystemExit(
            "Single-file mode requires --wav, --output_json, --meta OR use --data_dir"
        )

    res = eval_general_split(config, args.wav, args.output_json, args.meta)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
