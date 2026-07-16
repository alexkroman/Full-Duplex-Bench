"""Audio loading, slicing, and silence-trim (VAD) helpers for
eval_general_before_after.py."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

# Silero VAD singletons
_silero_model = None
_silero_utils = None  # tuple of silero functions


def _load_silero(force_reload: bool = False):
    """Load Silero VAD model + utils via torch.hub (cached)."""
    global _silero_model, _silero_utils
    if _silero_model is not None and _silero_utils is not None and not force_reload:
        return _silero_model, _silero_utils
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=force_reload,
        onnx=False,
    )
    return _silero_model, _silero_utils


# =============================================================
# Audio helpers
# =============================================================


def _load_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr


def _mix_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _slice_wave(
    wav: torch.Tensor, sr: int, start_s: float, end_s: float
) -> torch.Tensor:
    n = wav.shape[-1]
    s_idx = max(0, min(n, int(round(start_s * sr))))
    e_idx = max(0, min(n, int(round(end_s * sr))))
    if e_idx <= s_idx:
        return wav[..., :0]
    return wav[..., s_idx:e_idx]


# =============================================================
# Silero VAD wrapper (returns trimmed_wav, speech_ts list)
# =============================================================


def _apply_silero_vad(
    waveform: torch.Tensor,
    sr: int,
    threshold: float = 0.5,
    min_speech_ms: int = 60,
    min_silence_ms: int = 50,
    window_size_samples: Optional[int] = None,
    collapse: str = "concat",
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    model, utils = _load_silero()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    target_sr = 16000
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        wf_mono = waveform.mean(dim=0, keepdim=True)
    else:
        wf_mono = waveform
    if sr != target_sr:
        wf_mono = torchaudio.functional.resample(wf_mono, sr, target_sr)
        sr_vad = target_sr
    else:
        sr_vad = sr

    wav_1d = wf_mono.squeeze(0).cpu()

    params = {
        "threshold": threshold,
        "min_speech_duration_ms": min_speech_ms,
        "min_silence_duration_ms": min_silence_ms,
    }
    if window_size_samples is not None:
        params["window_size_samples"] = int(window_size_samples)

    speech_ts = get_speech_timestamps(
        wav_1d,
        model,
        sampling_rate=sr_vad,
        **params,
    )

    if not speech_ts:
        dur_s = waveform.shape[-1] / sr
        return waveform, [(0.0, dur_s)]

    segs = [(seg["start"] / sr_vad, seg["end"] / sr_vad) for seg in speech_ts]

    if collapse == "trim_edges":
        s_s, e_s = segs[0][0], segs[-1][1]
        trimmed = _slice_wave(waveform, sr, s_s, e_s)
        return trimmed, [(s_s, e_s)]

    # concat speech-only
    parts = [_slice_wave(waveform, sr, s_s, e_s) for (s_s, e_s) in segs]
    trimmed = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    # still return original speech-ts (in original timeline) for reference
    return trimmed, segs


# =============================================================
# Torchaudio VAD wrapper (compat)
# =============================================================


def _apply_torchaudio_vad(
    waveform: torch.Tensor, sr: int, vad_kwargs: Optional[dict] = None
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    vad_kwargs = vad_kwargs or {}
    try:
        trimmed = torchaudio.functional.vad(waveform, sr, **vad_kwargs)
        if trimmed.numel() == 0:
            dur_s = waveform.shape[-1] / sr
            return waveform, [(0.0, dur_s)]
        dur_s = trimmed.shape[-1] / sr
        return trimmed, [(0.0, dur_s)]
    except Exception:
        dur_s = waveform.shape[-1] / sr
        return waveform, [(0.0, dur_s)]


# =============================================================
# Trim dispatch (ALWAYS returns (trimmed_wav, speech_ts))
# =============================================================


def _trim_dispatch(
    wav: torch.Tensor,
    sr: int,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    mode = config.get("trim_mode", "torchaudio")
    if mode == "none":
        dur_s = wav.shape[-1] / sr
        return wav, [(0.0, dur_s)]
    if mode == "torchaudio":
        return _apply_torchaudio_vad(_mix_mono(wav), sr, config.get("vad_kwargs", {}))
    if mode == "silero":
        sv = config.get("silero_vad", {})
        return _apply_silero_vad(
            wav,
            sr,
            threshold=sv.get("threshold", 0.5),
            min_speech_ms=sv.get("min_speech_ms", 60),
            min_silence_ms=sv.get("min_silence_ms", 50),
            window_size_samples=sv.get("window_size_samples", None),
            collapse=sv.get("collapse", "concat"),
        )
    # fallback
    dur_s = wav.shape[-1] / sr
    return wav, [(0.0, dur_s)]
