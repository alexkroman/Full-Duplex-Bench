"""Audio metric backends for eval_general_before_after.py.

Model singletons (SQuIM, UTMOSv2), sudden-cutoff detection, and robust
multi-backend pitch / intensity statistics
(Parselmouth → Librosa → Torchaudio/manual fallbacks).
"""

import os
import tempfile
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import utmosv2

# Optional deps (lazy)
_parselmouth = None
_librosa = None


def _lazy_import_parselmouth():
    global _parselmouth
    if _parselmouth is not None:
        return _parselmouth
    try:
        import parselmouth  # praat-parselmouth

        _parselmouth = parselmouth
    except Exception:
        _parselmouth = False
    return _parselmouth


def _lazy_import_librosa():
    global _librosa
    if _librosa is not None:
        return _librosa
    try:
        import librosa

        _librosa = librosa
    except Exception:
        _librosa = False
    return _librosa


# -------------------------------------------------------------
# Global model singletons
# -------------------------------------------------------------
_subjective_model = None  # not currently used
_objective_model = None
_utmosv2_model = None


def _load_models():
    global _subjective_model, _objective_model, _utmosv2_model
    if _subjective_model is None:
        _subjective_model = SQUIM_SUBJECTIVE.get_model()
    if _objective_model is None:
        _objective_model = SQUIM_OBJECTIVE.get_model()
    if _utmosv2_model is None:
        if torch.cuda.is_available():
            _utmosv2_model = utmosv2.create_model(pretrained=True, device="cuda")
        else:
            _utmosv2_model = utmosv2.create_model(pretrained=True)


# =============================================================
# Constants for sudden cutoff
# =============================================================
FRAME_MS = 30  # per-frame length (ms)
CUTOFF_DB = -18  # drop threshold dB between adjacent frames
MARGIN_DB = 6  # prev frame must be >= noise_floor + margin to count
HIST_FRAMES = 30
EPS = 1e-10


def _rms_db(block: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(block**2)) + EPS)


def detect_sudden_cutoffs(
    waveform: torch.Tensor,
    sr: int,
    frame_ms: int = FRAME_MS,
    cutoff_db: float = CUTOFF_DB,
    margin_db: float = MARGIN_DB,
    hist_frames: int = HIST_FRAMES,
) -> List[float]:
    """Return list of times (s) where a sudden level drop is detected."""
    if waveform.ndim == 2:
        wav = waveform.mean(dim=0).cpu().numpy()
    else:
        wav = waveform.squeeze().cpu().numpy()

    hop = int(sr * frame_ms / 1_000)
    if hop <= 0:
        return []

    history = deque(maxlen=hist_frames)
    prev_db = _rms_db(wav[:hop]) if len(wav) >= hop else _rms_db(wav)
    times_s: List[float] = []

    for i in range(1, max(1, len(wav) // hop)):
        frame = wav[i * hop : (i + 1) * hop]
        if frame.size == 0:
            break
        cur_db = _rms_db(frame)
        history.append(cur_db)

        diff_db = cur_db - prev_db
        noise_floor = np.percentile(history, 10) if history else cur_db

        if diff_db <= cutoff_db and prev_db >= noise_floor + margin_db:
            times_s.append(i * frame_ms / 1_000)

        prev_db = cur_db

    return times_s


# =============================================================
# Metric helpers (SQuIM, UTMOS)
# =============================================================


def _run_squim_objective(wav: torch.Tensor) -> Dict[str, float]:
    _load_models()
    stoi, pesq, si_sdr = _objective_model(wav)
    return {
        "stoi": float(stoi.item()),
        "pesq": float(pesq.item()),
        "si_sdr": float(si_sdr.item()),
    }


def _run_utmosv2(wav: torch.Tensor, sr: int) -> float:
    _load_models()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
        torchaudio.save(tmpf.name, wav, sr)
        mos = _utmosv2_model.predict(input_path=tmpf.name)
    try:
        os.unlink(tmpf.name)
    except OSError:
        pass
    return float(mos)


# =============================================================
# Robust Pitch helpers (Parselmouth → Librosa → Torchaudio → 0)
# =============================================================


def _pitch_parselmouth(
    wav_np: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    time_step: float,
):
    pm = _lazy_import_parselmouth()
    if not pm:
        return None, None
    try:
        snd = pm.Sound(wav_np, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=fmin, pitch_ceiling=fmax)
        f0 = pitch.selected_array["frequency"]  # Hz
        times = pitch.xs()
        return f0.astype(np.float32), times.astype(np.float32)
    except Exception:
        return None, None


def _pitch_librosa(
    wav_np: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    frame_time: float,
):
    lb = _lazy_import_librosa()
    if not lb:
        return None, None
    try:
        hop_length = max(1, int(round(sr * frame_time)))
        f0, voiced_flag, voiced_probs = lb.pyin(
            wav_np.astype(np.float32),
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )
        times = lb.times_like(f0, sr=sr, hop_length=hop_length)
        return f0.astype(np.float32), times.astype(np.float32)
    except Exception:
        return None, None


def _pitch_torchaudio(
    wav_t: torch.Tensor,
    sr: int,
    frame_time: float,
    fmin: float,
    fmax: float,
):
    try:
        if wav_t.ndim == 2:
            wav_t = wav_t.mean(dim=0)
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        f0 = (
            torchaudio.functional.detect_pitch_frequency(
                wav_t,
                sample_rate=sr,
                frame_time=frame_time,
                freq_low=fmin,
                freq_high=fmax,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        hop = int(round(sr * frame_time))
        n = f0.shape[0]
        times = np.arange(n, dtype=np.float32) * (hop / sr)
        return f0.astype(np.float32), times
    except Exception:
        return None, None


def _compute_pitch_stats_robust(
    wav: torch.Tensor,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 600.0,
    frame_time: float = 0.01,
    voiced_floor_hz: float | None = None,
):
    """
    Robust multi-backend pitch stats (Praat -> librosa.pyin -> torchaudio YIN).
    Returns (mean_Hz, std_Hz) over **voiced frames only**.
    """
    if wav.ndim == 2:
        x = wav.mean(dim=0).cpu().numpy()
    else:
        x = wav.squeeze().cpu().numpy()

    f0, _ = _pitch_parselmouth(x, sr, fmin, fmax, frame_time)
    if f0 is None:
        f0, _ = _pitch_librosa(x, sr, fmin, fmax, frame_time)
    if f0 is None:
        f0, _ = _pitch_torchaudio(torch.from_numpy(x), sr, frame_time, fmin, fmax)

    if f0 is None:
        return 0.0, 0.0

    vf = ~np.isnan(f0)
    vf &= f0 > 0
    if voiced_floor_hz is not None:
        vf &= f0 >= voiced_floor_hz
    vf &= f0 <= fmax

    voiced = f0[vf]
    if voiced.size == 0:
        return 0.0, 0.0
    if voiced.size == 1:
        return float(voiced[0]), 0.0

    return float(np.mean(voiced)), float(np.std(voiced, ddof=1))


# =============================================================
# Robust Intensity helpers (Parselmouth → Librosa → Manual)
# =============================================================


def _intensity_parselmouth(
    wav_np: np.ndarray,
    sr: int,
    time_step: float,
):
    pm = _lazy_import_parselmouth()
    if not pm:
        return None
    try:
        snd = pm.Sound(wav_np, sampling_frequency=sr)
        intensity = snd.to_intensity(time_step=time_step)
        vals = intensity.values.T.flatten()
        vals = vals[vals > -200]  # Praat silence floor
        if vals.size == 0:
            return None
        return vals.astype(np.float32)
    except Exception:
        return None


def _intensity_librosa(
    wav_np: np.ndarray,
    sr: int,
    frame_time: float,
):
    lb = _lazy_import_librosa()
    if not lb:
        return None
    try:
        hop = max(1, int(round(sr * frame_time)))
        frame_length = hop
        rms = lb.feature.rms(
            y=wav_np.astype(np.float32),
            frame_length=frame_length,
            hop_length=hop,
            center=False,
        ).squeeze(0)
        rms = np.maximum(rms, 1e-10)
        db = 20.0 * np.log10(rms)
        return db.astype(np.float32)
    except Exception:
        return None


def _intensity_manual(
    wav_np: np.ndarray,
    sr: int,
    frame_time: float,
):
    frame_len = max(1, int(round(sr * frame_time)))
    hop = frame_len
    n = wav_np.shape[0]
    vals = []
    for s in range(0, n, hop):
        e = min(n, s + frame_len)
        frame = wav_np[s:e]
        if frame.size == 0:
            continue
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        vals.append(20.0 * np.log10(rms))
    if not vals:
        return None
    return np.array(vals, dtype=np.float32)


def _compute_intensity_stats_robust(
    wav: torch.Tensor,
    sr: int,
    frame_time: float = 0.01,
):
    if wav.ndim == 2:
        x = wav.mean(dim=0).cpu().numpy()
    else:
        x = wav.squeeze().cpu().numpy()

    vals = _intensity_parselmouth(x, sr, frame_time)
    if vals is None:
        vals = _intensity_librosa(x, sr, frame_time)
    if vals is None:
        vals = _intensity_manual(x, sr, frame_time)

    if vals is None or vals.size == 0:
        return 0.0, 0.0
    if vals.size == 1:
        return float(vals[0]), 0.0

    return float(np.mean(vals)), float(np.std(vals, ddof=1))
