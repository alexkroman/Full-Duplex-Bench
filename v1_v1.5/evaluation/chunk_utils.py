"""ASR chunk utilities (word-aligned splitting, speech stats, WPM) for
eval_general_before_after.py."""

from typing import Any, Dict, List, Optional, Tuple


def _norm_ts(ch: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    ts = ch.get("timestamp")
    if ts is None:
        ts = ch.get("timestamps")
    if not ts or len(ts) < 2:
        return None
    return float(ts[0]), float(ts[1])


def _choose_split_time_word_aligned(
    distractor_end: float, chunks: List[Dict[str, Any]]
) -> float:
    if not chunks:
        return distractor_end
    normed = []
    for ch in chunks:
        ts_pair = _norm_ts(ch)
        if ts_pair is None:
            continue
        normed.append(ts_pair)
    if not normed:
        return distractor_end
    normed.sort(key=lambda x: x[0])
    last_end = normed[-1][1]
    for s, e in normed:
        if distractor_end < s:
            return s
        if s <= distractor_end < e:
            return s
    return last_end


def _partition_chunks_word_aligned(
    chunks: List[Dict[str, Any]], split_t: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pre, post = [], []
    for ch in chunks:
        ts_pair = _norm_ts(ch)
        if ts_pair is None:
            continue
        s, e = ts_pair
        if e <= split_t:
            pre.append(ch)
        else:
            post.append(ch)
    return pre, post


def _speech_stats(chunks: List[Dict[str, Any]]) -> Tuple[float, int]:
    speech = 0.0
    n = 0

    if len(chunks) == 0:
        return speech, n

    start_speech, _ = _norm_ts(chunks[0])
    _, end_speech = _norm_ts(chunks[-1])

    if end_speech - start_speech >= 0:
        speech = end_speech - start_speech

    n = len(chunks)

    return speech, n


def _wpm_speech_only(chunks: List[Dict[str, Any]]) -> float:
    speech_s, n = _speech_stats(chunks)
    if speech_s <= 0:
        return 0.0
    return float(n / (speech_s / 60.0))
