#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_helpers.py

Helpers for the FDB-v3 benchmark pipeline (run_tool_benchmark.py):
ASR, audio latency measurement, LiveKit inference, and search verification.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


# ==============================================================================
# ASR
# ==============================================================================

def load_asr_model():
    """Load NeMo ASR model."""
    print("🔊 Loading ASR model...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    if hasattr(model, 'cuda'):
        model = model.cuda()
    print("✅ ASR model loaded")
    return model


def run_asr(asr_model, audio_path):
    """Run ASR on an audio file and return transcript."""
    try:
        outputs = asr_model.transcribe([str(audio_path)], timestamps=True)
        if not outputs:
            return {"text": "", "chunks": []}

        result = outputs[0]
        chunks = []
        text = ""

        if hasattr(result, "timestamp") and "word" in result.timestamp:
            for w in result.timestamp["word"]:
                text += w["word"] + " "
                chunks.append({
                    "text": w["word"],
                    "timestamp": [w["start"], w["end"]],
                })
        else:
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, str):
                text = result

        return {"text": text.strip(), "chunks": chunks}
    except Exception as e:
        print(f"  ❌ ASR error: {e}")
        return {"text": "", "chunks": [], "error": str(e)}


# ==============================================================================
# Latency Measurement
# ==============================================================================

def measure_latency_from_audio(input_path, output_path, silence_threshold_db=-40):
    """Measure response latency:
    Time from end of input audio to first non-silence in output audio.
    Returns latency in seconds.
    """
    try:
        from pydub import AudioSegment

        input_audio = AudioSegment.from_file(str(input_path))
        output_audio = AudioSegment.from_file(str(output_path))

        input_duration_s = len(input_audio) / 1000.0
        output_duration_s = len(output_audio) / 1000.0

        # Find first non-silent chunk in output (check every 50ms)
        chunk_ms = 50
        first_speech_ms = None
        for i in range(0, len(output_audio), chunk_ms):
            chunk = output_audio[i:i + chunk_ms]
            if chunk.dBFS > silence_threshold_db:
                first_speech_ms = i
                break

        if first_speech_ms is not None:
            first_speech_s = first_speech_ms / 1000.0
        else:
            first_speech_s = output_duration_s  # No speech detected

        return {
            "input_duration_s": round(input_duration_s, 3),
            "output_duration_s": round(output_duration_s, 3),
            "first_speech_s": round(first_speech_s, 3),
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# LiveKit Inference
# ==============================================================================

def run_livekit_inference(input_path, output_path, provider):
    """
    Stream input audio into a LiveKit room and record the agent's response
    by calling livekit_inference.py as a subprocess.
    """
    import uuid
    import subprocess

    room_name = f"eval-{uuid.uuid4().hex[:8]}"
    print(f"  🔗 Streaming via livekit_inference.py into room: {room_name}")

    client_script = PROJECT_ROOT / "livekit_inference.py"

    try:
        # Run the client script as a subprocess
        result = subprocess.run(
            [
                sys.executable, str(client_script),
                "-i", str(input_path),
                "-o", str(output_path),
                "--room", room_name
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✅ livekit_inference.py finished successfully.")

        # Parse STREAM_START_TIME
        stream_start_time = None
        for line in result.stdout.splitlines():
            if line.startswith("STREAM_START_TIME: "):
                try:
                    stream_start_time = float(line[19:])
                except:
                    pass
                break

        return room_name, stream_start_time
    except subprocess.CalledProcessError as e:
        print(f"  ❌ livekit_inference.py failed with exit code {e.returncode}")
        return None, None
    except Exception as e:
        print(f"  ❌ livekit_inference.py execution error: {e}")
        return None, None


# ==============================================================================
# Search Verification
# ==============================================================================

def verify_search_answer(item, transcript):
    """Verify a fast_search answer using evaluate_model_answers.py."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluate_model_answers import evaluate_single
        return evaluate_single(item, transcript)
    except Exception as e:
        return {"status": "error", "reason": str(e)}
