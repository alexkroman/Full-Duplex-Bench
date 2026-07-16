"""Smooth turn-taking task inference for Freeze-Omni.

Streams each CANDOR turn-taking input.wav through the pipeline, holding off
generation until the annotated turn end (via llm_prefill's is_last_chunk
flag), and records the time-aligned model output. Shared machinery lives in
freeze_omni_common.py.
"""

import json
import math
import os
import threading
import time
from copy import deepcopy
from datetime import datetime
from glob import glob

import numpy as np
import soundfile as sf
import torch
import torchaudio

import freeze_omni_common as fo
from freeze_omni_common import llm_prefill

###### hyperparameters setting ######
configs = fo.build_configs()

sid = "1"  # random session id

output_path = "./evaluation/exp_results/smooth_turn_taking"
root_file_dir = "./evaluation/data/candor_turn_taking/candor_turn_taking/*/input.wav"

####################################

fo.init_pools(configs)
fo.register_user(sid)
connected_users = fo.connected_users


def send_pcm(sid):
    """
    Sends PCM audio data to the dialogue system for processing.

    Parameters:
    - sid (str): The session ID of the user.
    """

    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()
    pad_duration = 8

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wav_files = sorted(glob(root_file_dir))

    # iterate through all the wav files
    for input_wav in wav_files:

        wav, fs = sf.read(input_wav.strip())

        file_name = input_wav.split("/")[-2]

        print("Processing: ", input_wav)

        wav = torch.tensor(wav)

        if fs != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(
                wav.float()
            )
            fs = 16000

        json_path = os.path.join(os.path.dirname(input_wav), "turn_taking.json")

        with open(json_path, "r") as f:
            turn_data = json.load(f)
        # get the first timestamp
        last_time_sec = turn_data[0]["timestamp"][0]
        last_chunk_count = math.ceil((last_time_sec * fs) / chunk_size)

        original_chunk_count = last_chunk_count

        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[: wav.shape[0]] = wav

        chunked_inputs = []
        for i in range(0, wav_input.shape[0], chunk_size):
            chunked_inputs.append(wav_input[i : i + chunk_size])

        entire_output_audio = None
        time_aligned_output_audio = None

        # # save the concat_wav as audio file
        sf.write(f"candor_turn_temp.wav", wav_input.numpy(), 16000)

        cnt = 0
        idx = 0

        while True:
            if connected_users[sid][1].stop_pcm:
                print("Sid: ", sid, " Stop pcm")
                connected_users[sid][1].stop_generate = True
                connected_users[sid][1].stop_tts = True
                break

            if cnt >= len(chunked_inputs):
                print("Sid: ", sid, " Finish pcm")
                break

            is_last_chunk = cnt >= original_chunk_count - 1

            time.sleep(0.16)
            # Get current date and time
            current_time = datetime.now()
            print("Real Time: ", current_time.strftime("%H:%M:%S"))

            e = chunked_inputs[cnt]
            cnt += 1
            print("Sid: ", sid, " Time: ", cnt * chunk_size / fs)

            res = connected_users[sid][1].wakeup_and_vad.predict(np.float32(e))
            print(res["status"])

            force_tts_over = False

            if res["status"] == "sl":
                print("Sid: ", sid, " Vad start")
                force_tts_over = True

                outputs = deepcopy(connected_users[sid][1].generate_outputs)
                outputs["adapter_cache"] = None
                outputs["encoder_cache"] = None
                outputs["pe_index"] = 0
                outputs["stat"] = "sl"
                outputs["last_id"] = None
                if "text" in outputs:
                    del outputs["text"]
                if "hidden_state" in outputs:
                    del outputs["hidden_state"]

                for i in range(len(res["feature_last_chunk"])):
                    send_dict = {}
                    send_dict["status"] = "sl" if i == 0 else "cl"
                    send_dict["feature"] = res["feature_last_chunk"][i]
                    outputs = llm_prefill(
                        send_dict, outputs, sid, is_first_pack=True, is_last_chunk=False
                    )
                send_dict = {"status": "cl", "feature": res["feature"]}
                outputs = llm_prefill(
                    send_dict, outputs, sid, is_last_chunk=is_last_chunk
                )

            elif res["status"] in ["cl", "el"]:
                send_dict = {"status": res["status"], "feature": res["feature"]}
                outputs = llm_prefill(
                    send_dict, outputs, sid, is_last_chunk=is_last_chunk
                )

            final_output_audio = None
            if not connected_users[sid][1].tts_data.is_empty():
                output_data = connected_users[sid][1].tts_data.get()

                final_output_audio = output_data.astype(np.float32) / 32768.0
                print(final_output_audio.shape)

                if final_output_audio is not None:
                    if connected_users[sid][1].tts_over_time > 0:
                        connected_users[sid][1].tts_over_time = 0

                    if entire_output_audio is None:
                        entire_output_audio = final_output_audio
                    else:
                        entire_output_audio = np.concatenate(
                            (entire_output_audio, final_output_audio)
                        )

            curr_chunk_output = None

            if force_tts_over:
                curr_chunk_output = np.zeros(3840)
                entire_output_audio = None
            else:
                if (
                    entire_output_audio is not None
                    and idx < len(entire_output_audio) // 3840
                ):
                    curr_chunk_output = entire_output_audio[
                        idx * 3840 : (idx + 1) * 3840
                    ]
                    idx += 1

                else:
                    curr_chunk_output = np.zeros(3840)

            if time_aligned_output_audio is None:
                time_aligned_output_audio = curr_chunk_output
            else:
                time_aligned_output_audio = np.concatenate(
                    (time_aligned_output_audio, curr_chunk_output)
                )

        # read the input audio file
        input_audio, fs = sf.read("candor_turn_temp.wav")
        # resample to 24000 Hz
        input_audio = torchaudio.transforms.Resample(orig_freq=fs, new_freq=24000)(
            torch.tensor(input_audio).float()
        )

        # save the input audio and output audio as two-channel audio file
        # if the length of input audio and output audio are not equal, pad the shorter one with zeros
        if input_audio.shape[0] > time_aligned_output_audio.shape[0]:
            time_aligned_output_audio = np.concatenate(
                (
                    time_aligned_output_audio,
                    np.zeros(input_audio.shape[0] - time_aligned_output_audio.shape[0]),
                )
            )
        elif input_audio.shape[0] < time_aligned_output_audio.shape[0]:
            input_audio = np.concatenate(
                (
                    input_audio,
                    np.zeros(time_aligned_output_audio.shape[0] - input_audio.shape[0]),
                )
            )

        if not os.path.exists(os.path.join(output_path, file_name)):
            os.makedirs(os.path.join(output_path, file_name))

        # resample all the audio to 16000 Hz
        time_aligned_output_audio = torchaudio.transforms.Resample(
            orig_freq=24000, new_freq=16000
        )(torch.tensor(time_aligned_output_audio).float())
        input_audio = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(
            torch.tensor(input_audio).float()
        )

        # save the input audio file
        sf.write(os.path.join(output_path, file_name, "input.wav"), input_audio, 16000)
        # save the output audio file
        sf.write(
            os.path.join(output_path, file_name, "output.wav"),
            time_aligned_output_audio,
            16000,
        )

        # save the two-channel audio file
        sf.write(
            os.path.join(output_path, file_name, "two_channel.wav"),
            np.stack([input_audio, time_aligned_output_audio], axis=1),
            16000,
        )

        connected_users[sid][1].interrupt()
        connected_users[sid][1].reset()
        # connected_users[sid][1].wakeup_and_vad.reset_vad()
        # connected_users[sid][1].wakeup_and_vad.in_dialog = True


if __name__ == "__main__":
    print("Start Freeze-Omni sever")
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
