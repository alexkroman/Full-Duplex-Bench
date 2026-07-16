"""Pause handling task inference for Freeze-Omni.

Streams each CANDOR pause-handling input.wav through the pipeline and records
the time-aligned model output. Shared machinery lives in freeze_omni_common.py.
"""

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

root_file_dir = (
    "/./evaluation/data/candor_pause_handling/candor_pause_handling/*/input.wav"
)
output_path = "./evaluation/exp_results/pause_handling"

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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wav_files = sorted(glob(root_file_dir))

    # iterate through all the wav files
    for input_wav in wav_files:

        wav, fs = sf.read(input_wav)

        # file_name = input_wav.split("/")[-2]
        file_name = f"{input_wav.split('/')[4]}/{input_wav.split('/')[-2]}"

        print("Processing: ", input_wav)

        wav = torch.tensor(wav)
        if fs != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(
                wav.float()
            )
            fs = 16000

        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[: wav.shape[0]] = wav

        chunked_inputs = []
        for i in range(0, wav_input.shape[0], chunk_size):
            chunked_inputs.append(wav_input[i : i + chunk_size])

        entire_output_audio = None
        time_aligned_output_audio = None

        # # save the concat_wav as audio file
        sf.write(f"candor_pause_temp.wav", wav_input.numpy(), 16000)

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

            chunk_start_time = cnt * chunk_size / fs
            chunk_end_time = (cnt + 1) * chunk_size / fs

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

                send_dict = {}
                for i in range(len(res["feature_last_chunk"])):
                    if i == 0:
                        send_dict["status"] = "sl"
                    else:
                        send_dict["status"] = "cl"
                    send_dict["feature"] = res["feature_last_chunk"][i]
                    outputs = llm_prefill(send_dict, outputs, sid, is_first_pack=True)
                send_dict["status"] = "cl"
                send_dict["feature"] = res["feature"]
                outputs = llm_prefill(send_dict, outputs, sid)

            elif res["status"] == "cl" or res["status"] == "el":
                send_dict = {}
                send_dict["status"] = res["status"]
                send_dict["feature"] = res["feature"]
                outputs = llm_prefill(send_dict, outputs, sid)

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
        input_audio, fs = sf.read("candor_pause_temp.wav")
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

        # save the input audio file
        sf.write(os.path.join(output_path, file_name, "input.wav"), input_audio, 24000)
        # save the output audio file
        sf.write(
            os.path.join(output_path, file_name, "output.wav"),
            time_aligned_output_audio,
            24000,
        )

        # save the two-channel audio file
        sf.write(
            os.path.join(output_path, file_name, "two_channel.wav"),
            np.stack([input_audio, time_aligned_output_audio], axis=1),
            24000,
        )

        connected_users[sid][1].interrupt()
        connected_users[sid][1].reset()
        connected_users[sid][1].wakeup_and_vad.reset_vad()
        # connected_users[sid][1].wakeup_and_vad.in_dialog = True


if __name__ == "__main__":
    print("Start Freeze-Omni sever")
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
