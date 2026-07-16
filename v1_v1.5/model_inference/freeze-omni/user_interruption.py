"""User interruption task inference for Freeze-Omni.

Builds a context + silence + interruption input for each synthetic example,
streams it through the pipeline, and records the time-aligned model output.
Shared machinery lives in freeze_omni_common.py.
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

wait_time = 7
padding_time = 15
sid = "1"  # random session id
output_path = "./evaluation/exp_results/user_interruption"
root_file_dir = (
    "./evaluation/data/synthetic_user_interruption/synthetic_user_interruption/*"
)

####################################

fo.init_pools(configs)


class UserInterruptionParams(fo.MyGlobalParams):
    """Keeps this task's original interrupt() semantics: session state is only
    reset once tts_data actually drains; on timeout the state is left as-is."""

    def interrupt(self, timeout=5.0):
        self.stop_generate = True
        self.tts_over = True
        start_time = time.time()

        # wait for generation to stop
        while True:
            time.sleep(0.01)
            if not self.is_generate:
                break
            if time.time() - start_time > timeout:
                print(
                    "Warning: Generation did not stop within {} seconds.".format(
                        timeout
                    )
                )
                break

        self.stop_generate = False

        # wait for tts_data to be empty
        inner_start = time.time()
        while True:
            time.sleep(0.01)
            if self.tts_data.is_empty():
                self.whole_text = ""
                self.tts_over = False
                self.tts_over_time += 1
                break
            if time.time() - inner_start > timeout:
                print("Warning: tts_data not empty after {} seconds.".format(timeout))
                break


fo.register_user(sid, UserInterruptionParams(fo.tts_pool, fo.pipeline_pool))
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

    file_dirs = sorted(glob(root_file_dir))

    for file_dir in file_dirs:

        file_name = file_dir.split("/")[-1]
        print("File name: ", file_name)

        wav, fs = sf.read(os.path.join(file_dir, "context.wav"))

        wav = torch.tensor(wav)
        if fs != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(
                wav.float()
            )
            fs = 16000

        # interruption input
        interrupt_wav, fs = sf.read(os.path.join(file_dir, "interrupt.wav"))
        interrupt_wav = torch.tensor(interrupt_wav)
        if fs != 16000:
            interrupt_wav = torchaudio.transforms.Resample(
                orig_freq=fs, new_freq=16000
            )(interrupt_wav.float())
            fs = 16000

        # concat wav, silence, and the interrupt_wav as the new input

        concat_wav = torch.cat(
            [
                wav,
                torch.zeros(wait_time * fs),
                interrupt_wav,
                torch.zeros(padding_time * fs),
            ]
        )

        wav_input = torch.zeros(
            math.ceil(concat_wav.shape[0] / chunk_size) * chunk_size
        )
        wav_input[: concat_wav.shape[0]] = concat_wav

        chunked_inputs = []
        for i in range(0, wav_input.shape[0], chunk_size):
            chunked_inputs.append(wav_input[i : i + chunk_size])

        entire_output_audio = None
        time_aligned_output_audio = None

        # save the concat_wav as audio file
        sf.write(f"interrupt_temp.wav", wav_input.numpy(), 16000)

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
        input_audio, fs = sf.read("interrupt_temp.wav")
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


if __name__ == "__main__":
    print("Start Freeze-Omni sever")
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
