"""Shared Freeze-Omni inference machinery.

The four task scripts (backchannel.py, pause_handling.py,
smooth_turn_taking.py, user_interruption.py) all drive the same
Freeze-Omni pipeline. This module holds the common pieces: model/pool
initialisation, the speech decoder loop, LLM prefill/generation, and the
per-session state container (MyGlobalParams). Each task script keeps only
its own audio streaming loop (send_pcm).
"""

import sys
import os

# Use current working directory as a fallback
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import argparse
import json
import threading
import time
from copy import deepcopy
from threading import Timer

import numpy as np
import torch

from web.pool import TTSObjectPool, pipelineObjectPool
from web.vad import VAD
from web.queue import PCMQueue, ThreadSafeQueue

TIMEOUT = 600
MAX_USERS = 1

# Populated by init_pools() / register_user()
connected_users = {}
tts_pool = None
pipeline_pool = None
server_configs = None


def build_configs():
    """Default Freeze-Omni hyperparameters shared by all task scripts."""
    # create empty configs for argument passing
    configs = {}
    configs["model_path"] = "./models/Freeze-Omni/checkpoints/checkpoints"
    configs["llm_path"] = "./models/Freeze-Omni/Qwen2-7B-Instruct"
    configs["top_p"] = 0.8
    configs["top_k"] = 20
    configs["temperature"] = 0.8
    configs["llm_exec_nums"] = 1
    return argparse.Namespace(**configs)


def init_pools(configs):
    """Create the inference pipeline / speech decoder pools and load server configs."""
    global tts_pool, pipeline_pool, server_configs
    # init inference pipelines pool
    pipeline_pool = pipelineObjectPool(size=configs.llm_exec_nums, configs=configs)
    # init speech decoder pool
    tts_pool = TTSObjectPool(size=MAX_USERS, model_path=configs.model_path)
    server_configs = json.loads(open(configs.model_path + "/server.json").read())


def register_user(sid, params=None):
    """Register a session: disconnect timer + per-session global params."""
    if params is None:
        params = MyGlobalParams(tts_pool, pipeline_pool)
    connected_users[sid] = [Timer(TIMEOUT, disconnect_user, [sid]), params]
    tts_pool.print_info()
    pipeline_pool.print_info()


def decoder(
    cur_hidden_state,
    cur_text,
    outputs,
    connected_users,
    sid,
    generate_num,
    last_text,
    is_last_chunk=False,
):
    """
    Decodes the current hidden state and text to generate audio segments using speech decoder.

    Parameters:
    - cur_hidden_state (list of torch.Tensor): The current hidden state of the language model.
    - cur_text (str): The current text to be synthesized.
    - connected_users (dict): A dictionary containing information about connected users.
    - sid (str): The session ID of the user.
    - is_last_chunk (bool, optional): Indicates if the current text is the last chunk of the input

    Returns:
    - int: The updated number of audio segments generated.
    """
    hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
    cur_text_procced = connected_users[sid][1].pipeline_obj.pipeline_proc.post_process(
        cur_text
    )
    print("Synthesis: ", [cur_text_procced])
    embeddings = connected_users[sid][
        1
    ].pipeline_obj.pipeline_proc.model.llm_decoder.model.embed_tokens(
        torch.tensor(
            connected_users[sid][1].pipeline_obj.pipeline_proc.model.tokenizer.encode(
                cur_text_procced
            )
        ).cuda()
    )
    codec_chunk_size = server_configs["decoder_first_chunk_size"]
    codec_padding_size = server_configs["decoder_chunk_overlap_size"]
    seg_threshold = server_configs["decoder_seg_threshold_first_pack"]
    if generate_num != 0:
        codec_chunk_size = server_configs["decoder_chunk_size"]
        seg_threshold = server_configs["decoder_seg_threshold"]
    for seg in connected_users[sid][1].tts_obj.tts_proc.run(
        embeddings.reshape(-1, 896).unsqueeze(0),
        server_configs["decoder_top_k"],
        hidden_state_output.reshape(-1, 896).unsqueeze(0),
        codec_chunk_size,
        codec_padding_size,
        server_configs["decoder_penalty_window_size"],
        server_configs["decoder_penalty"],
        server_configs["decoder_N"],
        seg_threshold,
    ):
        if generate_num == 0:
            try:
                split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                seg = seg[:, :, split_idx:]
            except:
                print("Do not need to split")
                pass
        generate_num += 1
        if connected_users[sid][1].tts_over:
            print("_________________________")
            print("tts_over")
            print("_________________________")
            connected_users[sid][1].tts_data.clear()
            connected_users[sid][1].whole_text = ""
            break
        connected_users[sid][1].tts_data.put(
            (seg.squeeze().float().cpu().numpy() * 32768).astype(np.int16)
        )
        print("Generate: ", generate_num)
    return generate_num


def generate(outputs, sid):
    """
    Generates speech dialogue output based on the current state and user session ID.

    Parameters:
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.

    Returns:
    - None
    """
    # Stage3: start speak
    connected_users[sid][1].is_generate = True

    outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
        None, **outputs
    )
    connected_users[sid][1].generate_outputs = deepcopy(outputs)

    cur_hidden_state = []
    cur_hidden_state.append(outputs["hidden_state"])

    connected_users[sid][1].whole_text = ""
    # Stage4: contiune speak until stat is set to 'sl'
    # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
    stop = False
    cur_text = ""
    last_text = ""
    generate_num = 0
    while True:
        if connected_users[sid][1].stop_generate:
            break
        if len(outputs["past_tokens"]) > 100:
            stop = True
        if stop:
            break
        del outputs["text"]
        del outputs["hidden_state"]
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
            None, **outputs
        )
        connected_users[sid][1].generate_outputs = deepcopy(outputs)
        if outputs["stat"] == "cs":
            cur_hidden_state.append(outputs["hidden_state"])
            if "�" in outputs["text"][len(last_text) :]:
                continue
            connected_users[sid][1].whole_text += outputs["text"][len(last_text) :]
            cur_text += outputs["text"][len(last_text) :]
            # print([connected_users[sid][1].whole_text])
            if generate_num == 0 or (len(cur_hidden_state) >= 20):
                suffix_list = [
                    ",",
                    "，",
                    "。",
                    "：",
                    "？",
                    "！",
                    ".",
                    ":",
                    "?",
                    "!",
                    "\n",
                ]
            else:
                suffix_list = ["。", "：", "？", "！", ".", "?", "!", "\n"]
            if outputs["text"][len(last_text) :].endswith(tuple(suffix_list)) and (
                len(cur_hidden_state) >= 4
            ):
                if (
                    outputs["text"][len(last_text) :].endswith(".")
                    and last_text[-1].isdigit()
                ):
                    pass
                else:
                    if not connected_users[sid][1].tts_over:
                        if len(cur_hidden_state) > 0:
                            generate_num = decoder(
                                cur_hidden_state,
                                cur_text,
                                outputs,
                                connected_users,
                                sid,
                                generate_num,
                                last_text,
                            )
                            cur_text = ""
                            cur_hidden_state = []
            last_text = outputs["text"]
        else:
            break
    if not connected_users[sid][1].tts_over:
        if len(cur_hidden_state) != 0:
            generate_num = decoder(
                cur_hidden_state,
                cur_text,
                outputs,
                connected_users,
                sid,
                generate_num,
                last_text,
                is_last_chunk=True,
            )
            cur_text = ""
    connected_users[sid][1].is_generate = False


def llm_prefill(data, outputs, sid, is_first_pack=False, is_last_chunk=True):
    """
    Prefills the LLM of speech dialogue system using speech.

    Parameters:
    - data (dict): A dictionary containing the current state of the user's input,
                   including features and status.
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.
    - is_first_pack (bool, optional): Indicates if the current input packet is the first one in a new conversation
    - is_last_chunk (bool, optional): Indicates if the current input packet is the last one
      in a conversation. When False, a detected endpoint ('ss') is ignored and listening
      continues (used by smooth_turn_taking); the default True triggers generation.
    """

    if data["status"] == "sl":
        # Satge1: start listen
        # stat will be auto set to 'cl' after Stage1
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data["feature"]), **outputs
        )

    if data["status"] == "el":
        connected_users[sid][1].wakeup_and_vad.in_dialog = False
        print("Sid: ", sid, " Detect vad time out")

    if data["status"] == "cl":
        if outputs["stat"] == "cl":
            # Stage2: continue listen
            # stat will be auto set to 'ss' when endpoint is detected
            outputs = connected_users[sid][
                1
            ].pipeline_obj.pipeline_proc.speech_dialogue(
                torch.tensor(data["feature"]), **outputs
            )

            print("predict stat:", outputs["stat"])
        if is_first_pack:
            outputs["stat"] = "cl"
        if outputs["stat"] == "el":
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            print("Sid: ", sid, " Detect invalid break")
        if outputs["stat"] == "ss":
            if is_last_chunk:
                connected_users[sid][1].interrupt()

                print("Sid: ", sid, " Detect break")
                connected_users[sid][1].wakeup_and_vad.in_dialog = False
                generate_thread = threading.Thread(
                    target=generate, args=(deepcopy(outputs), sid)
                )
                generate_thread.start()
            else:
                outputs["stat"] = "cl"
    return outputs


def disconnect_user(sid):
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to time out")
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]


class MyGlobalParams:
    def __init__(self, tts_pool, pipeline_pool):
        """
        Initialize the GlobalParams class with necessary components for managing global parameters and states.

        Parameters:
        - tts_pool: Pool of speech decoder.
        - pipeline_pool: Pool of inference pipeline.

        Returns:
        - None
        """
        self.tts_pool = tts_pool
        self.pipeline_pool = pipeline_pool

        self.tts_obj = self.tts_pool.acquire()
        self.pipeline_obj = self.pipeline_pool.acquire()
        # init default prompt
        init_outputs = self.pipeline_obj.pipeline_proc.speech_dialogue(
            None,
            stat="pre",
            role="You are a helpful voice assistant.\
                                                                             Your answer should be coherent, natural, simple, complete.\
                                                                             Do not answer too long.",
        )
        self.system_role = deepcopy(init_outputs)

        self.wakeup_and_vad = VAD()
        self.reset()

    def set_prompt(self, prompt):
        self.system_role = self.pipeline_obj.pipeline_proc.speech_dialogue(
            None, stat="pre", role=prompt
        )

    def reset(self):
        self.stop_generate = False
        self.is_generate = False
        self.wakeup_and_vad.in_dialog = False
        self.generate_outputs = deepcopy(self.system_role)
        self.whole_text = ""

        self.tts_over = False
        self.tts_over_time = 0
        self.tts_data = ThreadSafeQueue()

        self.stop_tts = False
        self.stop_pcm = False

    def interrupt(self, timeout=5.0):
        """
        Interrupt generation: wait for the generate thread to exit and the
        tts_data queue to drain, with a timeout guard against infinite waits.
        """
        self.stop_generate = True
        self.tts_over = True

        # Wait for the generation thread to finish, up to 'timeout' seconds
        start_time = time.time()
        while self.is_generate:
            if time.time() - start_time > timeout:
                print("Interrupt: Timeout waiting for is_generate to stop")
                break
            time.sleep(0.01)
        self.stop_generate = False

        # Wait for the tts_data queue to clear, up to 'timeout' seconds
        start_time = time.time()
        while not self.tts_data.is_empty():
            if time.time() - start_time > timeout:
                print("Interrupt: Timeout waiting for tts_data to clear")
                break
            time.sleep(0.01)

        self.whole_text = ""
        self.tts_over = False
        self.tts_over_time += 1

    def release(self):
        self.tts_pool.release(self.tts_obj)
        self.pipeline_pool.release(self.pipeline_obj)

    def print(self):
        print("stop_generate:", self.stop_generate)
        print("is_generate:", self.is_generate)
        print("whole_text:", self.whole_text)
        print("tts_over:", self.tts_over)
        print("tts_over_time:", self.tts_over_time)
