"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List
import numpy as np
import torch as th
import torchaudio as ta
import os
import tqdm


class SequenceDataset:
    def __init__(
        self,
        segments: List[str],
        base_dir: str,
        segment_length: int = 600,  # 20s per segment    # 1600 / 48000 * 600 = 20.0
        audio_sample_rate: int = 48000,
        frame_rate: int = 30,
    ):
        self.segment_length = segment_length
        self.frame_rate = frame_rate
        self.audio_sample_rate = audio_sample_rate
        self.base_dir = base_dir

        # Load statistics for participant's inputs
        self.other_expr_min = np.load(f"{base_dir}/participant/camera01/smirk_outputs/stats/expression_min.npy")
        self.other_expr_max = np.load(f"{base_dir}/participant/camera01/smirk_outputs/stats/expression_max.npy")
        self.other_head_min = np.load(f"{base_dir}/participant/camera01/smirk_outputs/stats/headpose_min.npy")
        self.other_head_max = np.load(f"{base_dir}/participant/camera01/smirk_outputs/stats/headpose_max.npy")

        # Preload the data
        self.data = {}
        print("Loading data...")
        for segment in tqdm.tqdm(segments):
            # Head pose
            headpose = np.load(f"{base_dir}/headpose/{segment}--headpose.npy")
            # Expression codes
            expr = np.load(f"{base_dir}/expression_codes/{segment}.npy")  # (N, 256)
            
            audio, sample_rate = ta.load(f"{base_dir}/audio/{segment}.wav")
            assert sample_rate == self.audio_sample_rate

            # ASR tokens
            tokens = np.load(f"{base_dir}/wav2vec2_fairseq_base_ls960_asr_ls960_vad100/{segment}.npy")  # (N, 29)
            tokens = th.from_numpy(tokens).to(th.float32)
            token_res = tokens.shape[1] / headpose.shape[0]

            # Load SMIRK features for conditioning on participant's information
            if os.path.exists(f"{base_dir}/participant/camera01/smirk_outputs/{segment}--expression.npy"):
                expr_participant = np.load(f"{base_dir}/participant/camera01/smirk_outputs/{segment}--expression.npy")
                head_participant = np.load(f"{base_dir}/participant/camera01/smirk_outputs/{segment}--headpose.npy")
                if (expr_participant == 0).all():
                    print("All zeros", segment)
                if expr_participant.shape[0] != headpose.shape[0]:
                    print("Different number of frames", segment)
            else:
                continue

            # Normalize
            expr_participant = -1 + (expr_participant - self.other_expr_min) * 2 / (self.other_expr_max - self.other_expr_min)
            head_participant = -1 + (head_participant - self.other_head_min) * 2 / (self.other_head_max - self.other_head_min)

            self.data[segment] = {
                "head_R": self.upsample(th.from_numpy(headpose[:, :3, :3])),
                "head_t": self.upsample(th.from_numpy(headpose[:, :3, 3])),
                "expr_code": self.upsample(th.from_numpy(expr)),
                "self_audio": audio[0],
                "other_audio": audio[1],
                "self_tokens": self.upsample(tokens[0], input_frame_rate=token_res*30),
                "other_tokens": self.upsample(tokens[1], input_frame_rate=token_res*30),
                "other_expr": self.upsample(th.from_numpy(expr_participant)),
                "other_head": self.upsample(th.from_numpy(head_participant)),
            }

        # Create a list of all possible segment start frames, where each entry is a tuple of (segment, idx)
        self.segment_start_frames = []
        for segment in self.data:
            self.segment_start_frames += [(segment, i) for i in range(0, self.data[segment]["head_t"].shape[0] - segment_length)]
    
    def __len__(self):
        return len(self.segment_start_frames)

    def upsample(self, x, input_frame_rate=30):
        if self.frame_rate != input_frame_rate:
            _squeeze = False
            if len(x.shape) == 2:
                x = x[:, None]
                _squeeze = True
            last_dim = len(x.shape) - 1
            x = x.transpose(0, last_dim)
            x = th.nn.functional.interpolate(x, scale_factor=self.frame_rate / input_frame_rate, mode="linear")
            x = x.transpose(0, last_dim)
            if _squeeze:
                x = x[:, 0]
        return x

    def __getitem__(self, idx):
        segment = self.segment_start_frames[idx][0]
        frame_idx = self.segment_start_frames[idx][1]

        token_res = 1

        token_start = int(token_res*frame_idx)
        token_end = int(token_res*(frame_idx+self.segment_length))

        audio_res = self.audio_sample_rate / self.frame_rate
        audio_start = int(audio_res * frame_idx)
        audio_end = int(audio_res * (frame_idx + self.segment_length))
        if audio_end - audio_start > int(self.audio_sample_rate / self.frame_rate * self.segment_length):
            audio_end -= 1
        elif audio_end - audio_start < int(self.audio_sample_rate / self.frame_rate * self.segment_length):
            audio_end += 1

        data = {
            "expr_code": self.data[segment]["expr_code"][frame_idx:frame_idx+self.segment_length],
            "head_R": self.data[segment]["head_R"][frame_idx:frame_idx+self.segment_length],
            "head_t": self.data[segment]["head_t"][frame_idx:frame_idx+self.segment_length],
            "self_audio": self.data[segment]["self_audio"][audio_start:audio_end],
            "other_audio": self.data[segment]["other_audio"][audio_start:audio_end],
            "self_tokens": self.data[segment]["self_tokens"][token_start:token_end],
            "other_tokens": self.data[segment]["other_tokens"][token_start:token_end],
            "other_expr": self.data[segment]["other_expr"][frame_idx:frame_idx+self.segment_length],
            "other_head": self.data[segment]["other_head"][frame_idx:frame_idx+self.segment_length],
            "frame_idx": frame_idx,
            "segment": segment,
        }

        return data
