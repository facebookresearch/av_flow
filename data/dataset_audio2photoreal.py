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
import tqdm


class SequenceDataset:
    def __init__(
        self,
        segments: List[str],
        base_dir: str,
        segment_length: int = 600,
        audio_sample_rate: int = 48000,
        frame_rate: int = 30,
        augmented_audio: bool = False,
    ):
        self.segment_length = segment_length
        self.frame_rate = frame_rate
        self.audio_sample_rate = audio_sample_rate
        self.augmented_audio = augmented_audio
        self.base_dir = base_dir

        # Preload the data
        self.data = {}
        print("Loading data...")
        for segment in tqdm.tqdm(segments):
            missing_frames = np.load(f"{base_dir}/{segment}_missing_face_frames.npy")

            # Expression codes
            expr = np.load(f"{base_dir}/{segment}_face_expression.npy") # (N, 256)
            # Body pose
            body = np.load(f"{base_dir}/{segment}_body_pose.npy")

            # Handle missing or corrupted expression codes
            window = []
            for frame_idx in missing_frames:
                if len(window) == 0 or frame_idx - 1 == window[-1]:
                    window.append(frame_idx)
                else:
                    expr[window[0]: window[-1] + 1] = th.nn.functional.interpolate(th.Tensor([expr[window[0] - 1], expr[window[-1] + 1]])[None].permute(0, 2, 1), size=len(window), mode="linear").permute(0, 2, 1)[0]
                    window = [frame_idx]

            audio, sample_rate = ta.load(f"{base_dir}/{segment}_audio.wav")
            assert sample_rate == self.audio_sample_rate

            # ASR tokens
            tokens = np.load(f"{base_dir}/wav2vec2_fairseq_base_ls960_asr_ls960_vad100/{segment}.npy")  # ASR
            tokens = th.from_numpy(tokens).to(th.float32)
            token_res = tokens.shape[1] / expr.shape[0]

            self.data[segment] = {
                "expr_code": self.upsample(th.from_numpy(expr)),
                "body_pose": self.upsample(th.from_numpy(body)),
                "self_audio": audio[0],
                "other_audio": audio[1],
                "self_tokens": self.upsample(tokens[0], input_frame_rate=token_res*30),
                "other_tokens": self.upsample(tokens[1], input_frame_rate=token_res*30),
            }

        # Create a list of all possible segment start frames, where each entry is a tuple of (segment, idx)
        self.segment_start_frames = []
        for segment in self.data:
            self.segment_start_frames += [(segment, i) for i in range(0, self.data[segment]["expr_code"].shape[0] - segment_length)]
    
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
            "body_pose": self.data[segment]["body_pose"][frame_idx:frame_idx+self.segment_length],
            "self_audio": self.data[segment]["self_audio"][audio_start:audio_end],
            "other_audio": self.data[segment]["other_audio"][audio_start:audio_end],
            "self_tokens": self.data[segment]["self_tokens"][token_start:token_end],
            "other_tokens": self.data[segment]["other_tokens"][token_start:token_end],
            "frame_idx": frame_idx,
            "segment": segment,
        }

        return data
