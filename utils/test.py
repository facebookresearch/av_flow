"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import torch as th
import numpy as np


def generate(trainer, test_audio=None, test_text=None, test_tokens=None):
    if test_audio is None and test_text is None and test_tokens is None:
        print("Expected input tokens or audio or text.")
        return
    
    if test_audio is not None:
        # Get ASR tokens from audio
        from utils.extract_asr import extract_tokens
        assert os.path.exists(test_audio)
        seg_name = os.path.basename(test_audio)[:-4]
        tokens = extract_tokens(test_audio)
        tokens = upsample(tokens)
    elif test_text is not None:
        # Predict tokens from text
        from utils.text_to_tokens import get_tokens_from_text
        seg_name = test_text[:10].replace(" ", "_").lower()
        tokens = get_tokens_from_text(
            test_text,
            ckpt_path=f"{trainer.config.asset_dir}/../text_to_tokens.pth"
        )
    elif test_tokens is not None:
        # Load tokens from file
        seg_name = os.path.basename(test_tokens)[:-4]
        tokens = np.load(test_tokens)
        tokens = th.from_numpy(tokens).to(th.float32)
        tokens = upsample(tokens)
    
    trainer.artifacts_dir = trainer.artifacts_dir + "/results"
    os.makedirs(trainer.artifacts_dir + "/viz", exist_ok=True)

    segment_length = min(345 * 5, tokens.shape[1])
    tokens = tokens[:, :segment_length].cuda()
    conditions = {
        "self_tokens": tokens,
        "other_tokens": tokens,
    }

    trainer.generate(conditions, suffix=seg_name, segment_length=segment_length)


def upsample(tokens, scale_factor=86.13281230 / 49.99):
    tokens = tokens.permute(0, 2, 1).contiguous()
    tokens = th.nn.functional.interpolate(tokens, scale_factor=scale_factor, mode="linear")
    tokens = tokens.permute(0, 2, 1).contiguous()
    return tokens
