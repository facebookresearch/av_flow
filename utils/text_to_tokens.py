"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse

import torch as th
import numpy as np
from omegaconf import OmegaConf

from models.text_to_tokens.text import sequence_to_text, text_to_sequence, intersperse
from models.text_to_tokens.text_to_tokens import TextToTokens


def get_tokens_from_text(text, ckpt_path, steps=10, temperature=0.667, device="cuda"):
    text = text.strip()
    text_processed = process_text(text, device)

    encoder_config = OmegaConf.create(
        {
            'encoder_type': 'RoPE Encoder',
            'encoder_params': {
                'n_feats': 29, 'n_channels': 192, 'filter_channels': 768, 'filter_channels_dp': 256, 'n_heads': 2, 'n_layers': 6, 'kernel_size': 3, 'p_dropout': 0.1, 'spk_emb_dim': 64, 'n_spks': 1, 'prenet': True
            },
            'duration_predictor_params': {
                'filter_channels_dp': 256, 'kernel_size': 3, 'p_dropout': 0.1
            }
        }
    )
    decoder_config = OmegaConf.create(
        {
            'channels': [256, 256], 'dropout': 0.05, 'attention_head_dim': 64, 'n_blocks': 1, 'num_mid_blocks': 2, 'num_heads': 2, 'act_fn': 'snakebeta'
        }
    )
    cfm_config = OmegaConf.create({'name': 'CFM', 'solver': 'euler', 'sigma_min': 0.0001})
    data_statistics = OmegaConf.create({'mean': -2.8461, 'std': 4.5255})

    model = TextToTokens(
        n_vocab=178,
        n_feats=29,
        encoder=encoder_config,
        decoder=decoder_config,
        cfm=cfm_config,
        data_statistics=data_statistics,
    )
    model = model.to(device)
    ckpt = th.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)

    output = model.synthesize(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=steps,
        temperature=temperature,
    )
    return output


def process_text(text: str, device: th.device):
    print(f"Input text: {text}")
    x = th.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
        dtype=th.long,
        device=device,
    )[None]
    x_lengths = th.tensor([x.shape[-1]], dtype=th.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"Phonetized text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--text', type=str, help="Input text")
    parser.add_argument('-c', '--ckpt_path', type=str, help="Checkpoint path")
    parser.add_argument('-o', '--output_path', type=str, help="Output .npy file path")
    args = parser.parse_args()
    tokens = get_tokens_from_text(args.text, args.ckpt_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, tokens.detach().cpu().numpy())
