"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch as th

from models.text_to_tokens.flow_matching import CFM
from models.text_to_tokens.text_encoder import TextEncoder
from models.text_to_tokens.utils import (
    denormalize,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)


class TextToTokens(th.nn.Module):
    def __init__(
        self,
        n_vocab,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_feats = n_feats

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
        )

        if data_statistics is None:
            data_statistics = {
                "mean": 0.0,
                "std": 1.0,
            }
        self.register_buffer("mean", th.tensor(data_statistics["mean"]))
        self.register_buffer("std", th.tensor(data_statistics["std"]))

    @th.inference_mode()
    def synthesize(self, x, x_lengths, n_timesteps, temperature=1.0, length_scale=1.0):
        """
        Generates tokens from text.
        Args:
            x (th.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (th.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths)

        w = th.exp(logw) * x_mask
        w_ceil = th.ceil(w) * length_scale
        y_lengths = th.clamp_min(th.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = th.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        tokens = denormalize(decoder_outputs, self.mean, self.std).transpose(1, 2)
        return tokens
