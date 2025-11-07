"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple
import torch as th

from models.layers import TimeEmbedding, WindowedDiffusionTransformerBlock


class AVWindowedDiffusionTransformer(th.nn.Module):
    def __init__(
        self,
        visual_dim: int = 256,
        audio_dim: int = 256,
        condition_dim: int = 256,
        window_per_layer: Tuple[Tuple[int]] = ((16, 8), (16, 8), (16, 8), (16, 8)),
        d_model: int = 512,
        hidden_dim: int = 1024,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.time_embed = TimeEmbedding(d_model)
        self.input_proj_v = th.nn.Linear(visual_dim + condition_dim, d_model)
        self.input_proj_a = th.nn.Linear(audio_dim + condition_dim, d_model)
        self.transformer_blocks_v = th.nn.ModuleList(
            [WindowedDiffusionTransformerBlock(
                d_model, hidden_dim=hidden_dim, heads=heads, dropout=dropout, window_left=window_left, window_right=window_right
            ) for (window_left, window_right) in window_per_layer]
        )
        self.transformer_blocks_a = th.nn.ModuleList(
            [WindowedDiffusionTransformerBlock(
                d_model, hidden_dim=hidden_dim, heads=heads, dropout=dropout, window_left=window_left, window_right=window_right
            ) for (window_left, window_right) in window_per_layer]
        )
        self.fusion = th.nn.ModuleList(
            [th.nn.Linear(d_model * 2, d_model * 2) for _ in window_per_layer]
        )
        self.out_v = th.nn.Linear(d_model, visual_dim)
        self.out_a = th.nn.Linear(d_model, audio_dim)

    def forward(self, x_v, x_a, t):
        """
        :param x: B x T x input_dim noisy samples
        :param t: B-dimensional time step tensor in [0, 1]
        :return: B x T x output_dim denoised samples
        """
        t = self.time_embed(t)
        x_v = self.input_proj_v(x_v)
        x_a = self.input_proj_a(x_a)
        for block_v, block_a, fusion in zip(self.transformer_blocks_v, self.transformer_blocks_a, self.fusion):
            x_v = block_v(x_v, t)
            x_a = block_a(x_a, t)
            # Fusion
            x = th.cat([x_a, x_v], dim=-1)
            x = fusion(x)
            y_a, y_v = th.chunk(x, 2, dim=-1)
            x_a = x_a + y_a
            x_v = x_v + y_v
        x_v = self.out_v(x_v)
        x_a = self.out_a(x_a)
        return x_v, x_a


class AVDiffusionWrapper(th.nn.Module):
    def __init__(
        self,
        diffusion_module: th.nn.Module,
        participant_cond: bool = False,
    ):
        super().__init__()
        self.diffusion_module = diffusion_module
        self.participant_cond = participant_cond

    def forward(self, x, t, self_tokens, other_tokens, self_melspec=None, other_melspec=None, self_audio=None, other_audio=None,
                expr_code=None, headvec=None, body_pose=None, other_expr=None, other_head=None, get_audio=False):
        """
        :param x: B x T x input_dim noisy samples
        :param t: B-dimensional time step tensor in [0, 1]
        :param self_audio: B x n_samples of self audio, where n_samples is T * 1600
        :param other_audio: B x n_samples of other audio, where n_samples is T * 1600
        :return: B x T x input_dim denoised samples
        """
        ## 2 models with connections
        x_v, x_a = x.split([self.diffusion_module.visual_dim, self.diffusion_module.audio_dim], dim=-1)

        if self.participant_cond:
            # Dyadic conversations (actor - participant interaction)
            x_v = th.cat([x_v, self_tokens, other_tokens], dim=-1)
            x_a = th.cat([x_a, self_tokens, other_tokens], dim=-1)
            if other_expr is not None:
                x_v = th.cat([x_v, other_expr, other_head], dim=-1)
                x_a = th.cat([x_a, other_expr, other_head], dim=-1)
        else:
            # Default: use only actor's inputs
            x_v = th.cat([x_v, self_tokens], dim=-1)
            x_a = th.cat([x_a, self_tokens], dim=-1)

        x_v, x_a = self.diffusion_module(x_v, x_a, t)
        x = th.cat([x_v, x_a], dim=-1)
        
        return x
