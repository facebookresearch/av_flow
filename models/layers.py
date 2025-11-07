"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Optional
import math
import torch as th


class RotaryPositionalEmbeddings(th.nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.
    Adapted from: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (th.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = th.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = th.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = th.stack([th.cos(idx_theta), th.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: th.Tensor, input_pos: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [bsz, seq_len, num_heads, head_dim]
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None
        Returns:
            Tensor: output tensor with RoPE applied
        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, n_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not. When
        # input_pos is provided, we're in inference mode
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, n_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [1, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, n_d // 2, 2]
        x_out = th.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, n_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class WindowedSelfAttention(th.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window_left: int = None,
        window_right: int = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d_head = dim // heads
        self.window_left = window_left
        self.window_right = window_right

        self.mask = th.zeros(0, 0)
        self.qkv = th.nn.Linear(dim, 3 * dim)
        self.rope = RotaryPositionalEmbeddings(self.d_head, max_seq_len=max_seq_len)
        self.project = th.nn.Linear(dim, dim)

    def create_mask(self, dim: int, device="cpu"):
        """
        Creates a dim x dim mask with 1s along the diagnal band such that each token sees only its window_left to window_right neighbors
        if window_left or window_right is None, use infinite window (no masking in left/right direction)
        """
        if not self.mask.shape[0] == dim:
            self.mask = th.ones(dim, dim, device=device)
            # left mask
            if self.window_left is not None:
                self.mask = 1.0 - th.tril(self.mask, diagonal=-self.window_left-1)
            # right mask
            if self.window_right is not None:
                self.mask = th.tril(self.mask, diagonal=self.window_right)
            # update buffer
            self.mask = th.log(self.mask)  # values are either 0 or -inf
        else:
            self.mask = self.mask.to(device)

    def forward(self, x: th.Tensor):
        """
        :param x: B x T x dim tensor containing input to windowed self-attention layer
        :return: B x T x dim tensor containing output of multi-head attention layer
        """
        B, T = x.shape[0], x.shape[1]

        # compute queries, keys, and values from input
        x = self.qkv(x).view(B, T, self.heads, self.d_head * 3)
        q, k, v = th.split(x, self.d_head, dim=-1)

        # rotary positional encoding
        q = self.rope(q.contiguous())
        k = self.rope(k.contiguous())

        # compute attention scores
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.contiguous().transpose(1, 2)
        scores = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # apply mask and compute softmax
        self.create_mask(T, device=x.device)
        scores = scores + self.mask[None, None]
        scores = th.softmax(scores, dim=-1)

        # weighted sum of values
        x = th.matmul(scores, v)

        # merge heads back into dimension
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, self.dim)
        x = self.project(x)

        return x


class PositionalEncoding(th.nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.0,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = th.exp(- th.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = th.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = th.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = th.sin(pos * den)
        pos_embedding[:, 1::2] = th.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = th.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TimeEmbedding(th.nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.mlp = th.nn.Sequential(
            th.nn.Linear(dim // 4, dim),
            th.nn.GELU(),
            th.nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        """
        :param t: B-dimensional tensor with time steps in [0, 1]
        :return: B x dim tensor containing time embedding
        """
        fourier_dim = self.dim // 8
        emb = math.log(10_000) / (fourier_dim - 1)
        emb = th.exp(th.arange(fourier_dim, device=t.device) * (-emb))
        emb = t[..., None] * emb[None, :]
        emb = th.cat([th.sin(emb), th.cos(emb)], dim=-1)
        emb = self.mlp(emb)
        return emb


class AdaLnZero(th.nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.linear = th.nn.Linear(dim, 3 * dim)
        self.linear.weight.data[0:dim, :] = 0
        self.linear.bias.data.zero_()
    
    def forward(self, x):
        """
        :param x: ... x dim
        :return: alpha, beta, gamma of shape ... x dim
        """
        x = self.linear(x)
        alpha, beta, gamma = x.split(self.dim, dim=-1)
        return alpha, beta, gamma


class DiffusionFeedforward(th.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.ada_ln = AdaLnZero(dim)
        self.norm = th.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.block = th.nn.Sequential(
            th.nn.Linear(dim, hidden_dim),
            th.nn.GELU(),
            th.nn.Dropout(p=dropout),
            th.nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x, t):
        """
        :param x: B x T x dim
        :param t: B x dim
        :return: B x T x dim
        """
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        alpha, beta, gamma = self.ada_ln(t)
        y = self.norm(x) * (1 + gamma) + beta
        y = self.block(y) * alpha
        return x + y


class WindowedDiffusionSelfAttention(th.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        window_left: int = None,
        window_right: int = None,
    ):
        super().__init__()
        self.ada_ln = AdaLnZero(dim)
        self.norm = th.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = WindowedSelfAttention(dim, heads, window_left=window_left, window_right=window_right)
        self.dropout = th.nn.Dropout(p=dropout)

    def forward(self, x, t):
        """
        :param x: B x T x dim
        :param t: B x dim
        :return: B x T x dim
        """
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        alpha, beta, gamma = self.ada_ln(t)
        y = self.norm(x) * (1 + gamma) + beta
        y = self.self_attn(y) * alpha
        y = self.dropout(y)
        return x + y


class WindowedDiffusionTransformerBlock(th.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        heads: int = 4,
        dropout: float = 0.0,
        window_left: int = None,
        window_right: int = None
    ):
        super().__init__()
        self.self_attn = WindowedDiffusionSelfAttention(dim, heads, dropout, window_left, window_right)
        self.feedforward = DiffusionFeedforward(dim, hidden_dim, dropout)

    def forward(self, x, t):
        """
        :param x: B x T x dim
        :param t: B x dim
        :return: B x T x dim
        """
        x = self.self_attn(x, t)
        x = self.feedforward(x, t)
        return x
