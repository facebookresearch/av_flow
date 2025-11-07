"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import torch as th

from utils.rotation import R_to_qvec, qvec_to_R


class HeadposeTransform(th.nn.Module):
    def __init__(
        self,
        head_translation_mean_file: str,
        head_translation_std_file: str,
        head_qvec_mean_file: str,
        head_qvec_std_file: str,
    ):
        super().__init__()
        head_t_mean = th.from_numpy(np.load(head_translation_mean_file)).float()
        head_t_std = th.from_numpy(np.load(head_translation_std_file)).float()
        head_qvec_mean = th.from_numpy(np.load(head_qvec_mean_file)).float()
        head_qvec_std = th.from_numpy(np.load(head_qvec_std_file)).float()

        self.register_buffer("head_t_mean", head_t_mean)
        self.register_buffer("head_t_std", head_t_std)
        self.register_buffer("head_qvec_mean", head_qvec_mean)
        self.register_buffer("head_qvec_std", head_qvec_std)

    def headRt2headvec(self, head_R, head_t):
        head_qvec = (R_to_qvec(head_R) - self.head_qvec_mean.to(head_R.device)) / self.head_qvec_std.to(head_R.device)
        head_t = (head_t - self.head_t_mean.to(head_t.device)) / self.head_t_std.to(head_t.device)
        headvec = th.cat([head_qvec, head_t], dim=-1)
        return headvec
    
    def headvec2headRt(self, headvec):
        head_qvec, head_t = headvec.split([4, 3], dim=-1)
        head_qvec = head_qvec * self.head_qvec_std.to(headvec.device) + self.head_qvec_mean.to(headvec.device)
        head_R = qvec_to_R(head_qvec)
        head_t = head_t * self.head_t_std.to(headvec.device) + self.head_t_mean.to(headvec.device)
        return head_R, head_t


class InfiniteBatchSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        seed: int = None,
    ):
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        self.rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        while True:
            batch_indices = self.rng.choice(self.indices, size=(self.batch_size,), replace=False)
            yield batch_indices


class InfiniteDataloader(th.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers: int = 0,
        seed: int = None,
    ):
        sampler = InfiniteBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
        )
    
        super().__init__(
            dataset=dataset,
            batch_sampler=sampler,
            prefetch_factor=10 if num_workers > 0 else None,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
