"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch as th
import torchaudio as ta

from models.vocoder_utils import mel_spectrogram
import bigvgan


class MelSpectrogramforBigVGAN(th.nn.Module):
    def __init__(
        self,
        vocoder_ckpt_path: str,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 22050,
    ):
        super().__init__()

        if input_sample_rate == output_sample_rate:
            self.resampler = th.nn.Identity()
        else:
            self.resampler = ta.transforms.Resample(input_sample_rate, output_sample_rate)

        self.vocoder = bigvgan.BigVGAN.from_pretrained(vocoder_ckpt_path, use_cuda_kernel=False)

        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().cuda()

        self.audio_max_value = 1.1

        self.mel_mean = th.tensor(-5.5366).cuda()
        self.mel_std = th.tensor(2.1161).cuda()

    def forward(self, audio: th.Tensor):
        """
        :param audio: B x n_samples audio signal
        :return: B x T_melspec x 128 log mel spectrogram
        """
        audio = self.resampler(audio) / self.audio_max_value

        x = mel_spectrogram(audio, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)

        x = (x - self.mel_mean) / self.mel_std
        
        x = x.permute(0, 2, 1).contiguous()

        return x

    def reverse_to_audio(self, x: th.Tensor):
        x = x.permute(0, 2, 1).contiguous()

        x = x * self.mel_std + self.mel_mean

        audio_recon = self.vocoder(x)

        audio_recon = audio_recon.squeeze(1) * self.audio_max_value

        return audio_recon
