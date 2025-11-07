"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import argparse
import tqdm
import numpy as np
import glob

import torch as th
import torchaudio as ta

LABELS = ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')

bundle = ta.pipelines.WAV2VEC2_ASR_BASE_960H
wav2vec_asr_model = bundle.get_model().cuda()
# ckpt_wav2vec_asr = th.load("./wav2vec2_fairseq_base_ls960_asr_ls960.pth")
# wav2vec_asr_model = ta.models.wav2vec2_model(**ckpt_wav2vec_asr["params"]).cuda()
# wav2vec_asr_model.load_state_dict(ckpt_wav2vec_asr["model"])
wav2vec_asr_model.eval()


def extract_tokens(audio_path, step=16000 * 100):
    outputs = []
    with th.inference_mode():
        waveform, sample_rate = ta.load(audio_path)
        if sample_rate != 16000:
            waveform = ta.functional.resample(waveform, sample_rate, 16000)
        for idx in range(0, waveform.shape[-1], step):
            emissions, _ = wav2vec_asr_model(waveform[..., idx:idx+step].cuda())
            outputs.append(emissions)
        outputs = th.cat(outputs, dim=1)
    return outputs


def main(data_dir):
    segments = [seg.split("/")[-1][:-4] for seg in glob.glob(f"{data_dir}/*.wav")]

    os.makedirs(f"{data_dir}/wav2vec2_fairseq_base_ls960_asr_ls960/", exist_ok=True)

    for segment in tqdm.tqdm(segments):
        audio_path = os.path.join(data_dir, f"{segment}.wav")
        outputs = extract_tokens(audio_path=audio_path)        
        outputs = outputs.cpu().detach().numpy()
        np.save(f"{data_dir}/wav2vec2_fairseq_base_ls960_asr_ls960/{segment}.npy", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory path with the data")
    args = parser.parse_args()
    main(args.input_dir)
