"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import argparse

import recipes.config as conf
from learn.flowmatching_audiovisual_trainer import AVFlowMatchingTrainer
from models.diffusion import AVWindowedDiffusionTransformer, AVDiffusionWrapper
from utils.test import generate
from utils.test_dyadic import test_dyadic


def main(data_dir, output_dir, load_ckpt_path=None, test_mode=False, test_text=None, test_audio=None, test_tokens=None):
    DEFAULT_CONFIG = {
        # base config
        "artifacts_dir": output_dir,
        "asset_dir": data_dir,
        # dataset name
        "dataset_name": "50h", # Audio2Photoreal or 50h
        "subject": "PXB184",  # subject for Audio2Photoreal - subjects: ["PXB184", "RLW104", "TXB805", "GQS883"]
        # input/output dimensions
        "face_dim": 256,  # face expression codes
        "headvec_dim": 8,  # latent space of VAE for head dynamics
        "melspec_dim": 80,  # mel-spectrogram
        "output_audio_sample_rate": 22050,  # 22050 for BigVGAN vocoder
        "token_dim": 29,  # character-level logits - output of ASR
        "other_dim": 0,  # other conditioning signals (e.g. participant's expressions)
        "participant_cond": False,  # dyadic conversations - if True, set "other_dim" to 58 + 29
        # dataset config
        "dataset": {
            "segment_length": 345,
            "frame_rate": 86.13281230,
            "body": False,  # True for Audio2Photoreal dataset
            "input_audio_sample_rate": 48000,
            "input_head_dim": 7,  # 7 for 50h dataset, 4 for Audio2Photoreal dataset
        },
        # training config
        "trainer": {
            "batch_size": 16,
            "val_batch_size": 1,
            "learning_rate": 0.0001,
            "num_iterations": 1_000_000,
            "num_workers": 10,
            "val_frequency": 5000,
            "save_frequency": 5000,
            "headpose_weight": 0.2,
            # Flow matching parameters
            "sigma_min": 1e-6,
            "nfe_steps": 16,
            # Loss parameters
            "loss_fn_vision": "l1",
            "loss_fn_audio": "l1",
            "audio_weight": 3.0,
            # Head VAE
            "head_vae_ckpt_path": f"{data_dir}/head_vae/iter-0500000.pt",
            # Vocoder
            "vocoder_ckpt_path": f"{data_dir}/../bigvgan_base_22khz_80band",
        },
    }
    DEFAULT_CONFIG.update(
        {
            "diffusion_module": {
                "module_name": "windowed_diffusion_transformer_audiovisual",
                "kwargs": {
                    "visual_dim": DEFAULT_CONFIG["face_dim"] + DEFAULT_CONFIG["headvec_dim"],
                    "audio_dim": DEFAULT_CONFIG["melspec_dim"],
                    "condition_dim": DEFAULT_CONFIG["token_dim"] + DEFAULT_CONFIG["other_dim"],
                    "d_model": 512,
                    "hidden_dim": 1024,
                    "window_per_layer": ((16, 8), (16, 8), (16, 8), (16, 8), (16, 8), (16, 8), (16, 8), (16, 8)),
                },
            },
        }
    )
    config = conf.Config(DEFAULT_CONFIG).get()
    conf.save(f"{config.artifacts_dir}/config.yaml", config)

    if test_audio is not None or test_text is not None or test_tokens is not None:
        # Inference
        test_mode = True
    if test_mode and load_ckpt_path is None:
        print("No checkpoint provided to load")
        return

    if config.dataset_name == "Audio2Photoreal":
        ## Audio2Photoreal data
        from data.dataset_audio2photoreal import SequenceDataset
        assert config.dataset.body == True, "Audio2Photoreal uses renderer for the full body"
        assert config.dataset.input_head_dim == 4
        segments = sorted([seg.split("/")[-1][:-4].split("_")[0] for seg in glob.glob(f"{config.asset_dir}/*.wav")])
        train_segments = segments[:-2]
        val_segments = segments[-2:]
    else:
        ## Dataset 50h
        from data.dataset import SequenceDataset
        assert config.dataset.body == False
        assert config.dataset.input_head_dim == 7
        segments = [seg.split("/")[-1][:-4] for seg in glob.glob(f"{config.asset_dir}/expression_codes/*.npy")]
        val_segments = [seg for seg in segments if "20240408" in seg or "20240509" in seg]
        train_segments = [seg for seg in segments if seg not in val_segments]
    
    if test_mode:
        train_segments = train_segments[:1]
        val_segments = [seg for seg in val_segments if "20240509--1052" in seg]

    print(f"Load train dataset with {len(train_segments)} segments...")
    train_dataset = SequenceDataset(
        segments=train_segments,
        base_dir=config.asset_dir,
        segment_length=config.dataset.segment_length,
        frame_rate=config.dataset.frame_rate,
    )
    print(f"Load val dataset with {len(val_segments)} segments...")
    val_dataset = SequenceDataset(
        segments=val_segments,
        base_dir=config.asset_dir,
        segment_length=config.dataset.segment_length,
        frame_rate=config.dataset.frame_rate,
    )

    diffusion_module = AVWindowedDiffusionTransformer(**config.diffusion_module.kwargs)
    model = AVDiffusionWrapper(diffusion_module, participant_cond=config.participant_cond)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = AVFlowMatchingTrainer(model, train_dataset, val_dataset, config)
    if load_ckpt_path is not None:
        trainer.load_checkpoint(load_ckpt_path)
    
    if test_mode:
        # Inference
        if not config.participant_cond:
            # Any input (audio or text or tokens)
            generate(trainer, test_audio=test_audio, test_text=test_text, test_tokens=test_tokens)
        else:
            # Dyadic conversations
            partipant_images_path = os.path.join(data_dir, "participant/camera01/images/")
            trainer.val_dataloader = iter(val_dataset)
            if not os.path.exists(partipant_images_path):
                print("No participant images found")
                return
            test_dyadic(trainer, img_path=partipant_images_path, segment_length=config.dataset.segment_length)
    else:
        # Training
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory path with the data")
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory path to save results")
    parser.add_argument('-c', '--ckpt_path', type=str, default=None, help="Checkpoint path to load")
    parser.add_argument('--test', default=False, action="store_true", help="Test mode")
    parser.add_argument('--text', type=str, default=None, help="Text for inference")
    parser.add_argument('--audio', type=str, default=None, help="Audio path for inference")
    parser.add_argument('--tokens', type=str, default=None, help="Tokens .npy path for inference")
    args = parser.parse_args()
    main(
        data_dir=args.input_dir,
        output_dir=args.output_dir,
        load_ckpt_path=args.ckpt_path,
        test_mode=args.test,
        test_tokens=args.tokens,
        test_audio=args.audio,
        test_text=args.text,
    )
