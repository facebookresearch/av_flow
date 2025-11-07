"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import tqdm
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
import cv2

from data.utils import InfiniteDataloader, HeadposeTransform

from models.renderer import Renderer
from models.renderer_audio2photoreal import BodyRenderer

from utils.video import StreamingMP4File
 

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        # General parameters
        self.artifacts_dir = config["artifacts_dir"]
        self.save_frequency = config.trainer.get("save_frequency", 5000)
        self.num_workers = config.trainer.get("num_workers", 10)
        self.val_frequency = config.trainer.get("val_frequency", 100)
        # Optimization parameters
        self.batch_size = config.trainer["batch_size"]
        self.learning_rate = config.trainer["learning_rate"]
        self.num_iterations = config.trainer["num_iterations"]
        self.weight_decay = config.trainer.get("weight_decay", 0.0)
        self.val_batch_size = config.trainer.get("val_batch_size", self.batch_size)
        # Loss parameters
        self.headpose_weight = config.trainer.get("headpose_weight", 0.2)
        self.audio_weight = config.trainer.get("headpose_weight", 1.0)
        # Body argument for audio2photoreal dataset
        self.body = config.dataset.get("body", False)

        os.makedirs(f"{self.artifacts_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.artifacts_dir}/viz", exist_ok=True)

        self.model = th.nn.DataParallel(model).cuda().train()
        
        self.config = config

        if self.body:
            # Audio2Photoreal dataset
            self.renderer_body = BodyRenderer(subject=config.subject).cuda()
        else:
            # 50h dataset
            dec_ckpt = f"{config.asset_dir}/decoder/dec_params.pt"
            vert_mean = f"{config.asset_dir}/decoder/vert_mean.bin"
            vert_var = f"{config.asset_dir}/decoder/vert_var.txt"
            self.renderer_face = Renderer(dec_ckpt, vert_mean, vert_var).cuda()
            self.headpose_transform = HeadposeTransform(
                head_translation_mean_file=f"{config.asset_dir}/headpose/headpose_normstats/head_translation_mean.npy",
                head_translation_std_file=f"{config.asset_dir}/headpose/headpose_normstats/head_translation_std.npy",
                head_qvec_mean_file=f"{config.asset_dir}/headpose/headpose_normstats/head_qvec_mean.npy",
                head_qvec_std_file=f"{config.asset_dir}/headpose/headpose_normstats/head_qvec_std.npy",
            )

        self.train_dataloader = InfiniteDataloader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        self.train_dataloader = iter(self.train_dataloader)

        self.val_dataloader = InfiniteDataloader(
            dataset=val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            seed=1234,  # set seed for same validation sampling in each experiment
        )
        self.val_dataloader = iter(self.val_dataloader)
       
        self.optimizer = th.optim.AdamW(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        self.start_iteration = 1
        self.iter = 1
        self.writer = SummaryWriter(f"{self.artifacts_dir}/logs/")

    def save_checkpoint(self, iteration: int):
        filename = f"iter-{iteration:07d}.pt"
        th.save({
            "iteration": iteration,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, f"{self.artifacts_dir}/checkpoints/{filename}")
    
    def load_checkpoint(self, checkpoint: str):
        ckpt = th.load(checkpoint)
        self.start_iteration = ckpt["iteration"] + 1
        self.iter = self.start_iteration
        self.model.module.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    def train(self):
        # Training loop
        pbar = tqdm.tqdm(range(self.start_iteration, self.num_iterations + 1))
        for self.iter in pbar:
            data = next(self.train_dataloader)
            output = self.train_step(data)
            pbar.set_description(f"{output['loss'].cpu().item():.4f}")

            self.writer.add_scalar("loss", output["loss"], self.iter)
            if "output" in output.keys():
                for k, v in output["output"].items():
                    self.writer.add_scalar(k, v, self.iter)

            # Validation
            if self.iter % self.val_frequency == 0:
                val_data = next(self.val_dataloader)
                self.val_step(val_data)

            # Save checkpoint and make sure to flush tensorboard writer
            if self.iter % self.save_frequency == 0:
                self.save_checkpoint(self.iter)
                self.writer.flush()

        self.save_checkpoint(self.num_iterations)
        self.writer.close()

    def visualize(self, pred, gt=None, audio=None, audio_gen=None, suffix=None, multiple=False, images=None,
                  sample_rate=48000, sample_rate_gen=16000, frame_rate=30):
        """
        Visualize a sample (first element in batch)
        :param pred: {"expr_code": B x T x 256, "head_R": B x T x 3 x 3, "head_t": B x T x 3}
        :param gt: {"expr_code": B x T x 256, "head_R": B x T x 3 x 3, "head_t": B x T x 3}
        :param audio: B x n_samples
        :param suffix: optional suffix for output filename
        """
        if suffix is None or suffix == "":
            suffix = ""
        else:
            suffix = "_" + suffix

        with th.no_grad():
            frames = None
            if gt is not None:
                if self.body:
                    render = self.render_body(gt["body_pose"][0], gt["expr_code"][0])
                    frames_gt = render
                else:
                    render = self.render(gt["expr_code"][0], gt["head_R"][0], gt["head_t"][0])
                    frames_gt = render.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
                frames = frames_gt
            if images is not None:
                images = [cv2.imread(img)[..., ::-1][None, -1024:, :1024] for img in images]
                images = np.concatenate(images, axis=0)
                frames = np.concatenate([frames, images], axis=2) if frames is not None else images
            if multiple:
                n = pred["expr_code"].shape[0]
            else:
                n = 1
            for i in range(n):
                if self.body:
                    render = self.render_body(pred["body_pose"][i], pred["expr_code"][i])
                    frames_pred = render
                else:
                    render = self.render(pred["expr_code"][i], pred["head_R"][i], pred["head_t"][i])
                    frames_pred = render.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
                frames = np.concatenate([frames, frames_pred], axis=2) if frames is not None else frames_pred

        if audio is not None:
            video_stream = StreamingMP4File(
                f"{self.artifacts_dir}/viz/sample_iter-{self.iter:07d}{suffix}.mp4",
                mode="w",
                with_audio=True,
                video_kwargs={"framerate": frame_rate},
                audio_kwargs={"sample_rate": sample_rate}
            )
            for i in range(frames.shape[0]):
                video_stream.write({"video": frames[i]})
            video_stream.write({"audio": audio[0].cpu().numpy()})
            video_stream.close()

        if audio_gen is not None:
            video_stream = StreamingMP4File(
                f"{self.artifacts_dir}/viz/sample_iter-{self.iter:07d}{suffix}_gen.mp4",
                mode="w",
                with_audio=True,
                video_kwargs={"framerate": frame_rate},
                audio_kwargs={"sample_rate": sample_rate_gen}
            )
            for i in range(frames.shape[0]):
                video_stream.write({"video": frames[i]})
            video_stream.write({"audio": audio_gen.cpu().numpy()})
            video_stream.close()

    def render(self, expression_codes, head_R, head_t):
        # Rendering loop for the 50h dataset
        headpose = th.cat([head_R, head_t.unsqueeze(-1)], dim=-1)
        render = []
        with th.no_grad():
            for i in range(0, expression_codes.shape[0], 60):
                e = expression_codes[i:i+60]
                h = headpose[i:i+60]
                r, _ = self.renderer_face(e, headpose=h)
                render.append(r)
            render = th.cat(render, dim=0)
        return render

    def render_body(self, body_pose, expression_codes):
        # Rendering loop for the Audio2Photoreal dataset
        render = []
        with th.no_grad():
            for i in range(0, expression_codes.shape[0], 60):
                e = expression_codes[i:i+60]
                b = body_pose[i:i+60]
                r = self.renderer_body(b, e, only_head=True)
                render.append(np.stack(r, axis=0))
            render = np.concatenate(render, axis=0)
        return render
    
    def train_step(self, data):
        raise NotImplementedError()

    def val_step(self, data):
        raise NotImplementedError()
