"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import torch as th
import torchaudio as ta

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from learn.trainer import Trainer
from models.audio_encoders import MelSpectrogramforBigVGAN

from models.vae import TransformerEncoderVAE


class AVFlowMatchingTrainer(Trainer):
    def __init__(self, model, train_dataset, val_dataset, config):
        super().__init__(model, train_dataset, val_dataset, config)

        # Flow matching parameters
        self.sigma_min = config.trainer.get("sigma_min", 1e-6)
        self.nfe_steps = config.trainer.get("nfe_steps", 32)
        self.noise_temperature = config.trainer.get("noise_temperature", 1.0)
        self.input_sample_rate = config.dataset.get("input_audio_sample_rate", 48000)
        self.sample_rate = config.get("output_audio_sample_rate", 22050)
        self.frame_rate = config.dataset.frame_rate

        # Input dimensions
        self.face_dim = config.face_dim
        self.headvec_dim = config.headvec_dim
        self.melspec_dim = config.melspec_dim

        # Loss parameters
        self.loss_ema = None
        loss_fn_audio = config.trainer.get("loss_fn_audio", "l2")  # can be l1 or l2
        loss_fn_vision = config.trainer.get("loss_fn_vision", "l1")
        if loss_fn_audio == "l1":
            self.loss_fn_audio = lambda a, b: th.mean(th.abs(a - b))
        elif loss_fn_audio == "l2":
            self.loss_fn_audio = lambda a, b: th.mean((a - b) ** 2)
        if loss_fn_vision == "l1":
            self.loss_fn_vision = lambda a, b: th.mean(th.abs(a - b))
        elif loss_fn_vision == "l2":
            self.loss_fn_vision = lambda a, b: th.mean((a - b) ** 2)
        
        self.resampler = ta.transforms.Resample(self.input_sample_rate, self.sample_rate)

        self.melspec_extractor = MelSpectrogramforBigVGAN(
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.sample_rate,
            vocoder_ckpt_path=config.trainer.vocoder_ckpt_path,
        ).cuda()

        head_vae_args = {
            "input_size": config.dataset.input_head_dim,
            "latent_dim": config.headvec_dim,
            "hidden_size": 16,
            "n_layers": 1,
            "n_heads": 1,
            "kld_weight": 0.0001
        }
        self.head_vae = TransformerEncoderVAE(**head_vae_args)
        self.head_vae = th.nn.DataParallel(self.head_vae).cuda().eval()
        ckpt = th.load(config.trainer.head_vae_ckpt_path)
        self.head_vae.module.load_state_dict(ckpt["model_state_dict"])
    
    def prepare_model_inputs(self, batch):
        self_melspec = self.melspec_extractor(batch["self_audio"].cuda())
        other_melspec = self.melspec_extractor(batch["other_audio"].cuda())
        self_audio = self.resampler(batch["self_audio"])
        other_audio = self.resampler(batch["other_audio"])
        model_inputs = {
            "self_audio": self_audio.cuda(),
            "other_audio": other_audio.cuda(),
            "self_melspec": self_melspec.cuda(),
            "other_melspec": other_melspec.cuda(),
            "self_tokens": batch["self_tokens"].cuda(),
            "other_tokens": batch["other_tokens"].cuda(),
            "expr_code": batch["expr_code"].cuda(),
        }
        if self.body:
            model_inputs["body_pose"] = batch["body_pose"][...,  12:16].cuda()  # neck and head
        else:
            headvec = self.headpose_transform.headRt2headvec(batch["head_R"], batch["head_t"])
            model_inputs["headvec"] = headvec.cuda()
            model_inputs["other_expr"] = batch["other_expr"].cuda()
            model_inputs["other_head"] = batch["other_head"].cuda()

        return model_inputs

    def visualize_melspec(self, pred, gt, suffix=None, n=1):
        # Visualize mel-spectrogram as an image
        img_path = f"{self.artifacts_dir}/viz/sample_iter-{self.iter:07d}{suffix}.png"

        fig, (axs) = plt.subplots(1, n + 1, figsize=(10 * (n + 1), 10))

        gt = gt[0].transpose(0, 1).detach().cpu().numpy()
        axs[0].imshow(gt, origin='lower')
        axs[0].axis('off')
        for i in range(n):
            pred_i = pred[i].transpose(0, 1).detach().cpu().numpy()
            axs[i + 1].imshow(pred_i, origin='lower')
            axs[i + 1].axis('off')

        fig.savefig(img_path, bbox_inches='tight')
        fig.clf()
        plt.close("all")
    
    def downsample(self, x, target_frame_rate=30):
        # Downsample to 30 fps (useful before rendering)
        x = x.transpose(1, 2).contiguous()
        x = th.nn.functional.interpolate(x, scale_factor=target_frame_rate / self.frame_rate, mode="linear")
        x = x.transpose(1, 2).contiguous()
        return x

    def euler_solver(self, x0, model_inputs):
        self.model.eval()
        ts = th.linspace(0, 1, self.nfe_steps+1)[1:]

        t = 0        
        x = x0

        for tau in ts:
            dt = tau - t
            t = th.ones(x.shape[0], device=x.device) * t
            dphi_dt = self.model(x, t, **model_inputs)
            x = x + dt * dphi_dt
            t = tau
        
        return x

    def train_step(self, data):
        self.model.train()

        model_inputs = self.prepare_model_inputs(data)

        if self.body:
            headvec = model_inputs["body_pose"]
        else:
            headvec = model_inputs["headvec"]
        
        # VAE encoder for head pose
        _, headvec, _ = self.head_vae(headvec)

        x1 = th.cat([model_inputs["expr_code"], headvec, model_inputs["self_melspec"]], dim=-1)

        x0 = th.randn_like(x1)
        t_shape = [x0.shape[0]] + [1 for _ in range(len(x0.shape[1:]))]
        t = th.rand(t_shape, device=x0.device)

        xt = (1 - (1 - self.sigma_min) * t) * x0 + t * x1

        v = self.model(xt, t.squeeze(), **model_inputs)

        # Prediction of x1 based on predicted flow (assuming OT flow)
        x1_pred = v + (1 - self.sigma_min) * x0
        expr_pred, headvec_pred, melspec_pred = x1_pred.split([self.face_dim, self.headvec_dim, self.melspec_dim], dim=-1)

        self.optimizer.zero_grad()

        # Original OT flow loss is ||v - u|| but that's equivalent to ||x1_pred - x1||
        expr_loss = self.loss_fn_vision(expr_pred, model_inputs["expr_code"])
        headpose_loss = self.loss_fn_vision(headvec_pred, headvec)
        melspec_loss = self.loss_fn_audio(melspec_pred, model_inputs["self_melspec"])

        loss = expr_loss + self.headpose_weight * headpose_loss + self.audio_weight * melspec_loss

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss,
            "output": {
                "expr_loss": expr_loss,
                "headpose_loss": headpose_loss,
                "melspec_loss": melspec_loss,
            }
        }
    
    def val_step(self, data, suffix=None):
        self.model.eval()

        suffix = "" if suffix is None else "_" + suffix

        model_inputs = self.prepare_model_inputs(data)

        B, T, _ = model_inputs["expr_code"].shape

        x0 = th.randn(B, T, self.face_dim + self.headvec_dim + self.melspec_dim, device=model_inputs["expr_code"].device) * self.noise_temperature
        
        with th.no_grad():
            sample = self.euler_solver(x0, model_inputs)
        
        expr_pred, headvec_pred, melspec_pred = sample.split([self.face_dim, self.headvec_dim, self.melspec_dim], dim=-1)

        headvec_pred = self.head_vae.module.decode(headvec_pred)

        if self.body:
            body_pred = headvec_pred
            sample = {
                "expr_code": expr_pred,
                "body_pose": body_pred,
                "melspec": melspec_pred,
            }
            gt = {
                "expr_code": model_inputs["expr_code"],
                "body_pose": model_inputs["body_pose"],
                "melspec": model_inputs["self_melspec"],
            }
        else:
            head_R, head_t = self.headpose_transform.headvec2headRt(headvec_pred)
            sample = {
                "expr_code": expr_pred,
                "head_R": head_R,
                "head_t": head_t,
                "melspec": melspec_pred,
            }
            gt = {
                "expr_code": model_inputs["expr_code"],
                "head_R": data["head_R"].cuda(),
                "head_t": data["head_t"].cuda(),
                "melspec": model_inputs["self_melspec"],
            }

        # Downsample from 86 fps to 30 fps for visualization
        if self.frame_rate != 30:
            n_samples = 1
            sample["expr_code"] = self.downsample(sample["expr_code"])
            gt["expr_code"] = self.downsample(gt["expr_code"])
            if self.body:
                sample["body_pose"] = self.downsample(sample["body_pose"])
                gt["body_pose"] = self.downsample(gt["body_pose"])
            else:
                head_R_pred = []
                for i_sample in range(n_samples):
                    head_R_pred.append(self.downsample(sample["head_R"][i_sample].transpose(0, 1)).transpose(0, 1)[None])
                sample["head_R"] = th.cat(head_R_pred, dim=0)
                sample["head_t"] = self.downsample(sample["head_t"])
                gt["head_R"] = self.downsample(gt["head_R"][0].transpose(0, 1)).transpose(0, 1)[None]
                gt["head_t"] = self.downsample(gt["head_t"])

        audio = th.stack([data["self_audio"], data["other_audio"]], dim=1)

        self.visualize(sample, gt, audio, suffix=suffix, frame_rate=30, sample_rate_gen=self.sample_rate)
        self.visualize_melspec(melspec_pred, model_inputs["self_melspec"], suffix=suffix)

        return

    def test_step(self, data, suffix=None, n_samples=3, images=None, sample_rate=48000, save_results=True):
        self.model.eval()

        suffix = "" if suffix is None else "_" + suffix

        model_inputs = self.prepare_model_inputs(data)

        B, T, _ = data["expr_code"].shape

        _, model_inputs["headvec"], _ = self.head_vae(model_inputs["headvec"])

        expr_codes = []
        head_Rs = []
        head_ts = []
        headvecs = []
        melspecs = []
        body_poses = []
        for _ in range(n_samples):
            x0 = th.randn(B, T, self.face_dim + self.headvec_dim + self.melspec_dim, device=model_inputs["expr_code"].device) * self.noise_temperature

            with th.no_grad():
                sample = self.euler_solver(x0, model_inputs)
            
            expr_pred, headvec_pred, melspec_pred = sample.split([self.face_dim, self.headvec_dim, self.melspec_dim], dim=-1)

            headvec_pred = self.head_vae.module.decode(headvec_pred)

            if self.body:
                body_poses.append(headvec_pred)
            else:
                headvecs.append(headvec_pred)
                head_R, head_t = self.headpose_transform.headvec2headRt(headvec_pred)
                head_Rs.append(head_R)
                head_ts.append(head_t)

            expr_codes.append(expr_pred)
            melspecs.append(melspec_pred)

        if self.body:
            sample = {
                "expr_code": th.cat(expr_codes, dim=0),
                "body_pose": th.cat(body_poses, dim=0),
                "melspec": th.cat(melspecs, dim=0),
            }
            gt = {
                "expr_code": model_inputs["expr_code"],
                "body_pose": model_inputs["body_pose"],
                "melspec": model_inputs["self_melspec"],
            }
        else:
            sample = {
                "expr_code": th.cat(expr_codes, dim=0),
                "head_R": th.cat(head_Rs, dim=0),
                "head_t": th.cat(head_ts, dim=0),
                "headvec": th.cat(headvecs, dim=0),
                "melspec": th.cat(melspecs, dim=0),
            }
            gt = {
                "expr_code": model_inputs["expr_code"],
                "head_R": data["head_R"].cuda(),
                "head_t": data["head_t"].cuda(),
                "melspec": model_inputs["self_melspec"],
            }

        if self.frame_rate != 30:
            sample["expr_code"] = self.downsample(sample["expr_code"])
            gt["expr_code"] = self.downsample(gt["expr_code"])
            if self.body:
                sample["body_pose"] = self.downsample(sample["body_pose"])
                gt["body_pose"] = self.downsample(gt["body_pose"])
                sample["body_pose"] = gaussian_filter1d(sample["body_pose"].detach().cpu().numpy(), 5, axis=1)  # Smooth
                sample["body_pose"] = th.from_numpy(sample["body_pose"]).cuda()
            else:
                head_R_pred = []
                for i_sample in range(n_samples):
                    head_R_pred.append(self.downsample(sample["head_R"][i_sample].transpose(0, 1)).transpose(0, 1)[None])
                sample["head_R"] = th.cat(head_R_pred, dim=0)
                sample["head_t"] = self.downsample(sample["head_t"])
                sample["headvec"] = self.downsample(sample["headvec"])
                gt["head_R"] = self.downsample(gt["head_R"][0].transpose(0, 1)).transpose(0, 1)[None]
                gt["head_t"] = self.downsample(gt["head_t"])

        audio = th.stack([data["self_audio"], data["other_audio"]], dim=1)

        sample_audio = self.melspec_extractor.reverse_to_audio(sample["melspec"][0:1]).cpu().detach()
        audio_recon = self.melspec_extractor.reverse_to_audio(model_inputs["self_melspec"]).cpu().detach()

        if save_results:
            gt_path = f"{self.artifacts_dir}/viz/gt_recon-{self.iter:07d}{suffix}.wav"
            ta.save(gt_path, audio_recon, sample_rate=self.sample_rate)
            for i_sample in range(n_samples):
                gen_path = f"{self.artifacts_dir}/viz/sample-{self.iter:07d}{suffix}_{i_sample}.wav"
                sample_audio = self.melspec_extractor.reverse_to_audio(sample["melspec"][i_sample:i_sample+1]).cpu().detach()
                ta.save(gen_path, sample_audio, sample_rate=self.sample_rate)
            # Save video    
            self.visualize(
                sample, gt, audio, audio_gen=sample_audio, multiple=True, images=images,
                sample_rate=sample_rate, sample_rate_gen=self.sample_rate, frame_rate=30, suffix=suffix
            )
            self.visualize_melspec(sample["melspec"][0:1], model_inputs["self_melspec"], suffix=suffix, n=1)  # visualize only the first

        return {"sample": sample, "gt": gt}

    def generate(self, conditions, segment_length=345 * 5, suffix=None, n_samples=3, save_results=True):
        self.model.eval()

        suffix = "" if suffix is None else "_" + suffix
        B = 1
        T = segment_length

        expr_codes = []
        head_Rs = []
        head_ts = []
        headvecs = []
        melspecs = []
        body_poses = []
        for _ in range(n_samples):
            x0 = th.randn(B, T, self.face_dim + self.headvec_dim + self.melspec_dim).cuda() * self.noise_temperature

            with th.no_grad():
                sample = self.euler_solver(x0, conditions)
            
            expr_pred, headvec_pred, melspec_pred = sample.split([self.face_dim, self.headvec_dim, self.melspec_dim], dim=-1)

            headvec_pred = self.head_vae.module.decode(headvec_pred)

            if self.body:
                body_poses.append(headvec_pred)
            else:
                headvecs.append(headvec_pred)
                head_R, head_t = self.headpose_transform.headvec2headRt(headvec_pred)
                head_Rs.append(head_R)
                head_ts.append(head_t)

            expr_codes.append(expr_pred)
            melspecs.append(melspec_pred)

        if self.body:
            sample = {
                "expr_code": th.cat(expr_codes, dim=0),
                "body_pose": th.cat(body_poses, dim=0),
                "melspec": th.cat(melspecs, dim=0),
            }
        else:
            sample = {
                "expr_code": th.cat(expr_codes, dim=0),
                "head_R": th.cat(head_Rs, dim=0),
                "head_t": th.cat(head_ts, dim=0),
                "headvec": th.cat(headvecs, dim=0),
                "melspec": th.cat(melspecs, dim=0),
            }

        if self.frame_rate != 30:
            sample["expr_code"] = self.downsample(sample["expr_code"])
            if self.body:
                sample["body_pose"] = self.downsample(sample["body_pose"])
                sample["body_pose"] = gaussian_filter1d(sample["body_pose"].detach().cpu().numpy(), 5, axis=1)  # Smooth
                sample["body_pose"] = th.from_numpy(sample["body_pose"]).cuda()
            else:
                head_R_pred = []
                for i_sample in range(n_samples):
                    head_R_pred.append(self.downsample(sample["head_R"][i_sample].transpose(0, 1)).transpose(0, 1)[None])
                sample["head_R"] = th.cat(head_R_pred, dim=0)
                sample["head_t"] = self.downsample(sample["head_t"])
                sample["headvec"] = self.downsample(sample["headvec"])

        if save_results:
            for i_sample in range(n_samples):
                gen_path = f"{self.artifacts_dir}/viz/sample-{self.iter:07d}{suffix}_{i_sample}.wav"
                sample_audio = self.melspec_extractor.reverse_to_audio(sample["melspec"][i_sample:i_sample+1]).cpu().detach()
                ta.save(gen_path, sample_audio, sample_rate=self.sample_rate)
            # Save video
            self.visualize(
                sample, gt=None, audio=None, audio_gen=sample_audio, multiple=True,
                sample_rate_gen=self.sample_rate, frame_rate=30, suffix=suffix
            )
