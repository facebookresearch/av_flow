"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn


class VanillaVAE(nn.Module):

    def __init__(self,
                 input_size=512,
                 latent_dim=64,
                 hidden_dims=[512, 256, 128],
                 kld_weight=1.0,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        modules = []

        # Build Encoder
        prev_size = input_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.LayerNorm(prev_size),
                    nn.Linear(prev_size, h_dim),
                    nn.GELU())
            )
            prev_size = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dims[i]),
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.GELU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], input_size)

    def encode(self, input):
        result = self.encoder(input)

        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(
            self,
            recons,
            input,
            mu,
            log_var,
            **kwargs
        ) -> dict:
        # VAE loss function
        log_var = log_var.reshape(-1, log_var.shape[-1])
        mu = mu.reshape(-1, mu.shape[-1])

        recons_loss = torch.mean(torch.abs(recons - input))

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]


class TransformerEncoderVAE(VanillaVAE):

    def __init__(self,
                 input_size=7,
                 latent_dim=8,
                 hidden_size=16,
                 n_layers=1,
                 n_heads=1,
                 kld_weight=1.0,
                 **kwargs) -> None:
        super(TransformerEncoderVAE, self).__init__()

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        # Build Encoder
        modules = []

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            batch_first=True,
        )

        encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        input_linear = nn.Linear(input_size, hidden_size)

        modules.append(input_linear)
        modules.append(encoder)

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_size)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            batch_first=True,
        )

        decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers,
        )
        modules.append(decoder)

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_size, input_size)
