# code implement reference: https://github.com/AntixK/PyTorch-VAE
import torch
import torch.nn as nn 
import numpy as np
import os
import random
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

class VAE(nn.Module):
    def __init__(self, latent_dim=10, in_channels=3, hidden_channels: List = [32, 64, 128, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        encoder_list = []
        for hidden in hidden_channels:
            encoder_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(hidden)
                )
            )
            in_channels = hidden
        self.encoder = nn.Sequential(*encoder_list) # output: B, 256, 2, 2

        # predict mu and logvar respectively
        # here we predict logvar instead of var because it's easier to calculate kl divergence
        self.fc_mu = nn.Linear(hidden_channels[-1]*4, latent_dim) 
        self.fc_var = nn.Linear(hidden_channels[-1]*4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_channels[-1]*4)
        hidden_channels = hidden_channels[::-1]
        decoder_list = []
        for i in range(len(hidden_channels) - 1):
            decoder_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i],
                                       hidden_channels[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*decoder_list)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_channels[-1],
                                               hidden_channels[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_channels[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_channels[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        latent = self.reparameterize(mu, logvar)
        return latent, mu, logvar
    def decode(self, latent):
        latent = self.decoder_input(latent)
        latent = self.decoder(latent)
        x_reconstructed = self.final_layer(latent)
        return x_reconstructed
    def kl_loss(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    def forward(self, x):
        loss = {"kl_loss":0,
                "recon_loss":0,
                "loss":0}
        latent, mu, logvar = self.encode(x)
        x_reconstructed = self.decode(latent)
        loss["kl_loss"] = self.kl_loss(mu, logvar)
        loss["recon_loss"] = F.mse_loss(x_reconstructed, x)
        loss["loss"] = loss["kl_loss"] + loss["recon_loss"]
        return x_reconstructed, loss
    def sample(self, num_samples=1):
        latent = torch.randn(num_samples,
                        self.latent_dim)
        latent = latent.to(self.encoder.device)
        samples = self.decode(latent)
        return samples
    def varify(self, x, scale = 1e-3):
        x = x.view(-1, 3, 32, 32)
        B = x.shape[0]
        _, mu, logvar = self.encode(x)
        noise = torch.randn(B, self.latent_dim) * scale
        noise = noise.to(self.encoder.device)
        latent = mu + noise
        return self.decode(latent)


