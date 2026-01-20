import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        hidden_dims = [32, 64, 128, 256]
        all_dims = hidden_dims + [out_channels]
        current_in = in_channels

        for h_dim in all_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_in,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            current_in = h_dim
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        hidden_dims = [256, 128, 64, 32]
        current_in = in_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_in,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            current_in = h_dim

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    current_in,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.Tanh()
            )
        )
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder that extracts features from an input.
        :param features_decoder: Instance of a decoder that reconstructs an input from its features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.mu_layer = nn.Linear(n_features, z_dim)
        self.log_sigma2_layer = nn.Linear(n_features, z_dim)
        self.z_to_h = nn.Linear(z_dim, n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x)
        h = h.reshape(h.size(0), -1)

        mu = self.mu_layer(h)
        log_sigma2 = self.log_sigma2_layer(h)

        std = torch.exp(0.5 * log_sigma2)
        eps = torch.randn_like(mu)
        z = mu + (std * eps)
        # ========================
        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h_flat = self.z_to_h(z)
        h_spatial = h_flat.reshape(h_flat.size(0), *self.features_shape)
        x_rec = self.features_decoder(h_spatial)
        # ========================
        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            z = torch.randn(n, self.z_dim, device=device)
            samples = self.decode(z)

            # ========================
        # Detach and move to CPU for display purposes.
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    N = x.shape[0]
    dx = x[0].numel()
    dz = z_mu.shape[1]
    x_flat = x.view(N, -1)
    xr_flat = xr.view(N, -1)
    recon_sq_diff = torch.sum((x_flat - xr_flat) ** 2, dim=1)
    data_loss_term = (1.0 / (x_sigma2 * dx)) * recon_sq_diff
    tr_sigma = torch.sum(torch.exp(z_log_sigma2), dim=1)
    mu_sq = torch.sum(z_mu ** 2, dim=1)
    log_det_sigma = torch.sum(z_log_sigma2, dim=1)
    kldiv_loss_term = tr_sigma + mu_sq - dz - log_det_sigma
    total_loss_term = data_loss_term + kldiv_loss_term
    loss = total_loss_term.mean()
    data_loss = data_loss_term.mean()
    kldiv_loss = kldiv_loss_term.mean()

    # ========================

    return loss, data_loss, kldiv_loss
