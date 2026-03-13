"""
Model architecture for latent diffusion model.
Includes U-Net, VAE, and text encoder components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return out + residual


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for text conditioning."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image features [B, C, H, W]
            context: Text features [B, seq_len, context_dim]
        """
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        q = self.q(x)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]

        kv = self.kv(context)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_heads, seq_len, head_dim]
        k, v = kv[0], kv[1]

        # Cross-attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return out + residual


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class UNetModel(nn.Module):
    """
    U-Net model for latent diffusion.
    Includes self-attention and cross-attention for text conditioning.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (4, 2, 1),
        channel_mult: Tuple[int] = (1, 2, 4, 4),
        num_heads: int = 8,
        context_dim: int = 768,
        time_embed_dim: int = 1280
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Input convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])

        # Downsampling blocks
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, model_channels * mult, time_embed_dim)]

                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                    layers.append(CrossAttentionBlock(ch, context_dim, num_heads))

                self.input_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_channels.append(ch)
                ds *= 2

        # Middle blocks
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim),
            AttentionBlock(ch, num_heads),
            CrossAttentionBlock(ch, context_dim, num_heads),
            ResBlock(ch, ch, time_embed_dim)
        ])

        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_channels.pop()
                layers = [ResBlock(ch + ich, model_channels * mult, time_embed_dim)]

                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                    layers.append(CrossAttentionBlock(ch, context_dim, num_heads))

                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2

                self.output_blocks.append(nn.ModuleList(layers))

        # Output convolution
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Latent input [B, in_channels, H, W]
            timesteps: Timestep tensor [B]
            context: Text context [B, seq_len, context_dim]

        Returns:
            Predicted noise [B, out_channels, H, W]
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)

        # Input blocks
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, CrossAttentionBlock):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)

        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, CrossAttentionBlock):
                h = layer(h, context)
            else:
                h = layer(h)

        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, CrossAttentionBlock):
                    h = layer(h, context)
                else:
                    h = layer(h)

        return self.out(h)


class VAE(nn.Module):
    """Variational Autoencoder for image compression."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_dims: List[int] = [128, 256, 512, 512]
    ):
        super().__init__()

        # Encoder
        encoder_modules = []
        channels = in_channels
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, h_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_channels, 3, padding=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], latent_channels, 3, padding=1)

        # Decoder
        self.decoder_input = nn.Conv2d(latent_channels, hidden_dims[-1], 3, padding=1)

        decoder_modules = []
        hidden_dims_reversed = hidden_dims[::-1]
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                hidden_dims_reversed[-1],
                3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims_reversed[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims_reversed[-1], in_channels, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        h = self.decoder_input(z)
        h = self.decoder(h)
        return self.final_layer(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


if __name__ == '__main__':
    # Test models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test VAE
    vae = VAE().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    recon, mu, log_var = vae(x)
    print(f"VAE input: {x.shape}, output: {recon.shape}, latent: {mu.shape}")

    # Test U-Net
    unet = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=128,  # Smaller for testing
        context_dim=768
    ).to(device)

    latent = torch.randn(2, 4, 32, 32).to(device)
    timesteps = torch.randint(0, 1000, (2,)).to(device)
    context = torch.randn(2, 77, 768).to(device)

    noise_pred = unet(latent, timesteps, context)
    print(f"U-Net input: {latent.shape}, output: {noise_pred.shape}")
