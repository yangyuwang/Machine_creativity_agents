"""
Step 1: Train latent diffusion model from scratch.
This is the main training script for the U-Net denoising model.

MODIFICATION (model-structure only):
- Add a contrastive objective that trains special_token_embeddings via in-batch negatives (InfoNCE).
- Uses VAE latents as the "image representation" (no new CLI args; CLIP remains frozen).
- Adds two small projection heads:
    * image_proj: pooled latent -> hidden_size
    * token_proj: pooled special-token embeds -> hidden_size
- Total loss = diffusion_mse + CONTRASTIVE_WEIGHT * contrastive_loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append(str(Path(__file__).parent))

from utils.model_architecture import UNetModel, VAE
from utils.data_loader import PaintingDataset, create_data_loaders
from utils.special_tokens import SpecialTokenVocabulary
from utils.visualization import save_image_grid, plot_training_curves


# ----------------------------
# Contrastive config (NO CLI changes)
# ----------------------------
CONTRASTIVE_WEIGHT = 0.1   # how much to weight contrastive vs diffusion loss
CONTRASTIVE_TEMP = 0.07    # temperature for InfoNCE
MAX_SPECIAL_TOKENS = 10    # same behavior as your original code


class DiffusionModel:
    """Diffusion model with DDPM noise schedule."""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Cosine schedule (from Nichol & Dhariwal 2021)
        self.betas = self.cosine_beta_schedule(num_timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def cosine_beta_schedule(self, timesteps, beta_start, beta_end, s=0.008):
        """Cosine schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0).

        Args:
            x_start: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional noise to use

        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, context):
        """
        Reverse diffusion: sample x_{t-1} from x_t.
        """
        predicted_noise = model(x, t, context)

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        model_mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
        )

        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, model, shape, context, device):
        """Generate samples using DDPM sampling."""
        model.eval()
        x = torch.randn(shape).to(device)

        for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, context)

        return x


class TextEncoder(nn.Module):
    """
    Text encoder using CLIP with special token embeddings.
    PLUS: contrastive heads to train special tokens with InfoNCE.
    """

    def __init__(self, vocab, device='cuda'):
        super().__init__()
        self.device = device

        # Load CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

        # Freeze CLIP
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.vocab = vocab

        # Create learnable embeddings for special tokens
        num_special_tokens = len(vocab.token_to_id)
        self.special_token_embeddings = nn.Embedding(
            num_special_tokens,
            self.text_model.config.hidden_size
        ).to(device)
        nn.init.normal_(self.special_token_embeddings.weight, std=0.02)

        # ----------------------------
        # Contrastive heads (trainable)
        # ----------------------------
        hidden = self.text_model.config.hidden_size
        self.image_proj = nn.Linear(4, hidden, bias=True).to(device)   # pooled VAE latent has 4 channels in your setup
        self.token_proj = nn.Linear(hidden, hidden, bias=True).to(device)

    def encode(self, captions, special_tokens_list):
        """
        Encode captions with special tokens into ONE context sequence:
        [CLIP tokens ; special token embeds]
        """
        batch_size = len(captions)

        # Remove special tokens from captions for CLIP
        clean_captions = [self.vocab.remove_special_tokens(cap) for cap in captions]

        # Encode with CLIP
        inputs = self.tokenizer(
            clean_captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            clip_embeddings = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        # Add special token embeddings (append at end)
        special_embeds = []
        for tokens_dict in special_tokens_list:
            all_tokens = []
            for category in ['artist', 'year', 'gender', 'location']:
                all_tokens.extend(tokens_dict.get(category, []))

            if all_tokens:
                token_ids = [self.vocab.get_token_id(t) for t in all_tokens[:MAX_SPECIAL_TOKENS]]
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                embeds = self.special_token_embeddings(token_ids)  # [L, hidden]
            else:
                embeds = torch.zeros(1, self.text_model.config.hidden_size, device=self.device)

            special_embeds.append(embeds)

        # Pad special embeddings to same length
        max_special_len = max(e.shape[0] for e in special_embeds)
        padded_special = []
        for embeds in special_embeds:
            if embeds.shape[0] < max_special_len:
                padding = torch.zeros(
                    max_special_len - embeds.shape[0],
                    embeds.shape[1],
                    device=self.device
                )
                embeds = torch.cat([embeds, padding], dim=0)
            padded_special.append(embeds)

        special_embeds = torch.stack(padded_special)  # [B, max_special_len, hidden_dim]
        context = torch.cat([clip_embeddings, special_embeds], dim=1)
        return context

    def _pool_special_tokens(self, special_tokens_list):
        """
        Build a pooled special-token vector per example by averaging its special-token embeddings.
        Returns: pooled [B, hidden]
        """
        pooled = []
        hidden = self.text_model.config.hidden_size

        for tokens_dict in special_tokens_list:
            all_tokens = []
            for category in ['artist', 'year', 'gender', 'location']:
                all_tokens.extend(tokens_dict.get(category, []))

            if all_tokens:
                token_ids = [self.vocab.get_token_id(t) for t in all_tokens[:MAX_SPECIAL_TOKENS]]
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                embeds = self.special_token_embeddings(token_ids)  # [L, hidden]
                pooled_vec = embeds.mean(dim=0)  # [hidden]
            else:
                pooled_vec = torch.zeros(hidden, device=self.device)

            pooled.append(pooled_vec)

        return torch.stack(pooled, dim=0)  # [B, hidden]

    def contrastive_loss(self, latents_mu, special_tokens_list, temperature=CONTRASTIVE_TEMP):
        """
        Contrastive objective aligning:
        - image representation: pooled VAE latent mean (mu) -> image_proj -> normalized
        - token representation: pooled special token embeds -> token_proj -> normalized

        Uses in-batch negatives (InfoNCE), symmetric (i->t and t->i).
        """
        # latents_mu: [B, 4, H, W]
        B = latents_mu.shape[0]

        # Image repr: mean pool over spatial -> [B, 4]
        img_feat = latents_mu.mean(dim=(2, 3))  # [B, 4]
        img_feat = self.image_proj(img_feat)    # [B, hidden]

        # Token repr: mean of special token embeddings -> [B, hidden] then project
        tok_feat = self._pool_special_tokens(special_tokens_list)  # [B, hidden]
        tok_feat = self.token_proj(tok_feat)                       # [B, hidden]

        # Normalize
        img_feat = nn.functional.normalize(img_feat, dim=-1)
        tok_feat = nn.functional.normalize(tok_feat, dim=-1)

        # Similarity logits [B, B]
        logits = (img_feat @ tok_feat.t()) / temperature

        # Targets are diagonal matches
        targets = torch.arange(B, device=logits.device, dtype=torch.long)

        loss_i2t = nn.functional.cross_entropy(logits, targets)
        loss_t2i = nn.functional.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_i2t + loss_t2i)


def train_epoch(model, vae, text_encoder, diffusion, dataloader, optimizer, device, epoch):
    """Train diffusion model for one epoch (diffusion MSE + contrastive InfoNCE)."""
    model.train()
    text_encoder.special_token_embeddings.train()
    text_encoder.image_proj.train()
    text_encoder.token_proj.train()

    total_loss = 0.0
    total_diff = 0.0
    total_con = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['images'].to(device)
        captions = batch['captions']
        special_tokens = batch['special_tokens']

        # Encode images to latent space
        with torch.no_grad():
            mu, _ = vae.encode(images)
            latents_mu = mu  # Use mean for stability

        # Sample timesteps
        batch_size = latents_mu.shape[0]
        timesteps = torch.randint(
            0, diffusion.num_timesteps, (batch_size,),
            device=device, dtype=torch.long
        )

        # Add noise
        noise = torch.randn_like(latents_mu)
        noisy_latents = diffusion.q_sample(latents_mu, timesteps, noise)

        # Encode text (CLIP part frozen; special tokens trainable)
        context = text_encoder.encode(captions, special_tokens)

        # Predict noise
        predicted_noise = model(noisy_latents, timesteps, context)

        # Diffusion loss
        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)

        # Contrastive loss (trains special tokens via in-batch negatives)
        contrastive_loss = text_encoder.contrastive_loss(latents_mu, special_tokens)

        # Total
        loss = diffusion_loss + CONTRASTIVE_WEIGHT * contrastive_loss

        optimizer.zero_grad()
        loss.backward()

        # Clip grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(text_encoder.special_token_embeddings.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(text_encoder.image_proj.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(text_encoder.token_proj.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        total_diff += diffusion_loss.item()
        total_con += contrastive_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'diff': f'{diffusion_loss.item():.4f}',
            'con': f'{contrastive_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_diff = total_diff / len(dataloader)
    avg_con = total_con / len(dataloader)
    return avg_loss, avg_diff, avg_con


@torch.no_grad()
def validate(model, vae, text_encoder, diffusion, dataloader, device):
    """Validate diffusion model (reports total loss = diffusion + weight * contrastive)."""
    model.eval()
    text_encoder.special_token_embeddings.eval()
    text_encoder.image_proj.eval()
    text_encoder.token_proj.eval()

    total_loss = 0.0
    total_diff = 0.0
    total_con = 0.0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['images'].to(device)
        captions = batch['captions']
        special_tokens = batch['special_tokens']

        mu, _ = vae.encode(images)
        latents_mu = mu

        batch_size = latents_mu.shape[0]
        timesteps = torch.randint(
            0, diffusion.num_timesteps, (batch_size,),
            device=device, dtype=torch.long
        )

        noise = torch.randn_like(latents_mu)
        noisy_latents = diffusion.q_sample(latents_mu, timesteps, noise)

        context = text_encoder.encode(captions, special_tokens)
        predicted_noise = model(noisy_latents, timesteps, context)

        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
        contrastive_loss = text_encoder.contrastive_loss(latents_mu, special_tokens)

        loss = diffusion_loss + CONTRASTIVE_WEIGHT * contrastive_loss

        total_loss += loss.item()
        total_diff += diffusion_loss.item()
        total_con += contrastive_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_diff = total_diff / len(dataloader)
    avg_con = total_con / len(dataloader)
    return avg_loss, avg_diff, avg_con


@torch.no_grad()
def generate_samples(model, vae, text_encoder, diffusion, dataloader, device, save_path, json_path, num_samples=8):
    """Generate sample images."""
    model.eval()
    text_encoder.special_token_embeddings.eval()
    text_encoder.image_proj.eval()
    text_encoder.token_proj.eval()
    vae.eval()

    batch = next(iter(dataloader))
    captions = batch['captions'][:num_samples]
    special_tokens = [batch['special_tokens'][i] for i in range(num_samples)]

    context = text_encoder.encode(captions, special_tokens)

    latent_shape = (num_samples, 4, 32, 32)  # Assuming 256x256 images -> 32x32 latents
    latents = diffusion.sample(model, latent_shape, context, device)

    images = vae.decode(latents)

    save_image_grid(images, save_path, nrow=4)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vocabulary...")
    vocab = SpecialTokenVocabulary()
    if Path(args.vocab_path).exists():
        vocab.load_vocabulary(args.vocab_path)
    else:
        from utils.special_tokens import build_vocabulary_from_dataset
        vocab = build_vocabulary_from_dataset(args.jsonl_path, args.vocab_path)

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        jsonl_path=args.jsonl_path,
        image_dir=args.image_dir,
        vocab=vocab,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    print("Loading VAE...")
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        hidden_dims=[128, 256, 512, 512]
    ).to(device)

    if Path(args.vae_path).exists():
        checkpoint = torch.load(args.vae_path, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VAE from {args.vae_path}")
    else:
        print(f"Warning: VAE not found at {args.vae_path}. Using random weights.")

    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    print("Loading text encoder...")
    text_encoder = TextEncoder(vocab, device)
    print(f"Special token embeddings: {len(vocab.token_to_id)}")

    print("Creating U-Net model...")
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(4, 2, 1),
        channel_mult=(1, 2, 4, 4),
        num_heads=args.num_heads,
        context_dim=text_encoder.text_model.config.hidden_size,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {num_params:,}")

    diffusion = DiffusionModel(num_timesteps=args.num_timesteps)

    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                 'posterior_variance']:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    # Optimizer: include U-Net + special-token embeddings + contrastive heads
    params = (
        list(model.parameters()) +
        list(text_encoder.special_token_embeddings.parameters()) +
        list(text_encoder.image_proj.parameters()) +
        list(text_encoder.token_proj.parameters())
    )

    optimizer = optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )

    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_diff, train_con = train_epoch(
            model, vae, text_encoder, diffusion,
            train_loader, optimizer, device, epoch
        )

        val_loss, val_diff, val_con = validate(
            model, vae, text_encoder, diffusion, val_loader, device
        )

        scheduler.step()

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} (diff {train_diff:.4f}, con {train_con:.4f}, w={CONTRASTIVE_WEIGHT})")
        print(f"Val   Loss: {val_loss:.4f} (diff {val_diff:.4f}, con {val_con:.4f}, w={CONTRASTIVE_WEIGHT})")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'text_encoder_state_dict': text_encoder.special_token_embeddings.state_dict(),
                # Save the new heads too (model structure change only)
                'image_proj_state_dict': text_encoder.image_proj.state_dict(),
                'token_proj_state_dict': text_encoder.token_proj.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'diffusion_best.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f})")

        if epoch % args.sample_every == 0:
            generate_samples(
                model, vae, text_encoder, diffusion,
                val_loader, device,
                output_dir / f'samples_epoch_{epoch}.png',
                output_dir / f'samples_epoch_{epoch}.json'
            )

    checkpoint = {
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'text_encoder_state_dict': text_encoder.special_token_embeddings.state_dict(),
        'image_proj_state_dict': text_encoder.image_proj.state_dict(),
        'token_proj_state_dict': text_encoder.token_proj.state_dict(),
        'args': vars(args)
    }
    torch.save(checkpoint, output_dir / 'diffusion_final.pt')

    plot_training_curves(train_losses, val_losses, output_dir / 'training_curves.png')
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train latent diffusion model')

    # Data
    parser.add_argument('--jsonl_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged_1400_1600.jsonl')
    parser.add_argument('--image_dir', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/artwork_images')
    parser.add_argument('--vocab_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json')
    parser.add_argument('--vae_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt')

    # Model
    parser.add_argument('--model_channels', type=int, default=192,
                       help='Base model channels (smaller for faster training)')
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./outputs/diffusion')
    parser.add_argument('--sample_every', type=int, default=10)

    args = parser.parse_args()
    main(args)
