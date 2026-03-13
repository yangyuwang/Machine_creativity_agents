"""
Step 2: Train Variational Autoencoder (VAE) for image compression.
This creates the latent space for the diffusion model.
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

sys.path.append(str(Path(__file__).parent))

from utils.model_architecture import VAE
from utils.data_loader import PaintingDataset, create_data_loaders
from utils.special_tokens import SpecialTokenVocabulary
from utils.visualization import tensor_to_image, save_image_grid, plot_training_curves


def vae_loss(recon_x, x, mu, log_var, kl_weight=0.00001):
    """
    VAE loss = Reconstruction loss + KL divergence.

    Args:
        recon_x: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        mu: Latent mean [B, latent_dim, H, W]
        log_var: Latent log variance [B, latent_dim, H, W]
        kl_weight: Weight for KL divergence term

    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

    total_loss = recon_loss + kl_weight * kl_div

    return total_loss, recon_loss, kl_div


def train_epoch(model, dataloader, optimizer, device, epoch, kl_weight):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['images'].to(device)

        optimizer.zero_grad()

        # Forward pass
        recon, mu, log_var = model(images)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, log_var, kl_weight)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_loss


@torch.no_grad()
def validate(model, dataloader, device, kl_weight):
    """Validate VAE."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['images'].to(device)

        # Forward pass
        recon, mu, log_var = model(images)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, log_var, kl_weight)

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_loss


@torch.no_grad()
def save_reconstructions(model, dataloader, device, save_dir, num_samples=8):
    """Save reconstruction examples."""
    model.eval()
    batch = next(iter(dataloader))
    images = batch['images'][:num_samples].to(device)

    recon, _, _ = model(images)

    # Create comparison grid
    comparison = torch.cat([images, recon], dim=0)
    save_image_grid(
        comparison,
        save_dir / 'reconstructions.png',
        nrow=num_samples,
        titles=[f'Original {i}' for i in range(num_samples)] +
               [f'Recon {i}' for i in range(num_samples)]
    )


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = SpecialTokenVocabulary()
    if Path(args.vocab_path).exists():
        vocab.load_vocabulary(args.vocab_path)
    else:
        print(f"Vocabulary not found at {args.vocab_path}")
        print("Building vocabulary from dataset...")
        vocab.extract_tokens_from_jsonl(args.jsonl_path)
        vocab.build_vocabulary()
        vocab.save_vocabulary(args.vocab_path)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        jsonl_path=args.jsonl_path,
        image_dir=args.image_dir,
        vocab=vocab,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("Creating VAE model...")
    model = VAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        hidden_dims=args.hidden_dims
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, epoch, args.kl_weight
        )

        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, device, args.kl_weight)

        scheduler.step()

        # Log
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'vae_best.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f})")

        # Save reconstructions periodically
        if epoch % args.save_every == 0:
            save_reconstructions(model, val_loader, device, output_dir)

    # Save final model
    checkpoint = {
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }
    torch.save(checkpoint, output_dir / 'vae_final.pt')

    # Plot training curves
    plot_training_curves(train_losses, val_losses, output_dir / 'training_curves.png')

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_recon, test_kl = validate(model, test_loader, device, args.kl_weight)
    print(f"Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})")

    # Save test reconstructions
    save_reconstructions(model, test_loader, device, output_dir, num_samples=16)

    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE for latent diffusion')

    # Data
    parser.add_argument('--jsonl_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged_1400_1600.jsonl',
                       help='Path to JSONL dataset')
    parser.add_argument('--image_dir', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/artwork_images',
                       help='Directory containing images')
    parser.add_argument('--vocab_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json',
                       help='Path to save/load vocabulary')

    # Model
    parser.add_argument('--latent_channels', type=int, default=4,
                       help='Number of latent channels')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                       default=[128, 256, 512, 512],
                       help='Hidden dimensions for encoder/decoder')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--kl_weight', type=float, default=0.00001,
                       help='Weight for KL divergence term')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--output_dir', type=str, default='./outputs/vae',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save reconstructions every N epochs')

    args = parser.parse_args()
    main(args)
