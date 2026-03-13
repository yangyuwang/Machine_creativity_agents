"""
Visualization utilities for interpretability analysis.
Includes GIF generation, attention maps, and activation visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
import imageio
from pathlib import Path


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image array.

    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]

    Returns:
        Numpy array [H, W, C] in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    return img


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 4,
    titles: Optional[List[str]] = None
):
    """
    Save a grid of images.

    Args:
        images: Tensor [B, C, H, W]
        save_path: Path to save image
        nrow: Number of images per row
        titles: Optional titles for each image
    """
    batch_size = images.shape[0]
    ncol = (batch_size + nrow - 1) // nrow

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3))
    if ncol == 1 and nrow == 1:
        axes = np.array([[axes]])
    elif ncol == 1:
        axes = axes[np.newaxis, :]
    elif nrow == 1:
        axes = axes[:, np.newaxis]

    for idx in range(batch_size):
        row = idx // nrow
        col = idx % nrow
        ax = axes[row, col]

        img = tensor_to_image(images[idx])
        ax.imshow(img)
        ax.axis('off')

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)

    # Turn off unused subplots
    for idx in range(batch_size, ncol * nrow):
        row = idx // nrow
        col = idx % nrow
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_interpolation_gif(
    images: List[np.ndarray],
    save_path: str,
    duration: float = 0.1,
    loop: int = 0
):
    """
    Create GIF from list of images.

    Args:
        images: List of numpy arrays [H, W, C]
        save_path: Path to save GIF
        duration: Duration per frame in seconds
        loop: Number of loops (0 = infinite)
    """
    imageio.mimsave(
        save_path,
        images,
        duration=duration,
        loop=loop
    )


def visualize_attention_map(
    attention_weights: torch.Tensor,
    save_path: str,
    image: Optional[torch.Tensor] = None,
    tokens: Optional[List[str]] = None
):
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_heads, H*W, seq_len] or [H*W, seq_len]
        save_path: Path to save visualization
        image: Optional image to overlay attention on [C, H, W]
        tokens: Optional token labels for columns
    """
    if attention_weights.dim() == 3:
        # Average over heads
        attention_weights = attention_weights.mean(dim=0)

    attn = attention_weights.cpu().numpy()
    H_W = attn.shape[0]
    H = W = int(np.sqrt(H_W))

    fig, axes = plt.subplots(1, min(attn.shape[1], 8), figsize=(20, 3))
    if attn.shape[1] == 1:
        axes = [axes]

    for idx in range(min(attn.shape[1], 8)):
        attn_map = attn[:, idx].reshape(H, W)

        if image is not None:
            img_np = tensor_to_image(image)
            axes[idx].imshow(img_np, alpha=0.5)
            axes[idx].imshow(attn_map, alpha=0.5, cmap='hot')
        else:
            axes[idx].imshow(attn_map, cmap='hot')

        axes[idx].axis('off')
        if tokens and idx < len(tokens):
            axes[idx].set_title(tokens[idx], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_activation_patching_results(
    original_images: torch.Tensor,
    patched_images: List[torch.Tensor],
    layer_names: List[str],
    save_path: str,
    attribute_name: str = ""
):
    """
    Visualize activation patching results across layers.

    Args:
        original_images: Original generated images [B, C, H, W]
        patched_images: List of patched images, one per layer
        layer_names: Names of patched layers
        save_path: Path to save visualization
        attribute_name: Name of attribute being patched
    """
    num_layers = len(patched_images)
    batch_size = original_images.shape[0]

    fig, axes = plt.subplots(batch_size, num_layers + 1, figsize=((num_layers + 1) * 3, batch_size * 3))

    if batch_size == 1:
        axes = axes[np.newaxis, :]

    for b in range(batch_size):
        # Original image
        img = tensor_to_image(original_images[b])
        axes[b, 0].imshow(img)
        axes[b, 0].axis('off')
        if b == 0:
            axes[b, 0].set_title("Original", fontsize=12)

        # Patched images
        for l in range(num_layers):
            img = tensor_to_image(patched_images[l][b])
            axes[b, l + 1].imshow(img)
            axes[b, l + 1].axis('off')
            if b == 0:
                axes[b, l + 1].set_title(f"Patch {layer_names[l]}", fontsize=10)

    plt.suptitle(f"Activation Patching: {attribute_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_neuron_importance(
    neuron_scores: np.ndarray,
    neuron_indices: np.ndarray,
    layer_names: List[str],
    save_path: str,
    attribute_name: str = "",
    top_k: int = 20
):
    """
    Plot top-k most important neurons for an attribute.

    Args:
        neuron_scores: Importance scores [num_neurons]
        neuron_indices: Neuron indices [num_neurons]
        layer_names: Layer name for each neuron
        save_path: Path to save plot
        attribute_name: Name of attribute
        top_k: Number of top neurons to plot
    """
    # Sort by importance
    sorted_indices = np.argsort(neuron_scores)[::-1][:top_k]
    top_scores = neuron_scores[sorted_indices]
    top_neuron_ids = neuron_indices[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(top_k), top_scores)

    # Color by layer
    unique_layers = list(set(layer_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))
    layer_to_color = {layer: colors[i] for i, layer in enumerate(unique_layers)}

    for i, (bar, neuron_id) in enumerate(zip(bars, top_neuron_ids)):
        layer = layer_names[neuron_id]
        bar.set_color(layer_to_color[layer])

    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"Neuron {neuron_indices[i]}" for i in sorted_indices])
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_k} Neurons for {attribute_name}", fontsize=14)
    ax.invert_yaxis()

    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=layer_to_color[layer], label=layer)
                      for layer in unique_layers]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_steering_comparison_gif(
    images_by_alpha: List[Tuple[float, torch.Tensor]],
    save_path: str,
    attribute_name: str = "",
    duration: float = 0.2
):
    """
    Create GIF showing steering effect at different alpha values.

    Args:
        images_by_alpha: List of (alpha, image_tensor) tuples
        save_path: Path to save GIF
        attribute_name: Name of attribute being steered
        duration: Duration per frame
    """
    frames = []

    for alpha, image_tensor in images_by_alpha:
        # Convert to numpy
        img = tensor_to_image(image_tensor)

        # Add text overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        ax.text(
            0.5, 0.05,
            f"{attribute_name}\nα = {alpha:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            color='white',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        plt.tight_layout()

        # Convert plot to image
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (H, W, 4)
        frame = frame[:, :, :3]  # drop alpha -> (H, W, 3)
        frames.append(frame)
        plt.close(fig)

    # Create GIF
    imageio.mimsave(save_path, frames, duration=duration, loop=0)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label='Training Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization utilities...")

    # Test tensor to image
    test_tensor = torch.randn(1, 3, 256, 256)
    img = tensor_to_image(test_tensor)
    print(f"Tensor to image: {img.shape}")

    # Test image grid
    test_images = torch.randn(8, 3, 128, 128)
    save_image_grid(test_images, '/tmp/test_grid.png', nrow=4)
    print("Saved test grid")

    # Test GIF creation
    frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(10)]
    create_interpolation_gif(frames, '/tmp/test.gif')
    print("Created test GIF")

    print("All tests passed!")
