"""
Extract CLIP image embeddings; L2-normalize. Saves embeddings.npy and path list.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    raise ImportError("Please install open_clip_torch: pip install open_clip_torch")


def load_images(paths: List[Path], size: int = 224) -> torch.Tensor:
    """Load and preprocess images to tensor [N, 3, H, W] (CLIP-style norm)."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
        tensors.append(arr)
    return torch.from_numpy(np.stack(tensors)).permute(0, 3, 1, 2)


def extract_embeddings(
    image_paths: List[Path],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract CLIP image embeddings and L2-normalize. Returns [N, D] float32."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    N = len(image_paths)
    embeddings = []
    dtype = next(model.parameters()).dtype
    for start in tqdm(range(0, N, batch_size), desc="Embed"):
        batch_paths = image_paths[start : start + batch_size]
        x = load_images(batch_paths)
        x = x.to(device=device, dtype=dtype)
        with torch.no_grad():
            feat = model.encode_image(x)
        feat = feat.cpu().numpy().astype(np.float32)
        # Replace NaN/Inf to avoid matmul overflow and downstream segfaults
        feat = np.nan_to_num(feat, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        norm = np.where(norm > 1e-8, norm, 1.0)
        feat = feat / norm
        embeddings.append(feat)
    out = np.vstack(embeddings).astype(np.float32)
    # Final sanitize: no NaN/Inf, re-normalize so cosine math is safe
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    out = (out / norms).astype(np.float32)
    return out


def run(
    images_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load images from folder, extract embeddings, save. Returns (embeddings, paths)."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    paths = [str(p.resolve()) for p in paths]
    with open(output_dir / "image_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(paths))

    embeddings = extract_embeddings(
        [Path(p) for p in paths],
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    np.save(output_dir / "embeddings.npy", embeddings)
    return embeddings, paths
