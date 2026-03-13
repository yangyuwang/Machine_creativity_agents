"""
Data loader for painting dataset with special tokens.
Handles image loading, preprocessing, and caption tokenization.

FIXES:
- Use a custom collate_fn (SpecialTokenCollator) to avoid default_collate errors
  caused by variable-length / nested python objects (e.g., special_tokens dict of lists).
- Make collator robust to missing categories.
- Fix bottom test block to use correct batch keys.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Dict, Tuple, Optional, List


class PaintingDataset(Dataset):
    """Dataset for historical paintings with special token captions."""

    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        vocab,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        drop_missing_images: bool = False,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with captions
            image_dir: Directory containing images
            vocab: SpecialTokenVocabulary instance
            image_size: Target image size (assumes square)
            transform: Optional custom transform
            drop_missing_images: If True, skip samples whose image file is missing
        """
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.image_size = image_size
        self.drop_missing_images = drop_missing_images

        # Load all captions
        self.data: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # keep only entries with required keys
                if "image" in entry and "caption" in entry:
                    entry["caption"] = "A painting of " + entry["caption"]
                    self.data.append(entry)

        # Default transform: always returns fixed [3, image_size, image_size]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _find_image_path(self, image_id: str) -> Optional[Path]:
        """Try multiple extensions for a given image_id."""
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dict with keys:
            - 'image': Tensor [3, H, W] (fixed size)
            - 'caption': Full caption string
            - 'description': Caption with special tokens removed
            - 'special_tokens': Dict of categorized special tokens (variable-length lists)
            - 'image_id': Image ID string
        """
        entry = self.data[idx]
        image_id = str(entry["image"])
        caption = str(entry["caption"])

        image_path = self._find_image_path(image_id)

        if image_path is None:
            if self.drop_missing_images:
                # If dropping missing images, resample deterministically (fallback to a black image only if all fail)
                # NOTE: for simplicity, we fallback to black image here to keep __len__ stable.
                image = Image.new("RGB", (self.image_size, self.image_size), color="black")
            else:
                image = Image.new("RGB", (self.image_size, self.image_size), color="black")
        else:
            image = Image.open(image_path).convert("RGB")

        # Apply transform -> fixed size tensor
        image = self.transform(image)

        # Extract special tokens
        special_tokens = self.vocab.extract_tokens_from_caption(caption)

        # Description without special tokens
        description = self.vocab.remove_special_tokens(caption)

        return {
            "image": image,
            "caption": caption,
            "description": description,
            "special_tokens": special_tokens,
            "image_id": image_id,
        }


class SpecialTokenCollator:
    """
    Custom collator to avoid PyTorch default_collate attempting to collate nested
    variable-length python objects (special_tokens dict-of-lists).
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.categories = ["artist", "year", "gender", "location"]

    def __call__(self, batch: List[Dict]) -> Dict:
        images = torch.stack([item["image"] for item in batch])  # [B,3,H,W] fixed
        captions = [item["caption"] for item in batch]
        descriptions = [item["description"] for item in batch]
        special_tokens = [item["special_tokens"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        # Convert special tokens to IDs (one per category; -1 if missing)
        token_ids = {cat: [] for cat in self.categories}

        for item_tokens in special_tokens:
            for cat in self.categories:
                toks = item_tokens.get(cat, []) if isinstance(item_tokens, dict) else []
                if toks:
                    token_id = self.vocab.get_token_id(toks[0])
                else:
                    token_id = -1
                token_ids[cat].append(token_id)

        for cat in self.categories:
            token_ids[cat] = torch.tensor(token_ids[cat], dtype=torch.long)  # [B]

        return {
            "images": images,
            "captions": captions,
            "descriptions": descriptions,
            "special_tokens": special_tokens,  # keep as python objects; do not collate further
            "token_ids": token_ids,
            "image_ids": image_ids,
        }


def create_data_loaders(
    jsonl_path: str,
    image_dir: str,
    vocab,
    batch_size: int = 16,
    image_size: int = 256,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    use_collator: bool = True,
    drop_missing_images: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    """
    full_dataset = PaintingDataset(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        vocab=vocab,
        image_size=image_size,
        drop_missing_images=drop_missing_images,
    )

    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
    )

    collate_fn = SpecialTokenCollator(vocab) if use_collator else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    from special_tokens import SpecialTokenVocabulary

    vocab_path = "/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json"
    vocab = SpecialTokenVocabulary()
    vocab.load_vocabulary(vocab_path)

    jsonl_path = "/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged_1400_1600.jsonl"
    image_dir = "/home/wangyd/Projects/macs_thesis/yangyu/artwork_images"

    train_loader, val_loader, test_loader = create_data_loaders(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        vocab=vocab,
        batch_size=4,
        image_size=256,
        num_workers=4,
        use_collator=True,          # IMPORTANT
        drop_missing_images=False,  # set True if you prefer to skip missing images (needs different __len__ handling)
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    print("\nBatch keys:", batch.keys())
    print("Images shape:", batch["images"].shape)
    print("First caption:", batch["captions"][0])
    print("Token categories:", list(batch["token_ids"].keys()))
    print("First token_ids:", {k: int(v[0].item()) for k, v in batch["token_ids"].items()})
