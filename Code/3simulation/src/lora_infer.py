"""
LoRA inference scaffold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _placeholder_image(path: Path, text: str, size: int = 512) -> None:
    img = Image.new("RGB", (size, size), (246, 243, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, size - 20, size - 20], outline=(120, 90, 70), width=3)
    draw.text((36, 42), text[:70], fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA inference scaffold")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-json", type=str, default=None, help="Optional prompts json list")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = ["Renaissance portrait, workshop light"] * 4
    if args.prompt_json:
        p = Path(args.prompt_json)
        if p.is_file():
            prompts = json.loads(p.read_text(encoding="utf-8"))
    for i, pr in enumerate(prompts):
        _placeholder_image(out / f"sample_{i:03d}.png", pr)

    meta = {
        "status": "dry_run",
        "message": "LoRA infer scaffold executed. Replace with real pipeline when model stack is ready.",
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "n_images": len(prompts),
    }
    (out / "lora_infer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[lora_infer] dry-run outputs written:", out)


if __name__ == "__main__":
    main()
