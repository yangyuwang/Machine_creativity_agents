"""
Lightweight LoRA training scaffold.

This is intentionally minimal and safe:
- If diffusers/peft stack is unavailable, emit a dry-run metadata file.
- Keeps phase-2 wiring reproducible without forcing heavy dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA training scaffold")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "dry_run",
        "message": "LoRA scaffold executed. Integrate diffusers/peft trainer for real training.",
        "data_dir": str(Path(args.data_dir).resolve()),
        "round": int(args.round),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
    }
    (out / "lora_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[lora_train] dry-run metadata written:", out / "lora_train_meta.json")


if __name__ == "__main__":
    main()
