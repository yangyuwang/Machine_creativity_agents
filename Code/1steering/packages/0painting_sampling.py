#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
from typing import Dict, Optional, Tuple


# -----------------------------
# Paths (edit if needed)
# -----------------------------
IN_JSONL = "/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged.jsonl"  # your previous output
OUT_JSONL = "/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged_1400_1600.jsonl"

ARTWORK_CSV = "/home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv"

YEAR_MIN = 1400
YEAR_MAX = 1600


# -----------------------------
# Helpers
# -----------------------------
def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
            # allow "1500.0" etc.
            if re.fullmatch(r"-?\d+(\.\d+)?", x):
                x = float(x)
        return int(x)
    except Exception:
        return None

YEAR_TAG_RE = re.compile(r"<year_(\d{3,4})>")

def year_from_caption_tag(caption: str) -> Optional[int]:
    if not caption:
        return None
    m = YEAR_TAG_RE.search(caption)
    if not m:
        return None
    return safe_int(m.group(1))


def load_year_index(csv_path: str) -> Dict[str, Optional[int]]:
    """
    image_n -> Year (int or None)
    """
    idx: Dict[str, Optional[int]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image_n")
            if img is None or str(img).strip() == "":
                continue
            img_id = str(safe_int(img)) if safe_int(img) is not None else str(img).strip()
            idx[img_id] = safe_int(row.get("Year"))
    return idx


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    year_idx = load_year_index(ARTWORK_CSV)

    stats = {
        "total_input_lines": 0,
        "kept": 0,
        "dropped_out_of_range": 0,
        "dropped_missing_year": 0,
        "dropped_missing_image_in_csv": 0,
        "used_year_from_caption_tag": 0,
        "used_year_from_csv": 0,
    }

    with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            stats["total_input_lines"] += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
                continue

            image_id = str(obj.get("image", "")).strip()
            caption = obj.get("caption", "")

            # Prefer year from caption tag if present (since it's already in your output),
            # but validate/backup with CSV when missing.
            y = year_from_caption_tag(caption)
            if y is not None:
                stats["used_year_from_caption_tag"] += 1
            else:
                if image_id not in year_idx:
                    stats["dropped_missing_image_in_csv"] += 1
                    continue
                y = year_idx.get(image_id)
                stats["used_year_from_csv"] += 1

            if y is None:
                stats["dropped_missing_year"] += 1
                continue

            if YEAR_MIN <= y <= YEAR_MAX:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            else:
                stats["dropped_out_of_range"] += 1

    print("Done.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
