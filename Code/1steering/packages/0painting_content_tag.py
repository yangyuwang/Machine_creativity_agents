#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
from typing import Dict, Any, Optional, Tuple, List


# -----------------------------
# Paths (edit if needed)
# -----------------------------
IN_JSONL = "/home/wangyd/Projects/macs_thesis/yangyu/painting_content.jsonl"
OUT_JSONL = "/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged.jsonl"

DEMO_JSON = "/home/wangyd/Projects/macs_thesis/yangyu/artist_demographics/demographic_information_modified.json"
ARTWORK_CSV = "/home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv"


# -----------------------------
# Helpers
# -----------------------------
_slug_cleanup_re = re.compile(r"[^a-z0-9\-]+")

def slugify_token(x: str) -> str:
    x = (x or "").strip().lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[\s_/]+", "-", x)
    x = re.sub(r"-{2,}", "-", x)
    x = _slug_cleanup_re.sub("", x)
    x = x.strip("-")
    return x or "unknown"

def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and re.fullmatch(r"-?\d+(\.\d+)?", x.strip()):
            x = float(x)
        return int(x)
    except Exception:
        return None

def pick_time_matched_location(artist_demo: Dict[str, Any], year: Optional[int]) -> Optional[str]:
    if year is None:
        return artist_demo.get("birth_place")

    def pick_from_ranges(ranges: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not ranges:
            return None
        candidates = []
        for r in ranges:
            loc = r.get("location")
            if not loc:
                continue
            sy_i = safe_int(r.get("start_year"))
            ey_i = safe_int(r.get("end_year"))
            if sy_i is None:
                sy_i = -10**9
            if ey_i is None:
                ey_i = 10**9
            if sy_i <= year <= ey_i:
                candidates.append((sy_i, ey_i, loc))

        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        return candidates[0][2]

    loc = pick_from_ranges(artist_demo.get("residences"))
    if loc:
        return loc
    loc = pick_from_ranges(artist_demo.get("visits"))
    if loc:
        return loc
    return artist_demo.get("birth_place")

def split_location(loc: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not loc:
        return None, None
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[-1]

def normalize_gender(g: Optional[str]) -> str:
    g = (g or "").strip().lower()
    if g in {"male", "m"}:
        return "male"
    if g in {"female", "f"}:
        return "female"
    return "unknown"


# -----------------------------
# Load mappings
# -----------------------------
def load_demo(demo_path: str) -> Dict[str, Any]:
    with open(demo_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_artwork_index(csv_path: str) -> Dict[str, Tuple[str, Optional[int]]]:
    """
    Build image_id -> (artist_name, year_int)
    Uses columns: image_n, Artist_name, Year
    """
    idx: Dict[str, Tuple[str, Optional[int]]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image_n")
            if img is None or img == "":
                continue
            img_id = str(safe_int(img)) if safe_int(img) is not None else str(img).strip()

            artist = (row.get("Artist_name") or "").strip()
            if not artist:
                continue

            year = safe_int(row.get("Year"))
            idx[img_id] = (artist, year)
    return idx


# -----------------------------
# Main transform
# -----------------------------
def main() -> None:
    demo = load_demo(DEMO_JSON)
    artwork_idx = load_artwork_index(ARTWORK_CSV)

    stats = {
        "total_input_lines": 0,
        "skipped_no_artist_in_csv": 0,
        "skipped_artist_not_in_demo_json": 0,  # <-- your requested filter
        "missing_location": 0,
        "written": 0,
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
                continue

            image_id = str(obj.get("image", "")).strip()
            caption = (obj.get("caption") or "").strip()

            if image_id not in artwork_idx:
                stats["skipped_no_artist_in_csv"] += 1
                continue

            artist_name, year = artwork_idx[image_id]

            # IMPORTANT: demographic json keys are expected to match Artist_name exactly
            # If your keys are slugified (e.g., "andrea-del-verrochio"), change this line accordingly.
            if artist_name not in demo:
                stats["skipped_artist_not_in_demo_json"] += 1
                continue

            artist_demo = demo[artist_name]
            gender_tag = normalize_gender(artist_demo.get("gender"))

            loc_city_tag = None
            loc_country_tag = None
            loc_str = pick_time_matched_location(artist_demo, year)
            city, country = split_location(loc_str)
            if city:
                loc_city_tag = slugify_token(city)
            if country:
                loc_country_tag = slugify_token(country)
            if not city and not country:
                stats["missing_location"] += 1

            tags = [f"<artist_{slugify_token(artist_name)}>"]
            if year is not None:
                tags.append(f"<year_{year}>")
            tags.append(f"<gender_{gender_tag}>")
            if loc_city_tag:
                tags.append(f"<loc_{loc_city_tag}>")
            if loc_country_tag:
                tags.append(f"<loc_{loc_country_tag}>")

            out_obj = {
                "image": image_id,
                "caption": (caption + " " + " ".join(tags)).strip(),
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print("Done.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
