"""
VLM-based evaluation for generated images (API or mock).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def _mock_scores(prompt: str, image_path: Path) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    std = float(arr.std())
    entropy_proxy = float(np.std(arr.mean(axis=2)))
    h = int(hashlib.sha256((prompt + str(image_path)).encode("utf-8")).hexdigest()[:8], 16)
    jitter = (h % 1000) / 1000.0
    style_conformity = float(np.clip(0.35 + 0.35 * (std / 80.0) + 0.1 * jitter, 0.0, 1.0))
    novelty_score = float(np.clip(0.30 + 0.45 * (entropy_proxy / 75.0) + 0.1 * (1.0 - jitter), 0.0, 1.0))
    prompt_alignment = float(np.clip(0.65 + 0.2 * ((len(prompt) % 97) / 97.0), 0.0, 1.0))
    craft_score = float(np.clip(0.40 + 0.40 * (std / 85.0), 0.0, 1.0))
    return {
        "style_conformity": style_conformity,
        "novelty_score": novelty_score,
        "prompt_alignment": prompt_alignment,
        "craft_score": craft_score,
    }


def _build_vlm_prompt(prompt: str) -> str:
    return (
        "You are an art-style evaluator. "
        "Given the generated image and its text prompt, score from 0 to 1 with JSON only. "
        "Keys required: style_conformity, novelty_score, prompt_alignment, craft_score. "
        "No markdown, no extra text. "
        f"Original prompt: {prompt}"
    )


def _openai_vision_scores(
    *,
    prompt: str,
    image_path: Path,
    endpoint: str,
    model: str,
    api_key: str,
) -> dict:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_vlm_prompt(prompt)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"VLM request failed: {e}") from e

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"Invalid VLM response keys={list(body.keys())}")
    text = choices[0].get("message", {}).get("content", "").strip()
    parsed = json.loads(text)
    return {
        "style_conformity": float(parsed.get("style_conformity", np.nan)),
        "novelty_score": float(parsed.get("novelty_score", np.nan)),
        "prompt_alignment": float(parsed.get("prompt_alignment", np.nan)),
        "craft_score": float(parsed.get("craft_score", np.nan)),
    }


def run_vlm_eval(
    *,
    output_dir: Path,
    vlm_provider: str = "mock",
    vlm_model: str = "openai/gpt-4o-mini",
    vlm_endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    gen_root = output_dir / "generated"
    if not gen_root.is_dir():
        return pd.DataFrame()

    api_key = (
        os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    round_rows = []
    image_rows = []
    round_dirs = sorted([p for p in gen_root.glob("round_*") if p.is_dir()])
    total_eval = 0
    for rdir in round_dirs:
        p = rdir / "prompts_round.csv"
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8", newline="") as f:
            total_eval += len(list(csv.DictReader(f)))
    print(f"[vlm] rounds={len(round_dirs)} planned_evals={total_eval} provider={vlm_provider}")
    eval_pbar = tqdm(total=total_eval, desc="[vlm] images", unit="img")

    for rdir in round_dirs:
        try:
            round_idx = int(rdir.name.split("_")[-1])
        except Exception:
            continue
        prompts_csv = rdir / "prompts_round.csv"
        if not prompts_csv.is_file():
            continue
        with open(prompts_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        scores = []
        for row in rows:
            node_id = int(row["node_id"])
            prompt = row.get("prompt", "")
            img_path = rdir / f"node_{node_id:04d}.png"
            if not img_path.is_file():
                continue
            if vlm_provider == "openai":
                if not api_key:
                    raise RuntimeError("Missing OPENROUTER_API_KEY/OPENAI_API_KEY for VLM API.")
                one = _openai_vision_scores(
                    prompt=prompt,
                    image_path=img_path,
                    endpoint=vlm_endpoint,
                    model=vlm_model,
                    api_key=api_key,
                )
            else:
                one = _mock_scores(prompt, img_path)
            one["round"] = round_idx
            one["node_id"] = node_id
            scores.append(one)
            image_rows.append(one)
            eval_pbar.update(1)

        if scores:
            s_df = pd.DataFrame(scores)
            round_rows.append(
                {
                    "round": int(round_idx),
                    "vlm_style_conformity_mean": float(pd.to_numeric(s_df["style_conformity"], errors="coerce").mean()),
                    "vlm_novelty_score_mean": float(pd.to_numeric(s_df["novelty_score"], errors="coerce").mean()),
                    "vlm_prompt_alignment_mean": float(pd.to_numeric(s_df["prompt_alignment"], errors="coerce").mean()),
                    "vlm_craft_score_mean": float(pd.to_numeric(s_df["craft_score"], errors="coerce").mean()),
                    "vlm_eval_count": int(len(s_df)),
                }
            )
        print(f"[vlm] round={round_idx} done eval_count={len(scores)}")

    eval_pbar.close()
    image_df = pd.DataFrame(image_rows)
    if not image_df.empty:
        image_df.to_csv(output_dir / "vlm_image_scores.csv", index=False)
    round_df = pd.DataFrame(round_rows).sort_values("round")
    if not round_df.empty:
        round_df.to_csv(output_dir / "vlm_rounds.csv", index=False)
    return round_df
