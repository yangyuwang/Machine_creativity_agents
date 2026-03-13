"""
Round-based image generation stage.

Phase-1 default uses a deterministic mock generator (Pillow) for reliability.
Optional API provider hook is included for real backend integration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
import base64
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .prompting import build_prompt, prompt_record, strategy_type


def _is_llava_name(name: str) -> bool:
    return "llava" in str(name or "").strip().lower()


def _is_llava_placeholder_model(name: str) -> bool:
    return str(name or "").strip().lower() in {"llava", "llava-v1", "llava-1.5", "llava-1.6"}


def _enforce_llava_strict(*, provider: str, model: str, stage: str) -> None:
    p = str(provider or "").strip().lower()
    m = str(model or "").strip().lower()
    if p != "llava" and not _is_llava_name(m):
        raise ValueError(
            f"{stage} strict LLaVA mode requires provider='llava' "
            f"or model name containing 'llava'. got provider={provider!r}, model={model!r}"
        )
    if _is_llava_placeholder_model(model):
        raise ValueError(
            f"{stage} strict LLaVA mode requires a concrete model id, not placeholder {model!r}. "
            "Example format: vendor/model-name-with-llava"
        )


def _decode_data_url(data_url: str) -> bytes | None:
    if not isinstance(data_url, str):
        return None
    m = re.match(r"^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$", data_url)
    if not m:
        return None
    import base64
    return base64.b64decode(m.group(1))


def _center_distance(embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float64)
    emb = np.nan_to_num(emb, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    emb = emb / norms
    center = emb.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-8)
    cos_sim = np.clip(emb @ center, -1.0, 1.0)
    return (1.0 - cos_sim).astype(np.float64)


def _parse_decode_response_text(text: object, max_tags: int) -> tuple[str, list[str]]:
    """
    Parse VLM decode output robustly:
    1) strict JSON payload with caption/tags
    2) JSON object embedded in free text
    3) plain text fallback
    """
    raw = str(text or "").strip()
    if not raw:
        return "gallery summary unavailable", ["renaissance-inspired texture"]

    def _clean_tags(vals: object) -> list[str]:
        out: list[str] = []
        if isinstance(vals, list):
            for x in vals:
                t = str(x).strip()
                if t:
                    out.append(t)
        return out[: max(1, int(max_tags))]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cap = str(parsed.get("caption", "")).strip() or "gallery summary unavailable"
            tags = _clean_tags(parsed.get("tags"))
            if tags:
                return cap, tags
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                cap = str(parsed.get("caption", "")).strip() or "gallery summary unavailable"
                tags = _clean_tags(parsed.get("tags"))
                if tags:
                    return cap, tags
        except Exception:
            pass

    cap = raw[:300]
    tokens = re.split(r"[,;|/\n]+", raw)
    tags: list[str] = []
    for t in tokens:
        tt = re.sub(r"\s+", " ", t).strip(" -:.")
        if 3 <= len(tt) <= 42 and tt.lower() not in {"caption", "tags"}:
            tags.append(tt)
        if len(tags) >= max(1, int(max_tags)):
            break
    if not tags:
        tags = ["renaissance-inspired texture"]
    return cap, tags


def decode_with_vlm(
    image_paths: list[Path],
    *,
    provider: str = "mock",
    model: str = "",
    endpoint: str = "",
    max_tags: int = 5,
    strict_llava: bool = False,
) -> tuple[str, list[str]]:
    """
    Decode a gallery into a compact caption + tags.
    This is intentionally lightweight for closed-loop per-round calls.
    """
    if not image_paths:
        return "empty gallery", ["empty", "no-reference"]
    if strict_llava:
        _enforce_llava_strict(provider=provider, model=model, stage="decode")
    provider_norm = str(provider or "").strip().lower()
    model_norm = str(model or "").strip()
    endpoint_norm = str(endpoint or "").strip()
    # Strict LLaVA mode or explicit llava/openai provider uses API-based decode.
    if provider_norm in {"llava", "openai"} and model_norm and endpoint_norm:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY/OPENAI_API_KEY for VLM decode.")
        blocks = []
        for p in image_paths[:4]:
            if not p.is_file():
                continue
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        if not blocks:
            return "gallery contains unreadable images", ["unreadable", "fallback"]
        prompt = (
            "You are a Renaissance art observer. "
            "Summarize the gallery into JSON only with keys: "
            "caption (string), tags (array of short style tags). "
            f"Return at most {max(1, int(max_tags))} tags."
        )
        payload = {
            "model": model_norm,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}] + blocks}],
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            endpoint_norm,
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
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = "<no response body>"
            raise RuntimeError(
                f"decode_with_vlm failed: HTTP {e.code} {e.reason}. "
                f"endpoint={endpoint_norm}, model={model_norm}, body={detail[:1500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"decode_with_vlm failed: endpoint={endpoint_norm}, model={model_norm}, error={e}"
            ) from e
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(text, list):
            text = "".join(
                str(block.get("text", "")) for block in text
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            )
        cap, tags = _parse_decode_response_text(text, max_tags=max_tags)
        return cap, tags
    # Phase-1 stable decode: deterministic visual heuristics.
    bright_vals = []
    contrast_vals = []
    warm_vals = []
    for p in image_paths:
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
        if arr.size == 0:
            continue
        gray = arr.mean(axis=2)
        bright_vals.append(float(gray.mean()))
        contrast_vals.append(float(gray.std()))
        warm_vals.append(float(arr[..., 0].mean() - arr[..., 2].mean()))
    if not bright_vals:
        return "gallery contains unreadable images", ["unreadable", "fallback"]
    b = float(np.mean(bright_vals))
    c = float(np.mean(contrast_vals))
    w = float(np.mean(warm_vals))
    tags = []
    tags.append("high-key lighting" if b > 140 else "low-key lighting")
    tags.append("dramatic contrast" if c > 52 else "soft contrast")
    tags.append("warm palette" if w > 5 else "cool palette")
    tags.append("figurative composition")
    tags.append("renaissance-inspired texture")
    tags = tags[: max(1, int(max_tags))]
    caption = (
        f"gallery summary: {tags[0]}, {tags[1]}, {tags[2]}; "
        f"brightness={b:.1f}, contrast={c:.1f}"
    )
    return caption, tags


def adapt_prompt_by_type(
    *,
    artist_type: str,
    base_prompt: str,
    decoded_caption: str,
    decoded_tags: list[str],
    rng: random.Random,
) -> tuple[str, str]:
    """Rewrite prompt according to artist strategy."""
    tags = [t.strip() for t in decoded_tags if str(t).strip()]
    if not tags:
        tags = ["classical motif"]
    if artist_type == "master":
        # Master: keep identity, absorb only a small amount.
        keep = tags[:2]
        adapted = f"{base_prompt} subtle influence from {', '.join(keep)}."
        return adapted, "self_consistency"
    if artist_type == "rebel":
        # Rebel: explicitly anti-mainstream.
        anti = ", ".join(tags[:4])
        adapted = (
            f"{base_prompt} opposite of ({anti}), anti-mainstream arrangement, "
            "unexpected composition, controlled dissonance."
        )
        return adapted, "differentiation"
    # Follower (default): imitate visible mainstream.
    picked = tags[:3]
    if len(tags) > 3 and rng.random() < 0.5:
        picked.append(rng.choice(tags[3:]))
    adapted = (
        f"{base_prompt} inspired by visible gallery: {', '.join(picked)}. "
        f"reference note: {decoded_caption}"
    )
    return adapted, "imitation"


def create_image(
    *,
    prompt: str,
    out_path: Path,
    gen_engine: str,
    api_provider: str,
    api_model: str,
    api_fallback_models: list[str] | None,
    api_endpoint: str,
    api_retry_max: int,
    api_retry_backoff_sec: float,
    image_size: int,
    strict_llava: bool = False,
) -> np.ndarray:
    """Plugin-style creation interface."""
    provider_norm = str(api_provider or "").strip().lower()
    if strict_llava:
        _enforce_llava_strict(provider=provider_norm, model=api_model, stage="create")
    if gen_engine == "api" and provider_norm in {"openai", "llava"}:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY/OPENAI_API_KEY for API generation.")
        models_to_try = [api_model] + [m for m in (api_fallback_models or []) if m and m != api_model]
        arr = None
        last_err: Exception | None = None
        for model_name in models_to_try:
            for attempt in range(1, max(1, int(api_retry_max)) + 1):
                try:
                    arr = _api_render_openai(
                        prompt,
                        out_path,
                        api_model=model_name,
                        api_endpoint=api_endpoint,
                        api_key=api_key,
                    )
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max(1, int(api_retry_max)):
                        time.sleep(float(api_retry_backoff_sec) * attempt)
            if arr is not None:
                break
        if arr is None:
            raise RuntimeError(f"create_image failed all retries: {last_err}")
        return arr
    return _mock_render(prompt, out_path, size=image_size)


def _mock_render(prompt: str, out_path: Path, size: int = 512) -> np.ndarray:
    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(h)
    img = Image.new("RGB", (size, size), (245, 240, 230))
    draw = ImageDraw.Draw(img)
    for _ in range(28):
        x0 = rng.randint(0, size - 1)
        y0 = rng.randint(0, size - 1)
        x1 = min(size - 1, x0 + rng.randint(20, 220))
        y1 = min(size - 1, y0 + rng.randint(20, 220))
        color = (rng.randint(40, 220), rng.randint(30, 180), rng.randint(20, 140))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=rng.randint(1, 3))
    for _ in range(20):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        r = rng.randint(6, 48)
        color = (rng.randint(80, 240), rng.randint(80, 220), rng.randint(70, 200))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    arr = np.asarray(img).astype(np.float64)
    return arr


def _api_render_openai(
    prompt: str,
    out_path: Path,
    *,
    api_model: str,
    api_endpoint: str,
    api_key: str,
) -> np.ndarray:
    """
    OpenAI-compatible image generation with support for:
    - /images/generations style responses (b64_json or url)
    - /chat/completions image responses (OpenRouter modalities)
    """
    endpoint_norm = api_endpoint.rstrip("/")
    if endpoint_norm.endswith("/chat/completions"):
        payload_dict = {
            "model": api_model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            # OpenRouter image-capable chat models typically require image modality.
            "modalities": ["image"],
        }
    else:
        payload_dict = {
            "model": api_model,
            "prompt": prompt,
            "size": "1024x1024",
            "response_format": "b64_json",
        }
    payload = json.dumps(payload_dict).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # OpenRouter recommends app identification headers.
    if "openrouter.ai" in endpoint_norm:
        if os.environ.get("OPENROUTER_SITE_URL"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
        if os.environ.get("OPENROUTER_APP_NAME"):
            headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
    req = urllib.request.Request(
        api_endpoint,
        data=payload,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = "<no response body>"
        raise RuntimeError(
            f"API render failed: HTTP {e.code} {e.reason}. endpoint={api_endpoint}. body={detail[:1200]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API render failed: {e}") from e

    raw = None
    # OpenAI images/generations style
    if isinstance(body.get("data"), list) and body["data"]:
        data0 = body["data"][0]
        b64_json = data0.get("b64_json")
        if b64_json:
            import base64
            raw = base64.b64decode(b64_json)
        elif data0.get("url"):
            with urllib.request.urlopen(data0["url"], timeout=90) as r2:
                raw = r2.read()

    # OpenRouter chat/completions image style:
    # choices[0].message.images[0].image_url.url = data:image/...;base64,...
    if raw is None:
        try:
            msg = body["choices"][0]["message"]
            images = msg.get("images") or []
            if images:
                data_url = images[0].get("image_url", {}).get("url", "")
                raw = _decode_data_url(data_url)
        except Exception:
            raw = None

    # Some providers return image blocks in message.content list.
    if raw is None:
        try:
            msg = body["choices"][0]["message"]
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    # Shape A: {"type":"image_url","image_url":{"url":"data:image/..."}}
                    if block.get("type") == "image_url":
                        u = (block.get("image_url") or {}).get("url", "")
                        raw = _decode_data_url(u)
                    # Shape B: {"type":"output_image","image_url":"data:image/..."}
                    if raw is None and block.get("type") == "output_image":
                        u2 = block.get("image_url", "")
                        raw = _decode_data_url(u2)
                    # Shape C: {"type":"output_image","b64_json":"..."}
                    if raw is None and block.get("type") == "output_image" and block.get("b64_json"):
                        import base64
                        raw = base64.b64decode(block["b64_json"])
                    if raw is not None:
                        break
            elif isinstance(content, str):
                raw = _decode_data_url(content)
        except Exception:
            raw = None

    # Some providers keep image URL on top-level message field.
    if raw is None:
        try:
            msg = body["choices"][0]["message"]
            if isinstance(msg.get("image_url"), str):
                raw = _decode_data_url(msg["image_url"])
        except Exception:
            raw = None

    if raw is None:
        message_preview = ""
        try:
            msg_obj = body.get("choices", [{}])[0].get("message", {})
            message_preview = json.dumps(msg_obj, ensure_ascii=False)[:1200]
        except Exception:
            message_preview = "<unavailable>"
        raise RuntimeError(
            "API response has no recognized image payload. "
            f"endpoint={api_endpoint}, keys={list(body.keys())}, message_preview={message_preview}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    arr = np.asarray(Image.open(out_path).convert("RGB")).astype(np.float64)
    return arr


def _sample_nodes_for_round(nodes: pd.DataFrame, top_k: int, per_round: int, rng: random.Random) -> list[int]:
    n = len(nodes)
    if n == 0:
        return []
    k = min(max(1, top_k), n)
    ranked = nodes.sort_values("I", ascending=False).head(k)["id"].astype(int).tolist()
    if len(ranked) <= per_round:
        return ranked
    return rng.sample(ranked, per_round)


def _rounds_from_iter(obs_rounds: Iterable[int], round_every: int) -> list[int]:
    rounds = sorted({int(r) for r in obs_rounds})
    return [r for r in rounds if r % max(1, round_every) == 0]


def generate_round_images(
    *,
    output_dir: Path,
    nodes: pd.DataFrame,
    embeddings: np.ndarray,
    mode: str,
    round_idx: int,
    gen_per_round: int = 8,
    gen_top_k: int = 80,
    gen_engine: str = "api",
    api_provider: str = "mock",
    api_model: str = "gpt-image-1",
    api_fallback_models: list[str] | None = None,
    api_endpoint: str = "https://api.openai.com/v1/images/generations",
    api_retry_max: int = 3,
    api_retry_backoff_sec: float = 2.0,
    image_size: int = 512,
    visible_gallery: dict[int, list[int]] | None = None,
    artist_types: dict[int, str] | None = None,
    decode_provider: str = "mock",
    decode_model: str = "",
    decode_endpoint: str = "",
    strict_llava_create: bool = False,
    strict_llava_decode: bool = False,
    feedback_i_scale: float = 1.0,
    feedback_s_scale: float = 1.0,
    feedback_m_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Generate images for one specific round and return per-image records + summary."""
    output_dir = Path(output_dir)
    d_center = _center_distance(embeddings)
    d_cut = float(np.median(d_center))
    gen_root = output_dir / "generated"
    round_dir = gen_root / f"round_{int(round_idx):04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42 + int(round_idx))
    selected = _sample_nodes_for_round(nodes, top_k=gen_top_k, per_round=gen_per_round, rng=rng)
    rows: list[dict] = []
    rgb_stds: list[float] = []
    d_vals: list[float] = []
    strategy_counter: Counter[str] = Counter()
    decode_tag_counter: Counter[str] = Counter()
    for nid in selected:
        row = nodes.loc[nodes["id"] == nid].iloc[0]
        s_type = strategy_type(float(d_center[nid]), d_cut)
        artist_type = (
            str(artist_types.get(int(nid), "")).strip().lower()
            if artist_types is not None
            else ""
        )
        if not artist_type:
            artist_type = "follower" if s_type == "conform" else "rebel"
        local_rng = random.Random((round_idx + 1) * 1000003 + int(nid))
        base_prompt = build_prompt(
            mode=mode,
            round_idx=int(round_idx),
            node_id=int(nid),
            strategy=s_type,
            S=float(row["S"]),
            M=float(row["M"]),
            I=float(row["I"]),
            rng=local_rng,
        )
        gallery_ids = []
        if visible_gallery is not None:
            gallery_ids = [int(x) for x in visible_gallery.get(int(nid), []) if int(x) != int(nid)]
        gallery_paths = []
        for gid in gallery_ids:
            hit = nodes.loc[nodes["id"] == gid]
            if not hit.empty:
                gallery_paths.append(Path(str(hit.iloc[0]["path"])))
        decoded_caption, decoded_tags = decode_with_vlm(
            gallery_paths,
            provider=decode_provider,
            model=decode_model,
            endpoint=decode_endpoint,
            max_tags=5,
            strict_llava=strict_llava_decode,
        )
        prompt, learning_mode = adapt_prompt_by_type(
            artist_type=artist_type,
            base_prompt=base_prompt,
            decoded_caption=decoded_caption,
            decoded_tags=decoded_tags,
            rng=local_rng,
        )
        out_img = round_dir / f"node_{int(nid):04d}.png"
        arr = create_image(
            prompt=prompt,
            out_path=out_img,
            gen_engine=gen_engine,
            api_provider=api_provider,
            api_model=api_model,
            api_fallback_models=api_fallback_models,
            api_endpoint=api_endpoint,
            api_retry_max=api_retry_max,
            api_retry_backoff_sec=api_retry_backoff_sec,
            image_size=image_size,
            strict_llava=strict_llava_create,
        )
        rgb_stds.append(float(arr.std()))
        d_vals.append(float(d_center[nid]))
        rec = prompt_record(round_idx=int(round_idx), node_id=int(nid), strategy=s_type, prompt=prompt)
        rec["image_path"] = str(out_img.resolve())
        rec["artist_type"] = artist_type
        rec["learning_mode"] = learning_mode
        rec["gallery_ids"] = "|".join(str(x) for x in gallery_ids)
        rec["decoded_caption"] = decoded_caption
        rec["decoded_tags"] = "|".join(decoded_tags)
        rec["base_prompt"] = base_prompt
        rec["create_provider"] = str(api_provider)
        rec["create_model"] = str(api_model)
        rec["decode_provider"] = str(decode_provider)
        rec["decode_model"] = str(decode_model)
        # Lightweight social feedback signals for evolution update.
        rec["feedback_I"] = float(
            (1.0 if learning_mode == "imitation" else 0.7 if learning_mode == "differentiation" else 0.5)
            * max(0.0, float(feedback_i_scale))
        )
        rec["feedback_S"] = float(
            (0.6 if learning_mode == "self_consistency" else 0.3)
            * max(0.0, float(feedback_s_scale))
        )
        rec["feedback_M"] = float(
            (0.8 if learning_mode == "imitation" else 0.4)
            * max(0.0, float(feedback_m_scale))
        )
        strategy_counter[learning_mode] += 1
        for t in decoded_tags:
            decode_tag_counter[str(t).strip().lower()] += 1
        rows.append(rec)
    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        df_rows.to_csv(round_dir / "prompts_round.csv", index=False)
    else:
        with open(round_dir / "prompts_round.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "node_id", "strategy", "prompt"])
            writer.writeheader()
    summary = {
        "round": int(round_idx),
        "generated_count": len(rows),
        "prompt_unique_ratio": float(len({r["prompt"] for r in rows}) / max(1, len(rows))),
        "rgb_std_mean": float(np.mean(rgb_stds)) if rgb_stds else np.nan,
        "center_deviation_proxy": float(np.mean(d_vals)) if d_vals else np.nan,
        "share_imitation": float(strategy_counter.get("imitation", 0) / max(1, len(rows))),
        "share_differentiation": float(strategy_counter.get("differentiation", 0) / max(1, len(rows))),
        "share_self_consistency": float(strategy_counter.get("self_consistency", 0) / max(1, len(rows))),
        "decoded_top_tags": "|".join([k for k, _ in decode_tag_counter.most_common(5)]),
    }
    return df_rows, summary


def run_generation(
    *,
    output_dir: Path,
    nodes: pd.DataFrame,
    embeddings: np.ndarray,
    mode: str,
    obs_rounds: Iterable[int],
    gen_round_every: int = 5,
    gen_per_round: int = 8,
    gen_top_k: int = 80,
    gen_engine: str = "api",
    api_provider: str = "mock",
    api_model: str = "gpt-image-1",
    api_fallback_models: list[str] | None = None,
    api_endpoint: str = "https://api.openai.com/v1/images/generations",
    api_retry_max: int = 3,
    api_retry_backoff_sec: float = 2.0,
    api_seed_policy: str = "round_node",
    image_size: int = 512,
    strict_llava_create: bool = False,
) -> pd.DataFrame:
    """
    Generate round-wise images and return per-round summary DataFrame.
    """
    output_dir = Path(output_dir)
    gen_root = output_dir / "generated"
    gen_root.mkdir(parents=True, exist_ok=True)
    d_center = _center_distance(embeddings)
    d_cut = float(np.median(d_center))
    rng = random.Random(42)

    rounds = _rounds_from_iter(obs_rounds, gen_round_every)
    summary_rows = []
    failure_rows = []
    feedback = np.zeros(len(nodes), dtype=np.float64)
    total_targets = 0
    for rr in rounds:
        sel = _sample_nodes_for_round(nodes, top_k=gen_top_k, per_round=gen_per_round, rng=random.Random(42 + int(rr)))
        total_targets += len(sel)
    print(f"[gen] rounds={len(rounds)} planned_images={total_targets} engine={gen_engine}/{api_provider}")

    img_pbar = tqdm(total=total_targets, desc="[gen] images", unit="img")

    for r in rounds:
        round_dir = gen_root / f"round_{int(r):04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        selected = _sample_nodes_for_round(nodes, top_k=gen_top_k, per_round=gen_per_round, rng=rng)
        print(f"[gen] round={int(r)} selected={len(selected)}")
        prompt_rows = []
        rgb_stds = []
        d_vals = []

        for nid in selected:
            row = nodes.loc[nodes["id"] == nid].iloc[0]
            s_type = strategy_type(float(d_center[nid]), d_cut)
            local_rng = random.Random((r + 1) * 1000003 + int(nid))
            prompt = build_prompt(
                mode=mode,
                round_idx=int(r),
                node_id=int(nid),
                strategy=s_type,
                S=float(row["S"]),
                M=float(row["M"]),
                I=float(row["I"]),
                rng=local_rng,
            )
            prompt_rows.append(prompt_record(round_idx=int(r), node_id=int(nid), strategy=s_type, prompt=prompt))
            out_img = round_dir / f"node_{int(nid):04d}.png"

            if gen_engine == "api" and str(api_provider or "").strip().lower() in {"openai", "llava"}:
                api_key = os.environ.get("OPENAI_API_KEY", "").strip()
                if not api_key:
                    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
                if not api_key:
                    key_file = Path.home() / ".openai_api_key"
                    if key_file.is_file():
                        api_key = key_file.read_text(encoding="utf-8").strip()
                if not api_key:
                    raise RuntimeError(
                        "Missing API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY "
                        "or ~/.openai_api_key."
                    )
                if strict_llava_create:
                    _enforce_llava_strict(provider=api_provider, model=api_model, stage="create")
                models_to_try = [api_model]
                for m in (api_fallback_models or []):
                    mm = str(m).strip()
                    if mm and mm not in models_to_try:
                        models_to_try.append(mm)
                arr = None
                last_err: Exception | None = None
                for model_name in models_to_try:
                    for attempt in range(1, max(1, int(api_retry_max)) + 1):
                        try:
                            arr = _api_render_openai(
                                prompt,
                                out_img,
                                api_model=model_name,
                                api_endpoint=api_endpoint,
                                api_key=api_key,
                            )
                            if model_name != api_model or attempt > 1:
                                print(
                                    f"[gen] recovered node={int(nid)} round={int(r)} "
                                    f"model={model_name} attempt={attempt}"
                                )
                            break
                        except Exception as e:
                            last_err = e
                            print(
                                f"[gen][warn] node={int(nid)} round={int(r)} "
                                f"model={model_name} attempt={attempt} failed: {e}"
                            )
                            if attempt < max(1, int(api_retry_max)):
                                time.sleep(float(api_retry_backoff_sec) * attempt)
                    if arr is not None:
                        break
                if arr is None:
                    failure_rows.append({
                        "round": int(r),
                        "node_id": int(nid),
                        "model_primary": api_model,
                        "models_tried": "|".join(models_to_try),
                        "error": str(last_err)[:2000] if last_err else "unknown",
                    })
                    print(f"[gen][skip] round={int(r)} node={int(nid)} all retries failed, continue.")
                    img_pbar.update(1)
                    continue
            else:
                # Stable default for phase-1 quick closure.
                arr = _mock_render(prompt, out_img, size=image_size)

            rgb_stds.append(float(arr.std()))
            d_vals.append(float(d_center[nid]))
            feedback[int(nid)] += 1.0
            img_pbar.update(1)

        with open(round_dir / "prompts_round.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "node_id", "strategy", "prompt"])
            writer.writeheader()
            writer.writerows(prompt_rows)

        unique_ratio = len({p["prompt"] for p in prompt_rows}) / max(1, len(prompt_rows))
        summary_rows.append({
            "round": int(r),
            "generated_count": len(prompt_rows),
            "prompt_unique_ratio": float(unique_ratio),
            "rgb_std_mean": float(np.mean(rgb_stds)) if rgb_stds else np.nan,
            "center_deviation_proxy": float(np.mean(d_vals)) if d_vals else np.nan,
        })
        print(f"[gen] round={int(r)} done saved={len(prompt_rows)} -> {round_dir}")

    img_pbar.close()
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "generation_rounds.csv", index=False)
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(output_dir / "generation_failures.csv", index=False)
        print(f"[gen] failures logged -> {output_dir / 'generation_failures.csv'} ({len(failure_rows)})")
    # Save normalized node-level feedback for optional next-run coupling.
    if feedback.sum() > 0:
        feedback = feedback / feedback.max()
    np.save(output_dir / "generation_feedback.npy", feedback.astype(np.float32))
    return summary_df
