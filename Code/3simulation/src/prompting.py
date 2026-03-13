"""
Prompt construction utilities for generation stage.
"""
from __future__ import annotations

import random
from typing import Dict


def strategy_type(d_center: float, d_cut: float) -> str:
    return "conform" if d_center <= d_cut else "diff"


def institution_tags(mode: str) -> str:
    if mode == "norm_only":
        return "church guild workshop discipline, canonical iconography, conservative composition"
    if mode == "dual_patronage":
        return "church guild + court civic patronage, mixed canon and novelty, inter-city motif exchange"
    return "random patronage baseline, heterogeneous workshops, uneven style pressure"


def strategy_tags(stype: str) -> str:
    if stype == "conform":
        return "mainstream renaissance style, geometric balance, stable lighting"
    return "differentiated style, bolder contrast, uncommon motif variation"


def capital_tags(S: float, M: float, I: float) -> str:
    # Coarse buckets keep prompts readable and reproducible.
    s_tag = "high symbolic capital" if S >= 2.0 else "moderate symbolic capital"
    m_tag = "high market visibility" if M >= 2.0 else "moderate market visibility"
    i_tag = "high influence" if I >= 2.0 else "limited influence"
    return f"{s_tag}, {m_tag}, {i_tag}"


def build_prompt(
    *,
    mode: str,
    round_idx: int,
    node_id: int,
    strategy: str,
    S: float,
    M: float,
    I: float,
    rng: random.Random,
) -> str:
    subject_pool = [
        "religious narrative scene",
        "civic procession in an italian city",
        "portrait in workshop interior",
        "mythological allegory with architecture",
        "merchant patron family group portrait",
    ]
    subject = rng.choice(subject_pool)
    return (
        "Renaissance painting, oil on panel aesthetic, "
        f"round {round_idx}, node {node_id}, {subject}, "
        f"{institution_tags(mode)}, {strategy_tags(strategy)}, {capital_tags(S, M, I)}, "
        "museum quality, high detail."
    )


def prompt_record(
    *,
    round_idx: int,
    node_id: int,
    strategy: str,
    prompt: str,
) -> Dict[str, str]:
    return {
        "round": str(round_idx),
        "node_id": str(node_id),
        "strategy": strategy,
        "prompt": prompt,
    }
