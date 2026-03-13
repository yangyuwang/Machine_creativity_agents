# 1400-1600 Renaissance Style Evolution MVP

This MVP uses about 200 images to simulate art diffusion from 1400 to 1600.  
It models two coexisting patronage channels: **Church/Guild (normative)** and **Court/Bourgeois (competitive/novelty)**, and observes "canonization" vs. "school formation" through the citation network.

## 1) Choose a pipeline first: env_only vs env_plus_artist

Pick one pipeline before running commands.

### A. `env_only` (environment only, no artist behavioral closed loop)

Pipeline:

`exposure/citation/game dynamics -> metrics -> plots`

Key switches:

- `feedback_closed_loop = false`
- `gen_engine = none`

Run:

```bash
cd "/Users/1532016078qq.com/Desktop/MACS 37005/pj/agentic_simulation"
pip install -r requirements.txt

# Single run (environment only)
python main.py --config config_env_only.json --mode dual_patronage --viz-engine matplotlib
# Equivalent form (explicit pipeline dispatch):
# python main.py --pipeline env_only --config config_env_only.json --mode dual_patronage --viz-engine matplotlib
```

Recommended output paths:

- `runs_env_only/outputs_*`
- Current partitioned root: `runs_env_only/`

### B. `env_plus_artist` (environment + artist behavioral closed loop)

Pipeline:

`exposure -> decode -> adapt -> create -> citation/update -> next round`

Key switches:

- `feedback_closed_loop = true`
- `gen_engine = api` (or `lora`)

Run:

```bash
cd "/Users/1532016078qq.com/Desktop/MACS 37005/pj/agentic_simulation"
pip install -r requirements.txt

# Single closed-loop run (environment + artists)
python main.py --config config_env_plus_artist.json --mode dual_patronage --gen-engine api --viz-engine matplotlib
# Equivalent form (explicit pipeline dispatch):
# python main.py --pipeline env_plus_artist --config config_env_plus_artist.json --mode dual_patronage --gen-engine api --viz-engine matplotlib

# Batch closed-loop run (3x3: three environments x three artist strategies)
python main.py --pipeline batch --config config_env_plus_artist.json --batch-T 40 --batch-spawn-round-every 2 --batch-spawn-per-round 4 --batch-gen-top-k 120

# Batch environment-only run (3x1: all three environments with automatic summary)
python main.py --pipeline batch --batch-kind env_only --config config_env_only.json --batch-T 40
```

Recommended output paths:

- `runs_env_plus_artist/outputs_*`
- `runs_env_plus_artist/batch_runs/*`

### C. Output naming system (automatic)

Single runs (`main.py`) are automatically saved by mechanism:

- `runs_env_only/single_runs/{timestamp}-{mode}-t{T}`
- `runs_env_plus_artist/single_runs/{timestamp}-{mode}-{strategy}-t{T}-sp{spawn_every}x{spawn_per}-gk{gen_top_k}`

Batch runs (`--pipeline batch`) automatically create batch directories:

- `runs_env_plus_artist/batch_runs/batch-{grid}-{timestamp}-{config}-t{T}-sp{spawn_every}x{spawn_per}-gk{gen_top_k}`
- Sub-experiment directory naming:
  - `e{env_idx}-s{strategy_idx}-{mode}-{strategy}-rr{rebel_ratio}`

Both single and batch directories include manifest files:

- `run_manifest.json` (single run)
- `batch_manifest.json` (batch run)

## 2) Basic run (generic entry)

```bash
cd "/Users/1532016078qq.com/Desktop/MACS 37005/pj/agentic_simulation"
pip install -r requirements.txt
python main.py --mode dual_patronage
```

Available modes:

```bash
python main.py --mode baseline_random
python main.py --mode norm_only
python main.py --mode dual_patronage
```

## 3) Mapping three institutions to the 1400-1600 narrative

- `baseline_random`: no institutional filtering; candidates are influence-weighted and then randomly selected as the control group.  
- `norm_only`: Church/Guild dominated; only the normative channel is active (convergence, inheritance, local workshop copying).  
- `dual_patronage`: Church/Guild + Court/Bourgeois coexist; norm and competition run in parallel (tension between canon and innovation).

Unified per-round core variables (same names across all modes):

- `candidates_ids`: influence-weighted sampled candidate set
- `selected_ids`: final selected set for this round (**the only set used for** influence updates, citation generation, and payoff calculation)
- `selected_norm_ids`: selected by normative channel (empty in random mode)
- `selected_comp_ids`: selected by competition channel (empty in random/norm_only)
- `citations_added`: new citation edges `(ref, node)` (debug and metrics use the same definition)

## 4) Unified mechanism (selection + citation + game dynamics)

- **Selection**
  - `baseline_random`: `selected_ids` is a random fixed-ratio sample from `candidates_ids`
  - `norm_only`: top by `score_norm`, with `selected_ids = selected_norm_ids`
  - `dual_patronage`: normative and competitive channels run in parallel; competitive channel must satisfy quality floor `score_norm >= tau`
- **Citation (at least 2, at most 3 per selected node)**
  - `exemplar`: highest-influence node in kNN
  - `peer`: random peer in kNN
  - `global`: append a global citation with probability `p_global_*` (simulates inter-city diffusion/prints)
- **Game payoff (fixes the all-zero issue)**
  - Strategy groups: by `d_center` quantiles into `conform(q20)` / `diff(q80)` / `mid`
  - Instant payoff `u_i(t)`:
    - hit in `norm` channel: `+game_aS`
    - hit in `comp` channel: `+game_aM`
    - hit in `baseline_random`: `+game_aR`
  - Report within-strategy means: `payoff_*_mean` and `payoff_*_cum_mean`

## 5) Controllable switches (README highlights)

Key parameters in `config.json`:

- `p_global_norm` (low by default, inheritance within local guild networks)
- `p_global_dual` (high by default, stronger inter-city circulation)
- `p_global_random` (inter-city spread in random control)
- `tau` (quality floor for competition channel)
- `game_aS`, `game_aM`, `game_aR` (three payoff weights)

Interpretation:

- Under low `p_global_norm`, `norm_only` is more likely to produce localized school formation (modularity may rise).
- If `p_global_norm` is increased in `norm_only` (or exemplar canonical guidance is strengthened), unified canonization may appear (modularity drops).
- Under higher `p_global_dual`, `dual_patronage` often shows higher cross-cluster citation and lower modularity.

## 6) New/updated outputs

Key fields in `metrics.csv`:

- `clusters_all`, `clusters_visible`
- `cross_cluster_citation_rate`
- `share_conform_topK`
- `payoff_conform_mean`, `payoff_diff_mean`
- `payoff_conform_cum_mean`, `payoff_diff_cum_mean`
- `n_selected`, `n_selected_norm`, `n_selected_comp`
- `generated_count`, `prompt_unique_ratio`, `rgb_std_mean`, `center_deviation_proxy` (when generation stage is enabled)
- `vlm_style_conformity_mean`, `vlm_novelty_score_mean`, `vlm_prompt_alignment_mean`, `vlm_craft_score_mean` (when VLM evaluation is enabled)

New plots:

- `plots/game_dynamics.png`
- `plots/citation_cross_rate.png`
- `plots/clusters_all_vs_visible.png`
- `plots/institution_vs_generation.png` (when generation stage is enabled)
- `plots/vlm_dynamics.png` (when VLM evaluation is enabled)

Generation-stage artifacts (when `gen_engine` is enabled):

- `generated/round_xxxx/*.png`
- `generated/round_xxxx/prompts_round.csv`
- `generation_rounds.csv`
- `generation_feedback.npy`
- `generated_lora/*` (`gen_engine=lora` scaffold outputs)
- `vlm_image_scores.csv` (VLM scores per image)
- `vlm_rounds.csv` (mean VLM score per round)

## 7) Generation and coupling (Phase1/2/3)

Run examples:

```bash
# Simulation + metrics + plotting only
python main.py --mode dual_patronage --viz-engine matplotlib

# Phase-1: API generation (default mock gives reproducible local closed loop)
python main.py --mode dual_patronage --gen-engine api --viz-engine matplotlib

# Phase-2: LoRA scaffold (writes train/infer metadata and placeholder images)
python main.py --mode dual_patronage --gen-engine lora --viz-engine matplotlib

# API generation + VLM evaluation
python main.py --mode dual_patronage --gen-engine api --viz-engine matplotlib
```

Additional config keys (`config.json`):

- `gen_engine`: `none|api|lora`
- `gen_round_every`, `gen_per_round`, `gen_top_k`
- `api_provider`, `api_model`, `api_endpoint`, `api_seed_policy`
- `vlm_enabled`, `vlm_provider`, `vlm_model`, `vlm_endpoint`
- `strict_llava_create`, `strict_llava_decode`: strict LLaVA mode switches
- `feedback_from_generated`: whether to read `generation_feedback.npy` as initial influence bonus for the next simulation
- `feedback_closed_loop`: whether to enable closed loop mode ("generate new images -> embed -> add new nodes -> keep evolving")
- `spawn_round_every`, `spawn_per_round`, `max_dynamic_nodes`: generation injection frequency and scale in closed loop mode
- `lora_update_every`, `lora_steps`, `lora_batch_size`
- `rebel_ratio`, `strategy_policy`: closed-loop strategy control (`mixed|imitation_only|differentiation_only|self_consistency_only`)
- `gallery_size`, `master_visibility_boost`, `master_ratio`: exposure sampling and mainstream visibility control

Strict LLaVA mode notes:

- When `strict_llava_create=true`, the create stage accepts only `llava` provider/model (or model names containing `llava`); otherwise it raises an error.
- When `strict_llava_decode=true`, the decode stage applies the same strict LLaVA validation.
- This mode ensures consistent experimental protocol and avoids cases where config says "LLaVA" but another model is actually used.

Interpretation: this section captures "evolutionary game outcomes of lower-level creator strategies + constraints from upper-level institutional parameters on payoff structure/generation behavior." Read `game_dynamics.png` together with `institution_vs_generation.png`.

Additional file after enabling closed loop:

- `generated_nodes.csv`: mapping of injected new nodes and parent nodes (`new_node_id`, `parent_node_id`, `round`, `path`)

## 8) Batch runs (CPU-only, no GPU needed)

Use `main.py --pipeline batch` to run multiple experiments. Two batch types are supported:

- `--batch-kind env_plus_artist`: 3x3 (three environment levels x three strategy levels, default)
- `--batch-kind env_only`: 3x1 (three environment levels only)

- Environment levels: `baseline_random` / `norm_only` / `dual_patronage`
- Artist strategy levels: `imitation_only` / `differentiation_only` / `self_consistency_only`

- Force `--device cpu`
- `CUDA_VISIBLE_DEVICES=""` (disable local GPU)
- `api_provider=mock` (stable local generation, no GPU dependency)

Run examples:

```bash
cd "/Users/1532016078qq.com/Desktop/MACS 37005/pj/agentic_simulation"
python main.py --pipeline batch --config config_env_plus_artist.json --batch-T 40 --batch-spawn-round-every 2 --batch-spawn-per-round 4 --batch-gen-top-k 120

# env_only batch over three environments
python main.py --pipeline batch --batch-kind env_only --config config_env_only.json --batch-T 40
```

Outputs:

- `runs_env_plus_artist/batch_runs/<batch_id>/<run_id>/...`: `env_plus_artist` batch results
- `runs_env_plus_artist/batch_runs/<batch_id>/batch_summary.csv`: `env_plus_artist` batch summary
- `runs_env_plus_artist/batch_runs/<batch_id>/plots_compare/*`: auto-generated 3x3 comparison plots for `env_plus_artist`
- `runs_env_only/batch_runs/<batch_id>/<run_id>/...`: `env_only` batch results
- `runs_env_only/batch_runs/<batch_id>/batch_summary.csv`: `env_only` batch summary
- `runs_env_only/batch_runs/<batch_id>/plots_compare/*`: auto-generated 3-environment comparison plots for `env_only`

Current root-level partitioning (for clear protocol separation):

- `runs_env_only/`: environment mechanism only (no artist behavioral closed loop)
- `runs_env_plus_artist/`: environment + artist behavioral closed loop

## 9) How to interpret results

- Across `baseline_random / norm_only / dual_patronage`, `payoff` should not stay all zero over long horizons.  
- `clusters_all` reflects the structure of the full data (should be relatively stable), while `clusters_visible` reflects the evolution of the "visible world."  
- Interpret `cross_cluster_citation_rate` together with `modularity`:  
  higher cross-cluster citation usually comes with lower modularity; the reverse may indicate localized school formation.  
