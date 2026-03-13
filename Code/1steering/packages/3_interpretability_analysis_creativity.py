"""
Step 3: Interpretability analysis using *proper* activation patching (causal tracing).
Identifies which U-Net layers control specific attributes.

Key fixes vs your previous version:
1) Uses a correct corrupt trajectory (x_t^corrupt) and caches per-timestep activations.
2) Uses integer timesteps without duplicates (no linspace->long rounding artifacts).
3) Patches ONLY the block output at a chosen U-Net stage (input/middle/output) in a stable way.
4) Computes a more meaningful effect metric in latent space + (optional) pixel MSE.

ADDED (your request):
5) For each test case, also save an extra PNG that visualizes the *difference*
   between each patched image and the original clean image.
   File: {attribute}_patching_diff.png

NOTE: This does NOT change any patching parameters or your experiment logic.
It only adds an additional visualization output.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

# NEW: for difference plots
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from utils.model_architecture import UNetModel, VAE, ResBlock, CrossAttentionBlock
from utils.data_loader import PaintingDataset, create_data_loaders
from utils.special_tokens import SpecialTokenVocabulary
from utils.visualization import (
    visualize_activation_patching_results,
)

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
from model_training import TextEncoder, DiffusionModel


# ----------------------------
# Helpers
# ----------------------------
def make_inference_timesteps(num_timesteps: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Create a strictly-decreasing integer timestep schedule with no duplicates.
    Uses striding over [num_timesteps-1 ... 0].
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    stride = max(num_timesteps // num_steps, 1)
    ts = torch.arange(num_timesteps - 1, -1, -stride, device=device, dtype=torch.long)
    ts = ts[:num_steps]
    # Ensure last timestep is 0 (optional but nice)
    if ts.numel() > 0 and ts[-1].item() != 0:
        ts = torch.cat([ts, torch.zeros(1, device=device, dtype=torch.long)], dim=0)
    return ts


# NEW: difference plot saver
def save_patching_difference_grid(
    clean_image: torch.Tensor,
    patched_images: List[torch.Tensor],
    layer_names: List[str],
    save_path: Path,
    attribute_name: str,
    mode: str = "abs",               # "abs" or "signed"
    per_image_normalize: bool = True # visualization-only normalization for visibility
):
    """
    Save an extra PNG that shows differences between each patched image and the clean/original.

    clean_image: [1,3,H,W] or [3,H,W] in [-1,1]
    patched_images: list of tensors same shape as clean_image in [-1,1]

    mode:
      - "abs": show |patched - clean| (most interpretable)
      - "signed": show patched-clean mapped around 0.5 (more subtle)

    per_image_normalize:
      - True: each diff panel normalized independently (makes subtle diffs visible)
      - False: global scaling across all panels (comparable magnitude across layers)
    """

    def to_hw3_unit(x: torch.Tensor) -> np.ndarray:
        # x in [-1,1] -> [0,1] HWC
        if x.dim() == 4:
            x = x[0]
        x = x.detach().float().cpu()
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        x = x.permute(1, 2, 0).numpy()
        return x

    clean_np = to_hw3_unit(clean_image)

    diffs = []
    for p in patched_images:
        p_np = to_hw3_unit(p)
        d = (p_np - clean_np)

        if mode == "abs":
            d = np.abs(d)
        elif mode == "signed":
            d = 0.5 + 0.5 * d
            d = np.clip(d, 0, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        diffs.append(d)

    # visualization scaling only
    if mode == "abs":
        if per_image_normalize:
            diffs_vis = []
            for d in diffs:
                m = float(d.max())
                if m > 1e-8:
                    diffs_vis.append(d / m)
                else:
                    diffs_vis.append(d)
        else:
            gmax = max([float(d.max()) for d in diffs] + [1e-8])
            diffs_vis = [d / gmax for d in diffs]
    else:
        diffs_vis = diffs

    n = len(diffs_vis)
    fig_w = max(12, 2.2 * (n + 1))
    fig, axes = plt.subplots(1, n + 1, figsize=(fig_w, 3.2))
    fig.suptitle(f"Activation Patching Differences: {attribute_name}", fontsize=14)

    # Original
    axes[0].imshow(clean_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Diff panels
    for i, (d, name) in enumerate(zip(diffs_vis, layer_names), start=1):
        axes[i].imshow(d)
        axes[i].set_title(f"Diff\n{name}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Activation patcher
# ----------------------------
class ActivationPatcher:
    """
    Proper activation patching / causal tracing for this U-Net.

    Important: Your UNetModel's "blocks" are heterogeneous:
      - input_blocks: [Conv2d, ModuleList([...]), Conv2d (downsample), ...]
      - middle_block: ModuleList([ResBlock, Attn, CrossAttn, ResBlock])
      - output_blocks: ModuleList([ModuleList([...]), ModuleList([...]), ...])

    We patch at the *module output* for:
      - input_block_i : the whole input_blocks[i] module output (after its internal layers)
      - middle_block_i: the middle_block[i] module output
      - output_block_i: the whole output_blocks[i] ModuleList output (after its internal layers)

    This is robust (always a tensor) and matches your forward structure.
    """

    def __init__(self, model: UNetModel, diffusion: DiffusionModel, text_encoder: TextEncoder, vae: VAE, device):
        self.model = model
        self.diffusion = diffusion
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device

        self.layer_names = self._get_layer_names()

    def _get_layer_names(self) -> List[str]:
        names = []
        for i in range(len(self.model.input_blocks)):
            names.append(f"input_block_{i}")
        for i in range(len(self.model.middle_block)):
            names.append(f"middle_block_{i}")
        for i in range(len(self.model.output_blocks)):
            names.append(f"output_block_{i}")
        return names

    def _get_module_by_name(self, layer_name: str) -> nn.Module:
        if layer_name.startswith("input_block_"):
            idx = int(layer_name.split("_")[-1])
            m = self.model.input_blocks[idx]
        elif layer_name.startswith("middle_block_"):
            idx = int(layer_name.split("_")[-1])
            m = self.model.middle_block[idx]
        elif layer_name.startswith("output_block_"):
            idx = int(layer_name.split("_")[-1])
            m = self.model.output_blocks[idx]
        else:
            raise ValueError(f"Unknown layer_name: {layer_name}")

        # IMPORTANT: ModuleList is a container, not callable -> hook a callable child.
        if isinstance(m, nn.ModuleList):
            if len(m) == 0:
                raise RuntimeError(f"{layer_name} is an empty ModuleList.")
            return m[-1]  # last layer output == block output in your forward structure
        return m

    @torch.no_grad()
    def _forward_with_hook_capture(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward once, capture the output tensor of the specified module.
        Returns: (model_output, captured_activation)
        """
        captured = {}

        def hook_fn(_module, _inp, out):
            captured["act"] = out.detach().clone()

        module = self._get_module_by_name(layer_name)
        handle = module.register_forward_hook(hook_fn)
        out = self.model(latent, timestep, context)
        handle.remove()

        act = captured.get("act", None)
        if act is None:
            raise RuntimeError(f"Failed to capture activation for {layer_name}.")
        return out, act

    @torch.no_grad()
    def _forward_with_hook_patch(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        layer_name: str,
        replacement: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward once, replacing the output tensor of the specified module.
        Returns: model_output (predicted noise)
        """

        def hook_fn(_module, _inp, out):
            if out.shape != replacement.shape:
                raise RuntimeError(
                    f"Shape mismatch when patching {layer_name}: "
                    f"module out {tuple(out.shape)} vs replacement {tuple(replacement.shape)}"
                )
            return replacement

        module = self._get_module_by_name(layer_name)
        handle = module.register_forward_hook(hook_fn)
        out = self.model(latent, timestep, context)
        handle.remove()
        return out

    @torch.no_grad()
    def _p_sample_step_from_noise_pred(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        One DDPM p_sample-like update using your diffusion buffers.
        """
        ti = int(t.item())

        alpha_t = self.diffusion.alphas[ti].view(1, 1, 1, 1)
        beta_t = self.diffusion.betas[ti].view(1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[ti].view(1, 1, 1, 1)

        model_mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred
        )

        if ti > 0:
            noise = torch.randn_like(x)
            posterior_variance_t = self.diffusion.posterior_variance[ti].view(1, 1, 1, 1)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def _generate_trajectory(
        self,
        latent0: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        cache_layer: Optional[str] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Generate a trajectory x_t over the provided inference timesteps.
        """
        traj: Dict[int, Dict[str, torch.Tensor]] = {}
        x = latent0.clone()

        for t in timesteps:
            ti = int(t.item())
            t_batch = t.view(1)

            if cache_layer is None:
                noise_pred = self.model(x, t_batch, context)
                traj[ti] = {"x": x.detach().clone()}
            else:
                noise_pred, act = self._forward_with_hook_capture(x, t_batch, context, cache_layer)
                traj[ti] = {"x": x.detach().clone(), "act": act}

            x = self._p_sample_step_from_noise_pred(x, t_batch, noise_pred)

        traj["final"] = {"x": x.detach().clone()}
        return traj

    @torch.no_grad()
    def activation_patching_experiment(
        self,
        clean_prompt: str,
        clean_special_tokens: Dict[str, List[str]],
        corrupt_prompt: str,
        corrupt_special_tokens: Dict[str, List[str]],
        layers_to_test: Optional[List[str]] = None,
        num_steps: int = 50,
        seed: int = 0
    ) -> Dict:
        """
        Proper causal tracing.
        """
        if layers_to_test is None:
            layers_to_test = self.layer_names

        self.model.eval()
        self.vae.eval()

        clean_context = self.text_encoder.encode([clean_prompt], [clean_special_tokens])
        corrupt_context = self.text_encoder.encode([corrupt_prompt], [corrupt_special_tokens])

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        latent0 = torch.randn((1, 4, 32, 32), generator=g, device=self.device)

        timesteps = make_inference_timesteps(self.diffusion.num_timesteps, num_steps, self.device)

        print("Generating CLEAN baseline...")
        clean_traj = self._generate_trajectory(latent0, timesteps, clean_context, cache_layer=None)
        clean_latent_final = clean_traj["final"]["x"]
        clean_image = self.vae.decode(clean_latent_final)

        print("Generating CORRUPT baseline...")
        corrupt_traj = self._generate_trajectory(latent0, timesteps, corrupt_context, cache_layer=None)
        corrupt_latent_final = corrupt_traj["final"]["x"]
        corrupt_image = self.vae.decode(corrupt_latent_final)

        results = {
            "clean_image": clean_image,
            "corrupt_image": corrupt_image,
            "patched_images": {},
            "layer_effects": {},
        }

        for layer_name in layers_to_test:
            print(f"\nPatching layer: {layer_name}")

            corrupt_cache = self._generate_trajectory(latent0, timesteps, corrupt_context, cache_layer=layer_name)

            x = latent0.clone()
            for t in tqdm(timesteps, desc=f"Patched {layer_name}", leave=False):
                ti = int(t.item())
                t_batch = t.view(1)

                x_corrupt_t = corrupt_cache[ti]["x"]
                _, act_corrupt = self._forward_with_hook_capture(x_corrupt_t, t_batch, corrupt_context, layer_name)

                noise_pred = self._forward_with_hook_patch(x, t_batch, clean_context, layer_name, act_corrupt)

                x = self._p_sample_step_from_noise_pred(x, t_batch, noise_pred)

            patched_latent_final = x
            patched_image = self.vae.decode(patched_latent_final)
            results["patched_images"][layer_name] = patched_image

            latent_mse_to_clean = torch.nn.functional.mse_loss(patched_latent_final, clean_latent_final).item()
            latent_mse_to_corrupt = torch.nn.functional.mse_loss(patched_latent_final, corrupt_latent_final).item()

            pixel_mse_to_clean = torch.nn.functional.mse_loss(patched_image, clean_image).item()
            pixel_mse_to_corrupt = torch.nn.functional.mse_loss(patched_image, corrupt_image).item()

            results["layer_effects"][layer_name] = {
                "latent_mse_to_clean": latent_mse_to_clean,
                "latent_mse_to_corrupt": latent_mse_to_corrupt,
                "pixel_mse_to_clean": pixel_mse_to_clean,
                "pixel_mse_to_corrupt": pixel_mse_to_corrupt,
                "delta_latent": latent_mse_to_clean - latent_mse_to_corrupt,
                "delta_pixel": pixel_mse_to_clean - pixel_mse_to_corrupt,
            }

        return results


# ----------------------------
# Experiment driver
# ----------------------------
def analyze_attributes(patcher: ActivationPatcher, output_dir: str, num_steps: int = 50, seed: int = 0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        {
            "name": "creativity",
            "clean": (
                "a realistic portrait painting, straightforward subject, "
                "natural lighting, restrained palette, conventional composition "
                "(low creativity)"
            ),
            "corrupt": (
                "a surreal imaginative painting with unexpected elements, "
                "dreamlike atmosphere, bold color contrasts, experimental composition "
                "(high creativity)"
            ),
            "clean_tokens": {"artist": [], "year": [], "gender": [], "location": []},
            "corrupt_tokens": {"artist": [], "year": [], "gender": [], "location": []},
        },
    ]


    sample_layers = [
        "input_block_0", "input_block_3", "input_block_6",
        "middle_block_0", "middle_block_2",
        "output_block_3", "output_block_6", "output_block_9",
    ]

    all_results = {}

    for tc in test_cases:
        print(f"\n{'='*60}\nTesting attribute: {tc['name']}\n{'='*60}")

        results = patcher.activation_patching_experiment(
            clean_prompt=tc["clean"],
            clean_special_tokens=tc["clean_tokens"],
            corrupt_prompt=tc["corrupt"],
            corrupt_special_tokens=tc["corrupt_tokens"],
            layers_to_test=sample_layers,
            num_steps=num_steps,
            seed=seed,
        )
        all_results[tc["name"]] = results

        # Save images grid (original behavior)
        patched_images = [results["patched_images"][layer] for layer in sample_layers]
        visualize_activation_patching_results(
            original_images=results["clean_image"],
            patched_images=patched_images,
            layer_names=sample_layers,
            save_path=output_dir / f"{tc['name']}_patching.png",
            attribute_name=tc["name"],
        )

        # NEW: Save a difference grid
        save_patching_difference_grid(
            clean_image=results["clean_image"],
            patched_images=patched_images,
            layer_names=sample_layers,
            save_path=output_dir / f"{tc['name']}_patching_diff.png",
            attribute_name=tc["name"],
            mode="abs",
            per_image_normalize=True,
        )

        # Save effects JSON
        effects_path = output_dir / f"{tc['name']}_layer_effects.json"
        with open(effects_path, "w") as f:
            json.dump(results["layer_effects"], f, indent=2)

        # Print top layers by delta_latent
        items = list(results["layer_effects"].items())
        items.sort(key=lambda kv: kv[1]["delta_latent"], reverse=True)

        print("\nTop layers (most shifted toward corrupt, by delta_latent):")
        for layer, metrics in items[:5]:
            print(
                f"  {layer}: delta_latent={metrics['delta_latent']:.6f} "
                f"(lat_clean={metrics['latent_mse_to_clean']:.6f}, lat_corrupt={metrics['latent_mse_to_corrupt']:.6f})"
            )

    summary = {}
    for attr, res in all_results.items():
        best = max(res["layer_effects"].items(), key=lambda kv: kv[1]["delta_latent"])
        summary[attr] = {"top_layer": best[0], "top_metrics": best[1]}

    with open(output_dir / "attribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return all_results


# ----------------------------
# Main
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vocabulary...")
    vocab = SpecialTokenVocabulary()
    vocab.load_vocabulary(args.vocab_path)

    print("Loading VAE...")
    vae = VAE().to(device)
    vae_checkpoint = torch.load(args.vae_path, map_location=device)
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.eval()

    print("Loading text encoder...")
    text_encoder = TextEncoder(vocab, device)

    print("Loading diffusion model (U-Net)...")
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=args.model_channels,
        context_dim=text_encoder.text_model.config.hidden_size,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    text_encoder.special_token_embeddings.load_state_dict(checkpoint["text_encoder_state_dict"])
    text_encoder.special_token_embeddings.eval()

    diffusion = DiffusionModel(num_timesteps=1000)
    for attr in [
        "betas",
        "alphas",
        "alphas_cumprod",
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "posterior_variance",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    print("Creating activation patcher...")
    patcher = ActivationPatcher(model, diffusion, text_encoder, vae, device)

    print("\nRunning interpretability analysis...")
    analyze_attributes(
        patcher,
        output_dir=str(output_dir),
        num_steps=args.num_steps,
        seed=args.seed,
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretability analysis (activation patching)")

    parser.add_argument("--checkpoint_path", type=str,
                        default="/home/wangyd/Projects/macs_thesis/yangyu/outputs/diffusion/diffusion_best.pt")
    parser.add_argument("--vae_path", type=str,
                        default="/home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt")
    parser.add_argument("--vocab_path", type=str,
                        default="/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json")
    parser.add_argument("--model_channels", type=int, default=192)
    parser.add_argument("--output_dir", type=str,
                        default="/home/wangyd/Projects/macs_thesis/yangyu/outputs/interpretability")

    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
