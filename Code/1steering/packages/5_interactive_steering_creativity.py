"""
Step 5 (Improved): Interactive steering interface with sliders.
Adds more meaningful linear steering by:
  1) Computing steering vectors as an AVERAGE over multiple latents + multiple timesteps
  2) Normalizing vectors (RMS) so slider strength is comparable
  3) Applying steering ONLY in a timestep window (mid-steps by default), with a smooth schedule
  4) Reusing the SAME initial latent for base + all steered variants (clean comparisons)

Assumptions:
  - Your DiffusionModel has a method p_sample(model, x, t, context) like you used.
  - Your TextEncoder.encode([prompt], [special_tokens]) returns text context tensor.
  - Your VAE.decode(latent) returns image tensor in [-1,1].

Run:
  python step5_steering_ui.py --launch_interface --steer_layer middle_block_2
"""

import torch
import argparse
from pathlib import Path
import sys
import numpy as np
import gradio as gr
from tqdm import tqdm

# Local imports
sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_architecture import UNetModel, VAE
from utils.special_tokens import SpecialTokenVocabulary
from utils.visualization import tensor_to_image, create_steering_comparison_gif
from model_training import TextEncoder, DiffusionModel


# -----------------------------
# Helper: deterministic generator
# -----------------------------
def make_torch_generator(device: torch.device, seed: int | None):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


# -----------------------------
# Steering class
# -----------------------------
class SteeringVectorGenerator:
    """
    Generate and apply steering vectors for attributes.

    Key improvements:
      - compute_steering_vector_avg(): average over multiple latents + multiple timesteps
      - apply steering only in a timestep window with smooth schedule
      - reuse same initial latent for base and steered images
    """

    def __init__(self, model, diffusion, text_encoder, vae, device):
        self.model = model
        self.diffusion = diffusion
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self.steering_vectors = {}

    # -----------------------------
    # Hook utilities
    # -----------------------------
    def _register_middle_block_hook(self, layer_name: str, hook_fn):
        if "middle_block" not in layer_name:
            raise ValueError("Only middle_block_* supported (e.g., middle_block_0..3).")

        idx = int(layer_name.split("_")[-1])
        if idx < 0 or idx >= len(self.model.middle_block):
            raise IndexError(
                f"Invalid layer_name={layer_name}: idx={idx} out of range "
                f"(len(middle_block)={len(self.model.middle_block)})"
            )

        return self.model.middle_block[idx].register_forward_hook(hook_fn)

    @staticmethod
    def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float() ** 2) + eps)

    @staticmethod
    def _normalize_rms(vec: torch.Tensor, target_rms: float = 1.0) -> torch.Tensor:
        # vec: any shape
        cur = SteeringVectorGenerator._rms(vec)
        return vec * (target_rms / cur)

    # -----------------------------
    # Compute steering vectors (robust)
    # -----------------------------
    @torch.no_grad()
    def compute_steering_vector_avg(
        self,
        attribute_name: str,
        positive_prompt: str,
        positive_tokens: dict,
        negative_prompt: str,
        negative_tokens: dict,
        layer_name: str = "middle_block_2",
        # averaging controls:
        n_latents: int = 16,
        timesteps: list[int] = [300, 500, 700],
        seed: int = 123,
        # normalization:
        normalize: bool = True,
        target_rms: float = 1.0,
    ):
        """
        Compute steering vector as average over multiple latents and timesteps:
            v = E_{z,t}[ act(z,t,pos) - act(z,t,neg) ]

        This is MUCH more stable than single latent + single timestep.
        """

        self.model.eval()

        # Prepare contexts once (no need to redo inside loop)
        context_pos = self.text_encoder.encode([positive_prompt], [positive_tokens])
        context_neg = self.text_encoder.encode([negative_prompt], [negative_tokens])

        # Deterministic generator for the steering vector definition
        g = make_torch_generator(self.device, seed)

        # Activation capture
        activations = {}

        def capture_hook(_module, _inputs, output):
            # clone+detach so it cannot be mutated later
            activations["act"] = output.detach().clone()

        handle = self._register_middle_block_hook(layer_name, capture_hook)

        diffs = []
        for i in range(n_latents):
            latent = torch.randn((1, 4, 32, 32), device=self.device, generator=g)

            for t_int in timesteps:
                t = torch.tensor([t_int], device=self.device, dtype=torch.long)

                # Positive forward
                activations.clear()
                _ = self.model(latent, t, context_pos)
                if "act" not in activations:
                    handle.remove()
                    raise RuntimeError(f"Hook did not capture activations for {layer_name} (pos).")
                act_pos = activations["act"]

                # Negative forward
                activations.clear()
                _ = self.model(latent, t, context_neg)
                if "act" not in activations:
                    handle.remove()
                    raise RuntimeError(f"Hook did not capture activations for {layer_name} (neg).")
                act_neg = activations["act"]

                diffs.append(act_pos - act_neg)

        handle.remove()

        steering_vector = torch.stack(diffs, dim=0).mean(dim=0)

        if normalize:
            steering_vector = self._normalize_rms(steering_vector, target_rms=target_rms)

        self.steering_vectors[attribute_name] = {
            "vector": steering_vector.detach().clone(),
            "layer_name": layer_name,
            "meta": {
                "n_latents": n_latents,
                "timesteps": timesteps,
                "seed": seed,
                "normalize": normalize,
                "target_rms": target_rms,
                "pos_prompt": positive_prompt,
                "neg_prompt": negative_prompt,
            },
        }
        return steering_vector

    # -----------------------------
    # Steering schedule / gating
    # -----------------------------
    @staticmethod
    def _steering_scale_for_step(step_idx: int, total_steps: int, gate_lo: float, gate_hi: float):
        """
        Smooth gate on step index, not on raw diffusion timestep.
        gate_lo/gate_hi are fractions in [0,1], e.g. 0.2..0.8.
        Returns scale in [0,1].
        """
        if total_steps <= 1:
            return 1.0

        frac = step_idx / (total_steps - 1)

        if frac < gate_lo or frac > gate_hi:
            return 0.0

        # Map frac from [gate_lo, gate_hi] -> [0,1]
        x = (frac - gate_lo) / max(1e-8, (gate_hi - gate_lo))

        # Cosine ramp: 0->1->0 over x in [0,1]
        # peak at center
        return float(np.sin(np.pi * x) ** 2)

    # -----------------------------
    # Sampling with latent reuse
    # -----------------------------
    @torch.no_grad()
    def _sample_latent(
        self,
        context: torch.Tensor,
        latent_init: torch.Tensor,
        num_steps: int,
        steering: dict | None = None,
        seed: int | None = None,  # kept for API symmetry; latent_init already fixes randomness
        gate_lo: float = 0.2,
        gate_hi: float = 0.8,
    ):
        """
        Core sampler that starts from a provided latent_init and optionally applies steering.
        steering: dict with keys {vector, layer_name, strength}
        """

        self.model.eval()

        # Setup steering hook if needed
        handle = None
        if steering is not None:
            vec = steering["vector"]
            layer_name = steering["layer_name"]
            strength = float(steering["strength"])

            # We scale steering per-step via a mutable holder
            scale_holder = {"scale": 0.0}

            def steering_hook(_module, _inputs, output):
                # output + (strength * per_step_scale) * vec
                return output + (strength * scale_holder["scale"]) * vec

            handle = self._register_middle_block_hook(layer_name, steering_hook)

        # Use the provided initial latent (clone to avoid in-place surprises)
        latent = latent_init.clone()

        # Timesteps schedule for sampling
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=self.device,
        )

        for step_idx, t in enumerate(timesteps):
            if steering is not None:
                # Update per-step scale
                scale = self._steering_scale_for_step(step_idx, num_steps, gate_lo, gate_hi)
                scale_holder["scale"] = scale

            t_batch = t.unsqueeze(0)
            latent = self.diffusion.p_sample(self.model, latent, t_batch, context)

        if handle is not None:
            handle.remove()

        return latent

    
    @torch.no_grad()
    def generate_variants_same_latent(
        self,
        prompt: str,
        special_tokens: dict,
        num_steps: int = 50,
        seed: int | None = 42,
        creativity_strength: float = 0.0,
        gate_lo: float = 0.2,
        gate_hi: float = 0.8,
    ):
        """
        Generates 2 images using the SAME initial latent:
          - base
          - base + creativity steering

        Notes:
          - Requires a precomputed steering vector named "creativity" in self.steering_vectors.
          - Reusing the same initial latent makes differences attributable to steering, not randomness.
        """

        context = self.text_encoder.encode([prompt], [special_tokens])

        # Create one shared latent_init
        g = make_torch_generator(self.device, seed)
        latent_init = torch.randn((1, 4, 32, 32), device=self.device, generator=g)

        # Base
        latent_base = self._sample_latent(
            context=context,
            latent_init=latent_init,
            num_steps=num_steps,
            steering=None,
            gate_lo=gate_lo,
            gate_hi=gate_hi,
        )
        img_base = self.vae.decode(latent_base)

        # Creativity steered
        if abs(float(creativity_strength)) > 1e-8:
            latent_creative = self._sample_latent(
                context=context,
                latent_init=latent_init,
                num_steps=num_steps,
                steering={
                    "vector": self.steering_vectors["creativity"]["vector"],
                    "layer_name": self.steering_vectors["creativity"]["layer_name"],
                    "strength": float(creativity_strength),
                },
                gate_lo=gate_lo,
                gate_hi=gate_hi,
            )
            img_creative = self.vae.decode(latent_creative)
        else:
            img_creative = img_base

        return img_base, img_creative


# -----------------------------
# Steering vectors: precompute robustly
# -----------------------------
def precompute_steering_vectors(steering_gen: SteeringVectorGenerator, layer_name="middle_block_2"):
    """
    Compute steering vectors ONCE, robustly.
    """
    print("Pre-computing steering vectors (averaged, normalized, gated for sampling)...")

    
    steering_configs = [
        {
            "name": "creativity",
            "positive": (
                "a surreal imaginative painting with unexpected elements, dreamlike atmosphere, bold color contrasts, experimental composition (high creativity)",
                {"artist": [], "year": [], "gender": [], "location": []},
            ),
            "negative": (
                "a realistic portrait painting, straightforward subject, natural lighting, restrained palette, conventional composition (low creativity)",
                {"artist": [], "year": [], "gender": [], "location": []},
            ),
        },
    ]


    for cfg in steering_configs:
        name = cfg["name"]
        print(f"  Computing steering vector for {name} @ {layer_name} ...")

        pos_prompt, pos_tokens = cfg["positive"]
        neg_prompt, neg_tokens = cfg["negative"]

        _ = steering_gen.compute_steering_vector_avg(
            attribute_name=name,
            positive_prompt=pos_prompt,
            positive_tokens=pos_tokens,
            negative_prompt=neg_prompt,
            negative_tokens=neg_tokens,
            layer_name=layer_name,
            n_latents=16,
            timesteps=[300, 500, 700],
            seed=123,
            normalize=True,
            target_rms=1.0,
        )


# -----------------------------
# Gradio UI
# -----------------------------
def create_gradio_interface(steering_gen: SteeringVectorGenerator, vocab: SpecialTokenVocabulary, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    def generate_image(prompt_text, creativity_slider, seed, steps, gate_lo, gate_hi):
        seed = int(seed) if seed is not None else None
        steps = int(steps)

        # Parse special tokens from prompt if user includes them
        special_tokens = vocab.extract_tokens_from_caption(prompt_text)

        img_base, img_creative = steering_gen.generate_variants_same_latent(
            prompt=prompt_text,
            special_tokens=special_tokens,
            num_steps=steps,
            seed=seed,
            creativity_strength=float(creativity_slider),
            gate_lo=float(gate_lo),
            gate_hi=float(gate_hi),
        )

        images = [
            tensor_to_image(img_base),
            tensor_to_image(img_creative),
        ]
        return images

    with gr.Blocks(title="Diffusion Model Steering Interface (Improved)") as demo:
        gr.Markdown("# Interactive Diffusion Model Steering (Improved)")
        gr.Markdown(
            "This version uses averaged/normalized steering vectors, timestep-gated steering, "
            "and shared initial latents for clean comparisons."
        )

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., a portrait painting of a person, calm lighting, simple background",
                    value="a portrait painting of a person",
                )
                creativity_slider = gr.Slider(
                    minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                    label="Creativity Steering (negative=lower creativity, positive=higher creativity)"
                )
                steps_input = gr.Slider(
                    minimum=10, maximum=100, value=50, step=1,
                    label="Sampling Steps"
                )

                gate_lo_input = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    label="Steering Gate Start (fraction of steps)"
                )
                gate_hi_input = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                    label="Steering Gate End (fraction of steps)"
                )

                seed_input = gr.Number(label="Random Seed", value=42, precision=0)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Generated Images (Base / +Creativity)",
                    columns=2,
                    height=600,
                )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                creativity_slider,
                seed_input,
                steps_input,
                gate_lo_input,
                gate_hi_input,
            ],
            outputs=output_gallery,
        )

        gr.Markdown("## Notes")
        gr.Markdown(
            """
- For cleanest comparisons, keep the main text constant and use only the creativity slider.
- Steering is applied only in a mid-step window (gate sliders) with a smooth schedule (reduces artifacts).
- All variants reuse the same initial latent so differences are attributable to steering, not randomness.
"""
        )

    return demo


# -----------------------------
# Optional: GIFs (also improved via gating + shared latent)
# -----------------------------

def create_steering_gifs(steering_gen: SteeringVectorGenerator, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attribute = "creativity"
    base_prompt = "a portrait painting of a person"
    base_tokens = {"artist": [], "year": [], "gender": [], "location": []}

    print(f"Creating GIF for {attribute} steering...")

    alphas = np.linspace(-2.0, 2.0, 21)
    images_by_alpha = []

    for alpha in tqdm(alphas, desc=f"{attribute} GIF"):
        # For GIF: reuse same latent by keeping seed fixed; generate just the steered variant.
        img_base, img_creative = steering_gen.generate_variants_same_latent(
            prompt=base_prompt,
            special_tokens=base_tokens,
            num_steps=50,
            seed=42,
            creativity_strength=float(alpha),
            gate_lo=0.2,
            gate_hi=0.8,
        )
        images_by_alpha.append((float(alpha), img_creative))

    if images_by_alpha:
        create_steering_comparison_gif(
            images_by_alpha,
            output_dir / f"{attribute}_steering.gif",
            attribute_name=attribute,
            duration=0.2,
        )
        print(f"Saved {attribute}_steering.gif")


# -----------------------------
# Main

# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = SpecialTokenVocabulary()
    vocab.load_vocabulary(args.vocab_path)

    # Load VAE
    print("Loading VAE...")
    vae = VAE().to(device)
    vae_checkpoint = torch.load(args.vae_path, map_location=device)
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.eval()

    # Load text encoder
    print("Loading text encoder...")
    text_encoder = TextEncoder(vocab, device)

    # Load U-Net
    print("Loading diffusion model...")
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=args.model_channels,
        context_dim=text_encoder.text_model.config.hidden_size,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load special token embeddings into text encoder
    text_encoder.special_token_embeddings.load_state_dict(checkpoint["text_encoder_state_dict"])
    text_encoder.special_token_embeddings.eval()

    # Create diffusion (ensure buffers on device)
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

    # Create steering generator
    print("Creating steering vector generator...")
    steering_gen = SteeringVectorGenerator(model, diffusion, text_encoder, vae, device)

    # Precompute robust steering vectors ONCE
    precompute_steering_vectors(steering_gen, layer_name=args.steer_layer)

    if args.create_gifs:
        print("\nCreating steering GIFs...")
        create_steering_gifs(steering_gen, output_dir)

    if args.launch_interface:
        print("\nLaunching interactive interface...")
        demo = create_gradio_interface(steering_gen, vocab, output_dir)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive steering interface (Improved)")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/outputs/diffusion/diffusion_best.pt",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json",
    )
    parser.add_argument("--model_channels", type=int, default=192)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/outputs/interactive",
    )

    parser.add_argument(
        "--steer_layer",
        type=str,
        default="middle_block_2",
        help=(
            "Layer name for steering hooks. For your UNet middle_block: "
            "0=ResBlock, 1=SelfAttn, 2=CrossAttn, 3=ResBlock. "
            "Usually middle_block_2 (CrossAttentionBlock) is a strong choice."
        ),
    )

    parser.add_argument("--create_gifs", action="store_true", help="Create steering GIFs")
    parser.add_argument(
        "--launch_interface",
        action="store_true",
        default=True,
        help="Launch Gradio interface",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()
    main(args)
