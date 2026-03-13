"""
Step 4: Automatic neuron discovery using gradient-based attribution.
Identifies which specific neurons control each attribute.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent))

from utils.model_architecture import UNetModel, VAE
from utils.special_tokens import SpecialTokenVocabulary
from utils.visualization import plot_neuron_importance

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model_training import TextEncoder, DiffusionModel


class NeuronDiscovery:
    """Automatically discover neurons that control specific attributes."""

    def __init__(self, model, diffusion, text_encoder, vae, device):
        self.model = model
        self.diffusion = diffusion
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device

    def compute_neuron_importance(
        self,
        attribute_prompts,
        attribute_tokens_list,
        num_samples=10,
        method='gradient'
    ):
        """
        Compute importance score for each neuron w.r.t. an attribute.

        Args:
            attribute_prompts: List of prompts with target attribute
            attribute_tokens_list: List of special token dicts
            num_samples: Number of samples to average over
            method: 'gradient' or 'activation'

        Returns:
            neuron_scores: Dict mapping layer_name -> importance scores
            neuron_ids: Dict mapping layer_name -> neuron indices
        """
        self.model.train()  # Need gradients
        self.text_encoder.special_token_embeddings.train()

        neuron_gradients = {}
        neuron_activations = {}

        for sample_idx in tqdm(range(num_samples), desc='Computing importance'):
            # Random latent
            latent = torch.randn(1, 4, 32, 32, requires_grad=True).to(self.device)

            # Random timestep
            t = torch.randint(0, self.diffusion.num_timesteps, (1,)).to(self.device)

            # Encode text
            prompt = attribute_prompts[sample_idx % len(attribute_prompts)]
            tokens = attribute_tokens_list[sample_idx % len(attribute_tokens_list)]
            context = self.text_encoder.encode([prompt], [tokens])

            # Forward pass
            noise_pred = self.model(latent, t, context)

            # Compute "attribute presence" metric
            # Simple heuristic: mean absolute activation
            attribute_score = noise_pred.abs().mean()

            # Backward pass
            attribute_score.backward()

            # Collect gradients for each layer
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    if name not in neuron_gradients:
                        neuron_gradients[name] = []
                    neuron_gradients[name].append(param.grad.abs().cpu().numpy())

            # Zero gradients
            self.model.zero_grad()
            self.text_encoder.special_token_embeddings.zero_grad()

        # Average gradients across samples
        neuron_scores = {}
        for name, grads in neuron_gradients.items():
            avg_grad = np.mean(grads, axis=0)
            # Flatten to get per-neuron scores
            neuron_scores[name] = avg_grad.flatten()

        return neuron_scores

    def rank_neurons_by_attribute(
        self,
        attribute_name,
        positive_prompts,
        positive_tokens,
        negative_prompts=None,
        negative_tokens=None,
        top_k=100
    ):
        """
        Rank neurons by how much they correlate with an attribute.

        Uses contrastive approach: compare gradients for positive vs negative.

        Args:
            attribute_name: Name of attribute (e.g., 'gender_male')
            positive_prompts: Prompts with attribute present
            positive_tokens: Special tokens for positive prompts
            negative_prompts: Prompts with attribute absent
            negative_tokens: Special tokens for negative prompts
            top_k: Number of top neurons to return

        Returns:
            Dictionary with top neurons and their scores
        """
        print(f"\nComputing importance for {attribute_name}...")

        # Get gradients for positive examples
        print("  Computing positive gradients...")
        positive_scores = self.compute_neuron_importance(
            positive_prompts, positive_tokens, num_samples=5
        )

        if negative_prompts is not None:
            # Get gradients for negative examples
            print("  Computing negative gradients...")
            negative_scores = self.compute_neuron_importance(
                negative_prompts, negative_tokens, num_samples=5
            )

            # Compute contrast: positive - negative
            contrast_scores = {}
            for layer_name in positive_scores.keys():
                if layer_name in negative_scores:
                    contrast_scores[layer_name] = (
                        positive_scores[layer_name] - negative_scores[layer_name]
                    )
                else:
                    contrast_scores[layer_name] = positive_scores[layer_name]
        else:
            contrast_scores = positive_scores

        # Flatten all neurons and rank
        all_neurons = []
        for layer_name, scores in contrast_scores.items():
            for neuron_idx, score in enumerate(scores):
                all_neurons.append({
                    'layer': layer_name,
                    'neuron_idx': neuron_idx,
                    'score': float(score)
                })

        # Sort by score
        all_neurons.sort(key=lambda x: abs(x['score']), reverse=True)

        # Take top-k
        top_neurons = all_neurons[:top_k]

        return {
            'attribute': attribute_name,
            'top_neurons': top_neurons,
            'total_neurons_analyzed': len(all_neurons)
        }


def discover_all_attributes(discoverer, vocab, output_dir):
    """
    Discover neurons for all attribute categories.

    Args:
        discoverer: NeuronDiscovery instance
        vocab: Vocabulary
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define attribute test cases
    test_cases = [
        {
            "name": "creativity_high_vs_low",
            "positive_prompts": [
                "a surreal imaginative painting with unexpected elements, dreamlike atmosphere, bold color contrasts, experimental composition",
                "an abstract conceptual artwork, unconventional materials, playful distortion, high novelty",
                "a fantastical scene combining unrelated objects, impossible perspective, vibrant palette"
            ],
            "positive_tokens": [
                {"artist": [], "year": [], "gender": [], "location": []},
                {"artist": [], "year": [], "gender": [], "location": []},
                {"artist": [], "year": [], "gender": [], "location": []}
            ],
            "negative_prompts": [
                "a realistic portrait painting, straightforward subject, natural lighting, restrained palette, conventional composition",
                "a traditional still life painting, centered composition, accurate proportions, muted colors",
                "a documentary-style landscape painting, clear horizon line, realistic textures, no surreal elements"
            ],
            "negative_tokens": [
                {"artist": [], "year": [], "gender": [], "location": []},
                {"artist": [], "year": [], "gender": [], "location": []},
                {"artist": [], "year": [], "gender": [], "location": []}
            ]
        }
    ]


    all_results = {}

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Discovering neurons for: {test_case['name']}")
        print(f"{'='*60}")

        results = discoverer.rank_neurons_by_attribute(
            attribute_name=test_case['name'],
            positive_prompts=test_case['positive_prompts'],
            positive_tokens=test_case['positive_tokens'],
            negative_prompts=test_case.get('negative_prompts'),
            negative_tokens=test_case.get('negative_tokens'),
            top_k=100
        )

        all_results[test_case['name']] = results

        # Save results
        results_path = output_dir / f'{test_case["name"]}_top_neurons.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nTop 10 neurons for {test_case['name']}:")
        for i, neuron in enumerate(results['top_neurons'][:10]):
            print(f"  {i+1}. Layer: {neuron['layer']}, Neuron: {neuron['neuron_idx']}, Score: {neuron['score']:.6f}")

    # Save summary
    summary = {
        attr: {
            'num_top_neurons': len(results['top_neurons']),
            'top_layer': results['top_neurons'][0]['layer'] if results['top_neurons'] else None,
            'top_score': results['top_neurons'][0]['score'] if results['top_neurons'] else None
        }
        for attr, results in all_results.items()
    }

    with open(output_dir / 'neuron_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return all_results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
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
    model.load_state_dict(checkpoint['model_state_dict'])    

    text_encoder.special_token_embeddings.load_state_dict(
        checkpoint['text_encoder_state_dict']
    )

    # Create diffusion
    diffusion = DiffusionModel(num_timesteps=1000)
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                 'posterior_variance']:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    # Create discoverer
    print("Creating neuron discoverer...")
    discoverer = NeuronDiscovery(model, diffusion, text_encoder, vae, device)

    # Run discovery
    print("\nRunning automatic neuron discovery...")
    results = discover_all_attributes(discoverer, vocab, output_dir)

    print(f"\nNeuron discovery complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic neuron discovery')

    parser.add_argument('--checkpoint_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/outputs/diffusion/diffusion_best.pt')
    parser.add_argument('--vae_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt')
    parser.add_argument('--vocab_path', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json')
    parser.add_argument('--model_channels', type=int, default=192)
    parser.add_argument('--output_dir', type=str,
                       default='/home/wangyd/Projects/macs_thesis/yangyu/outputs/neuron_discovery')

    args = parser.parse_args()
    main(args)
