#!/bin/bash
#SBATCH --job-name=interpretability
#SBATCH --output=logs/interpret_%j.out
#SBATCH --error=logs/interpret_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

# Load modules (adjust for your cluster)
module load python/anaconda-2022.05 cuda/12.2
source activate gallery_gpt

# Create logs directory
mkdir -p logs

echo "Starting interpretability analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run interpretability analysis
python code/3_interpretability_analysis.py \
    --checkpoint_path /home/wangyd/Projects/macs_thesis/yangyu/outputs/diffusion/diffusion_best.pt \
    --vae_path /home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt \
    --vocab_path /home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json \
    --model_channels 192 \
    --output_dir /home/wangyd/Projects/macs_thesis/yangyu/outputs/interpretability

echo "Interpretability analysis complete!"

# # Run neuron discovery
# echo "Starting neuron discovery..."
# python code/4_neuron_discovery.py \
#     --checkpoint_path /home/wangyd/Projects/macs_thesis/yangyu/outputs/diffusion/diffusion_best.pt \
#     --vae_path /home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt \
#     --vocab_path /home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json \
#     --model_channels 192 \
#     --output_dir /home/wangyd/Projects/macs_thesis/yangyu/outputs/neuron_discovery

# # Run steering gifs
# echo "Starting linear steering..."
# python code/5_interactive_steering.py \
#     --checkpoint_path /home/wangyd/Projects/macs_thesis/yangyu/outputs/diffusion/diffusion_best.pt \
#     --vae_path /home/wangyd/Projects/macs_thesis/yangyu/outputs/vae/vae_best.pt \
#     --vocab_path /home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json \
#     --create_gifs \
#     --output_dir /home/wangyd/Projects/macs_thesis/yangyu/outputs/interactive


# echo "All interpretability tasks complete!"
