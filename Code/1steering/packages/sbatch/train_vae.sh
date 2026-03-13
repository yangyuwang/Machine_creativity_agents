#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --output=logs/vae_%j.out
#SBATCH --error=logs/vae_%j.err
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

# Load modules (adjust for your cluster)
module load python/anaconda-2022.05 cuda/12.2
source activate gallery_gpt

echo "Starting VAE training..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run VAE training
python code/2_vae_training.py \
    --jsonl_path /home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged_1400_1600.jsonl \
    --image_dir /home/wangyd/Projects/macs_thesis/yangyu/artwork_images \
    --vocab_path /home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --image_size 256 \
    --latent_channels 4 \
    --num_workers 8 \
    --output_dir /home/wangyd/Projects/macs_thesis/yangyu/outputs/vae

echo "VAE training complete!"
