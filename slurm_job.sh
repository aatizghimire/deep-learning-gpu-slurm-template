#!/bin/bash
#SBATCH --job-name=deep_learning_training
#SBATCH --output=logs/slurm-%j.out   # Save output logs to file
#SBATCH --error=logs/slurm-%j.err    # Save error logs to file
#SBATCH --partition=gpu              # Specify GPU partition (customize based on your setup)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --ntasks=1                   # Request 1 task (process)
#SBATCH --mem=16G                    # Memory allocation for the job
#SBATCH --time=02:00:00              # Time limit for the job (example: 2 hours)

# Load modules if necessary (e.g., CUDA)
module load cuda/11.7

# Run the training script
python scripts/train.py
