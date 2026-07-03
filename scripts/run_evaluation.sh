#!/bin/bash
#SBATCH --job-name=evaluate_model
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/homes/$USER/cvcs2026/logs/evaluate_%j.out
#SBATCH --error=/homes/$USER/cvcs2026/logs/evaluate_%j.err
#SBATCH --account=cvcs2026

module load py-torch/2.8.0-gcc-11.4.0-cuda-12.6.3
source /homes/$USER/cvcs2026/venv/bin/activate

python evaluate.py 