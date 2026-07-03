#!/bin/bash
#SBATCH --job-name=jupyter_cvcs
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/jupyter_%j.out
#SBATCH --account=cvcs2026

source /homes/$USER/cvcs2026/venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
