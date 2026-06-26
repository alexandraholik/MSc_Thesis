#!/bin/bash
#SBATCH --job-name=korean_matching
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rome
#SBATCH --mem=16G
#SBATCH --output=logs/matching_%j.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

pip install --quiet sentence-transformers pandas scikit-learn

cd ~/thesis
python run_matching_q2q.py
