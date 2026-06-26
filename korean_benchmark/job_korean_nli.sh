#!/bin/bash
#SBATCH --job-name=korean_nli
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --mem=16G
#SBATCH --output=logs/korean_nli_%j.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

pip install --quiet transformers torch pandas scikit-learn sentencepiece protobuf

cd ~/thesis
python run_korean_nli.py
