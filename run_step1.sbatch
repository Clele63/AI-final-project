#!/bin/bash
#SBATCH --job-name=step1_create_categories
#SBATCH --output=slurm_logs/step1_categories-%j.out
#SBATCH --error=slurm_logs/step1_categories-%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# --- Config ---

unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH" 
UV_PROJECT=$HOME/slurm/ai-final-project/step1

echo "=== Informations SLURM ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"
echo "GPU(s) alloué(s) : $CUDA_VISIBLE_DEVICES"
echo "--------------------------"

# --- Initialisation uv ---
echo "=== Préparation de l'environnement uv ==="

# Crée un répertoire de travail isolé pour ce job
mkdir -p "$UV_PROJECT"
cd "$UV_PROJECT"

# Initialise le projet s'il n'existe pas encore
if [ ! -f "pyproject.toml" ]; then
  uv init --python 3.12 --quiet
fi

# uv pip sync /home/lcoussy/AI-final-project/requirements.txt
uv pip install -r /home/lcoussy/AI-final-project/requirements.txt


echo "Étape 3/3 : Lancement du script Python..."
uv run python /home/lcoussy/AI-final-project/step1_create_categories.py \
    --input_csv "/home/lcoussy/AI-final-project/data/github_data_with_readmes.csv" \
    --output_dir "/home/lcoussy/AI-final-project/outputs/step1_clustering" \
    --n_categories 200 \
    --chunk_size 2000

echo "=== Fin du Job 1 ==="