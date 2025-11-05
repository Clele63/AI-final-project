#!/bin/bash
#SBATCH --job-name=step2_train_classifier
#SBATCH --output=slurm_logs/step2_training-%j.out
#SBATCH --error=slurm_logs/step2_training-%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# --- Config ---

# (Identique au step1) : Garantit un environnement propre
unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH" 

# (Adapté pour step2) : Répertoire isolé pour ce job
UV_PROJECT=$HOME/slurm/ai-final-project/step2

echo "=== Informations SLURM ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"
echo "GPU(s) alloué(s) : $CUDA_VISIBLE_DEVICES"
echo "--------------------------"

# --- Initialisation uv ---
echo "=== Préparation de l'environnement uv ==="

# (Identique au step1) : Crée et se place dans le dossier de travail isolé
mkdir -p "$UV_PROJECT"
cd "$UV_PROJECT"

# (Identique au step1) : Initialise le projet s'il n'existe pas
if [ ! -f "pyproject.toml" ]; then
  uv init --python 3.12 --quiet
fi

# (Identique au step1) : Installe les dépendances dans le venv isolé
# Le requirements.txt est à son chemin absolu
uv pip install -r /home/lcoussy/AI-final-project/requirements.txt

echo "Étape 3/3 : Lancement du script Python (Training)..."
# (Adapté pour step2) : Lance le script de training
# Tous les chemins de fichiers sont ABSOLUS
uv run python /home/lcoussy/AI-final-project/step2_train_classifier.py \
    --input_csv "/home/lcoussy/AI-final-project/data/github_data_with_readmes.csv" \
    --categories_file "/home/lcoussy/AI-final-project/outputs/step1_clustering/github_categories_database.json" \
    --output_model_dir "/home/lcoussy/AI-final-project/outputs/step2_classification/distilroberta_github_classifier" \
    --base_model "distilroberta-base" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 8

echo "=== Fin du Job 2 ==="