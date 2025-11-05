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
UV_EXEC="/home/lcoussy/.local/bin/uv"

echo "=== Job 1 (Clustering) Démarré ==="
echo "Utilisation de UV via : $UV_EXEC"
echo "----------------------------------------------------"

# --- Initialisation uv ---
cd $SLURM_SUBMIT_DIR
echo "Dossier de travail : $(pwd)"

# --- LA CORRECTION (en 3 étapes) ---

# ÉTAPE 1: Créer le .venv explicitement (comme le log l'a suggéré)
echo "Étape 1/3 : Création du .venv avec Python 3.12..."
$UV_EXEC venv --python 3.12

# ÉTAPE 2: Installer les paquets DANS le .venv
echo "Étape 2/3 : Installation de pandas, torch, etc. (pip sync)..."
$UV_EXEC pip sync requirements.txt

# ÉTAPE 3: Exécuter le script (maintenant dans le .venv rempli)
echo "Étape 3/3 : Lancement du script Python..."
$UV_EXEC run python step1_create_categories.py \
    --input_csv "data/github_data_with_readmes.csv" \
    --output_dir "outputs/step1_clustering" \
    --n_categories 200 \
    --chunk_size 2000

echo "=== Fin du Job 1 ==="