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
UV_EXEC="/home/lcoussy/.local/bin/uv"

echo "=== Job 2 (Training) Démarré ==="
echo "Utilisation de UV via : $UV_EXEC"
echo "----------------------------------------------------"

# --- Initialisation uv ---
cd $SLURM_SUBMIT_DIR
echo "Dossier de travail : $(pwd)"

# --- MÊME LOGIQUE (sécurisé) ---

# ÉTAPE 1: S'assurer que le .venv existe
echo "Étape 1/3 : Vérification du .venv..."
$UV_EXEC venv --python 3.12

# ÉTAPE 2: Synchroniser les dépendances (sera rapide)
echo "Étape 2/3 : Synchronisation des dépendances..."
$UV_EXEC pip sync requirements.txt

# ÉTAPE 3: Exécuter le script
echo "Étape 3/3 : Lancement du script Python..."
$UV_EXEC run python step2_train_classifier.py \
    --input_csv "inputs/github_data_with_readmes.csv" \
    --categories_file "outputs/step1_clustering/github_categories_database.json" \
    --output_model_dir "outputs/step2_classification/distilroberta_github_classifier" \
    --base_model "distilroberta-base" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 8

echo "=== Fin du Job 2 ==="