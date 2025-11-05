#!/bin/bash
#SBATCH --job-name=step2_train_classifier     # Nom du job
#SBATCH --output=slurm_logs/step2_training.log # Fichier de log
#SBATCH --ntasks=1                          # 1 tâche
#SBATCH --cpus-per-task=4                   # 4 CPUs (sur 32 dispo)
#SBATCH --mem=64G                           # 64GB de RAM (sur 192G dispo)
#SBATCH --time=06:00:00                     # Temps max (6 heures)
#SBATCH --partition=gpu-l40s                # <--- CHANGEMENT
#SBATCH --gres=gpu:l40s:1                   # <--- CHANGEMENT (demande 1 GPU L40S sur les 2)

echo "=== Début du job Step 2 (Classification) sur morgoth ==="
date

# 1. Charger les modules Python et CUDA de base
# !! À ADAPTER : commandes spécifiques à votre cluster
# ex: module load python/3.11 cuda/12.2 (ou autre)
# Faites 'module avail' sur morgoth pour voir les noms exacts
# module purge
# module load ...

# 2. S'assurer que 'uv' est dans le PATH
export PATH="$HOME/.local/bin:$PATH"

# 3. Créer l'environnement virtuel local
echo "Création de l'environnement virtuel avec 'uv venv .venv'..."
uv venv .venv --seed

# 4. Activer l'environnement
source .venv/bin/activate

# 5. Installer les dépendances avec 'uv pip sync'
echo "Synchronisation des dépendances avec 'uv pip sync requirements.txt'..."
uv pip sync requirements.txt

# 6. Lancer le script Python
echo "Lancement du script step2_train_classifier.py..."
python step2_train_classifier.py \
    --input_csv "inputs/github_data_with_readmes.csv" \
    --categories_file "outputs/step1_clustering/github_categories_database.json" \
    --output_model_dir "outputs/step2_classification/distilroberta_github_classifier" \
    --base_model "distilroberta-base" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 8

echo "=== Fin du job Step 2 ==="
date