#!/bin/bash
#SBATCH --job-name=step1_create_categories # Nom du job
#SBATCH --output=slurm_logs/step1_categories.log # Fichier de log
#SBATCH --ntasks=1                          # 1 tâche
#SBATCH --cpus-per-task=8                   # 8 CPUs (sur 32 dispo)
#SBATCH --mem=32G                           # 32GB de RAM (sur 192G dispo)
#SBATCH --time=02:00:00                     # Temps max (2 heures)
#SBATCH --partition=gpu-l40s                # <--- CHANGEMENT
#SBATCH --gres=gpu:l40s:1                   # <--- CHANGEMENT (demande 1 GPU L40S sur les 2)

echo "=== Début du job Step 1 (Clustering) sur morgoth ==="
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
echo "Lancement du script step1_create_categories.py..."
python step1_create_categories.py \
    --input_csv "inputs/github_data_with_readmes.csv" \
    --output_dir "outputs/step1_clustering" \
    --n_categories 200 \
    --chunk_size 2000

echo "=== Fin du job Step 1 ==="
date