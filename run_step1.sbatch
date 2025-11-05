#!/bin/bash
#SBATCH --job-name=step1_create_categories
#SBATCH --output=slurm_logs/step1_categories-%j.out # Fichier de sortie
#SBATCH --error=slurm_logs/step1_categories-%j.err  # Fichier d'erreur
#SBATCH --partition=gpu-l40s                   # Partition (confirmé)
#SBATCH --gres=gpu:1                             # 1 GPU (confirmé par la doc)
#SBATCH --ntasks=1                               # 1 Tâche (confirmé par la doc)
#SBATCH --cpus-per-task=8                    # CPUs pour l'encodage
#SBATCH --mem=32G                            # RAM pour le clustering
#SBATCH --time=02:00:00                      # Limite de temps

echo "=== Job 1 (Clustering) Démarré sur $SLURM_NODELIST ==="
echo "Job ID : $SLURM_JOB_ID"
echo "GPU(s) alloué(s) : $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------"

# --- 1. Chargement des modules ---
# CORRIGE L'ERREUR 'python: commande introuvable'
echo "Chargement des modules (Python, CUDA)..."
module purge
module load python/3.11   # <-- !! REMPLACEZ-MOI (par un nom de 'module avail')
module load cuda/12.2     # <-- !! REMPLACEZ-MOI (par un nom de 'module avail')
echo "Modules chargés."

# --- 2. Configuration de l'environnement UV ---
# CORRIGE L'ERREUR 'uv: commande introuvable'
# Nécessaire si uv est installé via pip --user
export PATH="$HOME/.local/bin:$PATH"

# On se place dans le dossier où le job a été soumis
# (là où se trouvent vos scripts .py et requirements.txt)
cd $SLURM_SUBMIT_DIR
echo "Dossier de travail : $(pwd)"

echo "Initialisation du projet UV (si non-existant)..."
# Crée pyproject.toml si besoin, en se basant sur la doc
if [ ! -f "pyproject.toml" ]; then
  uv init --quiet
fi

echo "Synchronisation des dépendances de requirements.txt..."
# Crée/met à jour le .venv local et installe les paquets
uv pip sync requirements.txt

# --- 3. Exécution du script ---
echo "Lancement du script step1_create_categories.py via 'uv run'..."
# 'uv run' active l'environnement .venv juste pour cette commande
uv run python step1_create_categories.py \
    --input_csv "inputs/github_data_with_readmes.csv" \
    --output_dir "outputs/step1_clustering" \
    --n_categories 200 \
    --chunk_size 2000

echo "=== Fin du Job 1 ==="