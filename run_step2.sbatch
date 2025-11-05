#!/bin/bash
#SBATCH --job-name=step2_train_classifier
#SBATCH --output=slurm_logs/step2_training-%j.out # Fichier de sortie
#SBATCH --error=slurm_logs/step2_training-%j.err  # Fichier d'erreur
#SBATCH --partition=gpu-l40s                  # Partition
#SBATCH --gres=gpu:1                            # 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                   # Moins de CPUs
#SBATCH --mem=64G                           # Plus de RAM pour le modèle
#SBATCH --time=06:00:00

echo "=== Job 2 (Training) Démarré sur $SLURM_NODELIST ==="
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
export PATH="$HOME/.local/bin:$PATH"

# On se place dans le dossier de soumission
cd $SLURM_SUBMIT_DIR
echo "Dossier de travail : $(pwd)"

echo "Synchronisation des dépendances (réutilisation du .venv)..."
# 'uv' est intelligent : il verra que pyproject.toml et .venv
# existent déjà (créés par le Job 1) et les réutilisera.
uv pip sync requirements.txt

# --- 3. Exécution du script ---
echo "Lancement du script step2_train_classifier.py via 'uv run'..."
uv run python step2_train_classifier.py \
    --input_csv "inputs/github_data_with_readmes.csv" \
    --categories_file "outputs/step1_clustering/github_categories_database.json" \
    --output_model_dir "outputs/step2_classification/distilroberta_github_classifier" \
    --base_model "distilroberta-base" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 8

echo "=== Fin du Job 2 ==="