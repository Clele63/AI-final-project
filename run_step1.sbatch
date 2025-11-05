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

UV_PROJECT=$HOME/slurm/ai-final-project/step1
UV_EXEC="/home/lcoussy/.local/bin/uv"

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
  $UV_EXEC init --python 3.12 --quiet
fi

# Installe PyTorch 
$UV_EXEC pip sync /home/lcoussy/AI-final-project/requirements.txt # Rien ne vous empeche d'utiliser un fichier contenant vos dépendances.

# --- Exécution du code ---
# echo "=== Test PyTorch sur GPU (via uv) ==="
# uv run python - <<'PYCODE'
# import torch, time

# print("Version PyTorch :", torch.__version__)
# print("CUDA disponible :", torch.cuda.is_available())
# print("GPU détecté :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun")

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     a = torch.randn((5000, 5000), device=device)
#     b = torch.randn((5000, 5000), device=device)
#     torch.cuda.synchronize()
#     start = time.time()
#     c = torch.matmul(a, b)
#     torch.cuda.synchronize()
#     print(f"Multiplication terminée en {time.time() - start:.4f} secondes sur le GPU.")
# else:
#     print("⚠️ Aucune carte GPU détectée par PyTorch.")
# PYCODE


# echo "=== Job 1 (Clustering) Démarré ==="
# echo "Utilisation de UV via : $UV_EXEC"
# echo "----------------------------------------------------"

# # --- Initialisation uv ---
# cd $SLURM_SUBMIT_DIR
# echo "Dossier de travail : $(pwd)"

# # --- LA CORRECTION (en 3 étapes) ---

# # ÉTAPE 1: Créer le .venv explicitement (comme le log l'a suggéré)
# echo "Étape 1/3 : Création du .venv avec Python 3.12..."
# $UV_EXEC venv --python 3.12

# # ÉTAPE 2: Installer les paquets DANS le .venv
# echo "Étape 2/3 : Installation de pandas, torch, etc. (pip sync)..."
# $UV_EXEC pip sync requirements.txt

# ÉTAPE 3: Exécuter le script (maintenant dans le .venv rempli)
echo "Étape 3/3 : Lancement du script Python..."
$UV_EXEC run python /home/lcoussy/AI-final-project/step1_create_categories.py \
    --input_csv "/home/lcoussy/AI-final-project/data/github_data_with_readmes.csv" \
    --output_dir "/home/lcoussy/AI-final-project/outputs/step1_clustering" \
    --n_categories 200 \
    --chunk_size 2000

echo "=== Fin du Job 1 ==="