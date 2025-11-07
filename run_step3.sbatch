#!/bin/bash
#SBATCH --job-name=step3_train_generator
#SBATCH --output=slurm_logs/step3_generating-%j.out
#SBATCH --error=slurm_logs/step3_generating-%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

# --- Config ---

unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH" 
UV_PROJECT=$HOME/slurm/ai-final-project/step3


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

uv pip install -r /home/lcoussy/AI-final-project/requirements.txt

export TRANSFORMERS_VERBOSITY=info
DATA_FILE="/home/lcoussy/AI-final-project/data/github_data_with_readmes.csv"
CATEG_DB_FILE="/home/lcoussy/AI-final-project/outputs/step1_clustering/github_categories_database.json"
LABELS_FILE="/home/lcoussy/AI-final-project/outputs/step2_classification/labels_multi_hot.npy"
THEMES_FILE="/home/lcoussy/AI-final-project/outputs/step2_classification/themes_list.txt"
OUTPUT_DIR="/home/lcoussy/AI-final-project/outputs/step3_generator/distilgpt2_github_generator"

echo "Étape 3/3 : Lancement du script Python (Generate Labels)..."
uv run python /home/lcoussy/AI-final-project/generate_labels_for_step3.py \
  --data_file "$DATA_FILE" \
  --categories_db_file "$CATEG_DB_FILE" \
  --output_file "$LABELS_FILE" \
  --embedding_model "all-MiniLM-L6-v2" \
  --similarity_threshold 0.6 \
  --chunk_size 1000


# --- Hyperparamètres (vous pouvez les ajuster ici ou dans le .sbatch) ---
MODEL_NAME="distilgpt2"
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=5e-5

echo "Étape 3/3 : Lancement du script Python (Training)..."
uv run python /home/lcoussy/AI-final-project/step3_train_generator.py \
    --data_file "$DATA_FILE" \
    --labels_file "$LABELS_FILE" \
    --categories_db_file "$CATEG_DB_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE


echo "=== Fin du Job $SLURM_JOB_ID ==="