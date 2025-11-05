#!/bin/sh
# Ce script lance le pipeline en deux étapes,
# en s'assurant que l'étape 2 ne démarre que si l'étape 1 a réussi.

set -e

# 1. Lancer l'étape 1 (Clustering)
# --parsable force sbatch à ne renvoyer que le Job ID
echo "Lancement du Job 1 (step1_create_categories.py)..."
JOB1_ID=$(sbatch --parsable run_step1.sbatch)

echo "Job 1 soumis. ID : ${JOB1_ID}"

# 2. Lancer l'étape 2 (Entraînement)
# On utilise $JOB1_ID pour créer la dépendance
echo "Lancement du Job 2 (step2_train_classifier.py) en attente de la réussite du Job 1..."
JOB2_ID=$(sbatch --parsable --dependency=afterok:${JOB1_ID} run_step2.sbatch)

echo "Job 2 soumis. ID : ${JOB2_ID}"
echo "Le pipeline est lancé."