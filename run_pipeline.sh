#!/bin/sh
# Script qui lance le pipeline en deux étapes, en s'assurant
# que l'étape 2 ne démarre qu'après la fin de l'étape 1.

set -e

# 1. Lancer l'étape 1 (Clustering)
echo "Lancement du Job 1 (step1_create_categories.py)..."
JOB1_ID=$(sbatch --parsable run_step1.sbatch)
echo "Job 1 soumis. ID : ${JOB1_ID}"

# Attendre que Job 1 soit terminé
echo "Attente de la fin du Job 1..."
while true; do
    STATUS=$(squeue -j ${JOB1_ID} -h -o "%T")
    if [ -z "$STATUS" ]; then
        echo "Job 1 terminé."
        break
    else
        echo "Job 1 en cours (${STATUS})... attente 30s"
        sleep 30
    fi
done

# Créer un venv spécifique à l'étape 2 si nécessaire
# echo "Création du venv pour l'étape 2..."
# python3 -m venv venv_step2
# source venv_step2/bin/activate
# pip install -r requirements_step2.txt

# 2. Lancer l'étape 2 (Entraînement)
echo "Lancement du Job 2 (step2_train_classifier.py)..."
JOB2_ID=$(sbatch --parsable run_step2.sbatch)
echo "Job 2 soumis. ID : ${JOB2_ID}"

# 3. Lancer l'étape 3 (Entraînement)
echo "Lancement du Job 3 (step3_train_generator.py)..."
JOB3_ID=$(sbatch --parsable run_step3.sbatch)
echo "Job 3 soumis. ID : ${JOB3_ID}"

echo "Le pipeline est lancé."
