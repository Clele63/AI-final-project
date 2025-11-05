# Projet d'IA : Classification de Dépôts GitHub

Ce projet implémente un pipeline complet pour la classification de dépôts GitHub. Il utilise une approche en deux étapes : d'abord, un clustering non supervisé pour **découvrir** des catégories pertinentes, puis une classification supervisée pour **entraîner** un modèle capable de prédire ces catégories.

L'ensemble du pipeline est conçu pour être exécuté sur un cluster de calcul via **SLURM** et utilise **`uv`** pour une gestion d'environnement rapide et reproductible.

## 🚀 Architecture du Pipeline

Le projet est divisé en deux jobs SLURM principaux, gérés par `run_pipeline.sh` :

1.  **Étape 1 : Création des Catégories (`step1_create_categories.py`)**
    * **Entrée** : `github_data_with_readmes.csv`
    * **Tâche** : Utilise `all-MiniLM-L6-v2` pour générer des embeddings et `MiniBatchKMeans` pour créer 200 clusters.
    * **Sortie** : `github_categories_database.json` (la base de données des catégories) et des graphiques d'analyse (`cluster_distribution.png`).

2.  **Étape 2 : Entraînement du Classifieur (`step2_train_classifier.py`)**
    * **Entrée** : `github_data_with_readmes.csv` + le JSON de l'étape 1.
    * **Tâche** : "Fine-tune" un modèle `distilroberta-base` pour la classification de séquences sur le GPU.
    * **Sortie** : Le modèle entraîné (`distilroberta_github_classifier/`), un rapport de classification (`classification_report.txt`) et des graphiques de performance (`training_plots.png`, `confusion_matrix.png`).

---

## 📋 Prérequis

Avant de lancer le pipeline, assurez-vous de :

1.  Avoir installé `uv` (ex: `pip install --user uv`).
2.  Avoir `git-lfs` installé pour récupérer le jeu de données.

## ⚡ Guide de Lancement Rapide

1.  **Récupérer les données (Git LFS)**
    Assurez-vous que le fichier CSV de données est bien téléchargé (et n'est pas juste un pointeur Git LFS) :
    ```bash
    git lfs pull
    ```

2.  **Adapter les scripts SBATCH**
    Vérifiez que les scripts `run_step1.sbatch` et `run_step2.sbatch` ciblent la bonne partition SLURM (`--partition=...`) et utilisent le bon chemin absolu vers `uv`.

3.  **Lancer le Pipeline**
    Rendez le script de lancement exécutable et lancez-le :
    ```bash
    chmod +x run_pipeline.sh
    ./run_pipeline.sh
    ```

4.  **Suivre l'exécution**
    Vous pouvez suivre la file d'attente avec `squeue -u $USER` et voir les sorties en direct avec :
    ```bash
    tail -f slurm_logs/step1_categories-*.out
    tail -f slurm_logs/step2_training-*.out
    ```

---

## 📚 Rapport Complet

Pour une analyse détaillée de la méthodologie, des défis rencontrés et des résultats, consultez le rapport complet :

**[Lire le rapport complet](./docs/rapport.md)**