# Projet d'IA : Tagging Multi-Label de Dépôts GitHub

Ce projet implémente un pipeline complet pour la **classification multi-label** (tagging) de dépôts GitHub.

L’objectif est de *tagguer automatiquement* un dépôt (ex : **IA**, **Web Dev**) en se basant sur le texte de son `README.md` et de sa description.

---

## Contexte et Données

Pour comprendre la problématique, la source des données et le prétraitement appliqué, commencez par lire le document de contexte.

► [*Lire le contexte du projet et l’analyse des données*](./docs/ctx-data.md)


## Architecture du Pipeline

Le projet est divisé en deux grandes étapes, chacune documentée en détail :

### Étape 1 : Découverte des Catégories (Clustering)

Un modèle non supervisé (`all-MiniLM-L6-v2` + `MiniBatchKMeans`) analyse **56 000 dépôts** pour *découvrir* **50 thèmes sémantiques**.

► [*Lire le rapport d'étape 1 : Clustering*](./docs/rapport_clustering.md)

### Étape 2 : Entraînement du Classifieur (Multi-Label)

Un modèle supervisé (`distilroberta-base`) est entraîné à prédire ces **50 thèmes** en tant que *tags* (un dépôt peut en avoir plusieurs).

► [*Lire le rapport d'étape 2 : Classification*](./docs/rapport_classification.md)
