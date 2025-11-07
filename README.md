# Projet d'IA : Tagging Multi-Label de Dépôts GitHub

Ce projet implémente un pipeline complet pour la **classification multi-label** (tagging) de dépôts GitHub.

L’objectif est de *tagguer automatiquement* un dépôt (ex : **IA**, **Web Dev**) en se basant sur le texte de son `README.md` et de sa description.

---

## Note sur la génération

Le cœur du travail repose sur la classification (Étapes 1 et 2). Dû à la complexité de la classification, la génération (Étape 3) n'a été ajoutée qu'à la fin et n'est donc en soi pas abouti.

Raisons de la priorité donnée à la classification :

* **tâche multi-label**, plus complexe qu'une classification binaire,
* **tags non prédéfinis**, découverts automatiquement par clustering,
* pipeline complet d'étiquetage dynamique avant l'entraînement supervisé.

---

## Contexte et Données

Pour comprendre la problématique, la source des données et le prétraitement appliqué, commencez par lire le document de contexte.

► [*Lire le contexte du projet et l’analyse des données*](./docs/ctx-data.md)

---

## Pré‑requis : Chargement du Modèle (git‑lfs)

Le modèle de classification est volumineux et donc stocké via **Git Large File Storage (git‑lfs)**.

Avant toute utilisation du classifieur, vous devez installer et initialiser git‑lfs, puis télécharger les poids du modèle :

```bash
sudo apk add git-lfs # Adapter à l'OS
git-lfs install
git-lfs pull
```

Sans cette étape, les fichiers du modèle ne seront pas présents et le classifieur ne pourra pas être chargé.

---

## Architecture du Pipeline

Le projet est divisé en deux grandes étapes (+ la génération), chacune documentée en détail :

### Étape 1 : Découverte des Catégories (Clustering)

Un modèle non supervisé (`all-MiniLM-L6-v2` + `MiniBatchKMeans`) analyse **56 000 dépôts** pour *découvrir* **100 thèmes sémantiques**. Un **LLM local** (`Phi-3`) nomme *automatiquement* ces thèmes.

► [*Lire le rapport d'étape 1 : Clustering*](./docs/rapport_clustering.md)

### Étape 2 : Entraînement du Classifieur (Multi-Label)

Un modèle supervisé (`distilroberta-base`) est entraîné à prédire ces **thèmes** en tant que *tags* (un dépôt peut en avoir plusieurs).

► [*Lire le rapport d'étape 2 : Classification*](./docs/rapport_classification.md)

### Étape 3 : Entraînement du Générateur (Génération)

Un modèle génératif (distilgpt2) est fine‑tuné pour produire une description à partir d'une liste de tags.

► [*Lire le rapport d'étape 3 : Génération*](./docs/rapport_generation.md)
