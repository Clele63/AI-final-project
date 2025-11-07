# Rapport d'Étape 1 : Découverte des Catégories (Clustering)

Cette première étape, exécutée par `step1_create_categories.py`, est entièrement non supervisée. Son objectif est de répondre à la question : "Quels sont les 50 thèmes sémantiques principaux présents dans nos 56 000 dépôts ?"

Le résultat de cette étape constitue la **base de connaissances** de nos catégories, qui servira de vérité terrain (ground truth) pour l'étape 2.

## 1. Démarche de Clustering

### 1.1. Vectorisation Sémantique

Pour regrouper les dépôts par "thème", nous transformons leur texte (`full_text`) en vecteurs numériques (**embeddings**) qui représentent leur sens.

* **Choix du modèle :** `all-MiniLM-L6-v2` (Sentence-Transformers)

  * Rapide et léger
  * Produits des embeddings de 384 dimensions
  * Capturent le sens global des phrases ou paragraphes, parfait pour description et README

### 1.2. Le Défi de la Mémoire : Clustering Incrémental

**Problème :** le jeu de données est trop volumineux pour être chargé en RAM. Un K-Means standard n'est pas viable.

**Solution :** MiniBatchKMeans (clustering incrémental)

* Lecture du CSV par morceaux (chunks) de 2000 lignes
* Génération des embeddings pour chaque chunk
* Mise à jour des centres des clusters "à la volée"

#### Concept clé (step1_create_categories.py)

```python
# 1. Initialiser le modèle d'embedding (GPU)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialiser le clustering pour 50 thèmes
kmeans = MiniBatchKMeans(n_clusters=50, n_init='auto', batch_size=256)

# 3. Boucler sur les chunks du CSV
for chunk in pd.read_csv("github_data_with_readmes.csv", chunksize=2000):
    chunk['full_text'] = chunk['description'].fillna('') + ' ' + chunk['readme_content'].fillna('')
    embeddings = embedding_model.encode(chunk['full_text'].tolist())
    kmeans.partial_fit(embeddings)
```

## 2. Nommage des Catégories (Semi-Manuel)

Après clustering, nous avons 50 clusters. Ce sont des prototypes (vecteurs centraux) mais les numéros seuls (0 à 49) ne sont pas lisibles.

### 2.1. Sortie Automatique : TF-IDF

* Pour chaque cluster, regrouper les textes et effectuer une analyse TF-IDF
* Extraire les **5 mots-clés les plus distinctifs**
* Sauvegarde dans : `outputs/step1_clustering/cluster_top_keywords.json`

Exemple :

```json
{
  "Cluster 1": {"name": "ai, model, torch, llm, agent", "repo_count": 5077},
  "Cluster 2": {"name": "ios, swift, apple, xcode, mobile", "repo_count": 346}
}
```

### 2.2. Action Manuelle : Nommage

* Éditer `github_categories_database.json` pour donner un **nom clair** à chaque cluster
* Exemple :

```json
"category_name": "IA / Machine Learning"  # pour Cluster 1
```

* Cette approche **semi-manuelle** garantit des catégories intelligibles et pertinentes pour un humain.

## 3. Analyse des Sorties

* Artefact principal : `github_categories_database.json` (50 prototypes et noms lisibles)
* Graphique de distribution : `outputs/step1_clustering/cluster_distribution.png`

**Analyse :**

* Les clusters ne sont pas équilibrés
* Quelques "méga-clusters" regroupent la majorité des dépôts (ex : "Développement Web (Frontend)", "IA / Machine Learning")
* Certains clusters très petits (ex : "Développement Blockchain")
* Cette distribution justifie l'utilisation d'une **moyenne pondérée (weighted avg)** pour évaluer le classifieur de l'étape 2.
