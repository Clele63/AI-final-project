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

## 2. Nommage des catégories

Après le clustering, nous obtenons 50 groupes représentés par leurs prototypes vectoriels. Les identifiants numériques (0 à 49) étant peu parlants, un pipeline d'étiquetage automatique leur attribue des noms lisibles.

---

### 2.1. Étape A : Extraction de mots-clés (TF-IDF)

Pour chaque cluster, nous regroupons jusqu'à 200 textes issus des dépôts associés.

Un TfidfVectorizer est appliqué sur ces textes pour extraire les 15 mots-clés les plus distinctifs du cluster. Ces mots-clés servent de résumé sémantique initial.

---

### 2.2. Étape B : Génération du nom (Phi-3)

Les mots-clés bruts ne sont pas utilisés tels quels. Ils sont injectés dans un prompt structuré destiné à un modèle local, **microsoft/Phi-3-mini-4k-instruct**, chargé via *transformers*.

Le modèle, configuré en *text-generation*, reçoit les mots-clés accompagnés d'une instruction : produire un nom de catégorie technique clair, unique et professionnel (exemples : "IA / Machine Learning", "DevOps & Infrastructure Cloud").

Le nom généré est retenu comme **nom final de la catégorie**.

---

### 2.3. Sorties et analyse

Les données suivantes sont sauvegardées dans : `outputs/step1_clustering/cluster_top_keywords.json` :

* les mots-clés TF-IDF (`name`),
* le nom généré par le LLM (`auto_name`),
* le nombre de dépôts par cluster (`repo_count`).

Le champ `auto_name` est également repris comme `category_name` dans : `outputs/step1_clustering/github_categories_database.json`, point de départ pour l'étape 2.

**Exemples de sortie (cluster_top_keywords.json)** :

```json
"Cluster 0": {
    "name": "video, youtube, img, io, http, src, code, tv, download, api, www, use, license, player, href",
    "auto_name": "Générique / Autre",
    "repo_count": 480
},
"Cluster 1": {
    "name": "ai, model, agent, api, models, openai, org, run, python, code, bash, img, arxiv, md, agents",
    "auto_name": "IA / Machine Learning",
    "repo_count": 809
},
"Cluster 2": {
    "name": "dart, src, pub, app, io, packages, png, width, dev, master, ui, td, href, widget, shields",
    "auto_name": "Développement Web (Frontend)",
    "repo_count": 327
},
"Cluster 3": {
    "name": "js, http, javascript, html, jquery, code, css, org, web, license, use, io, element, browser, build",
    "auto_name": "Développement Web (Frontend)",
    "repo_count": 915
},
"Cluster 4": {
    "name": "crates, cargo, license, img, docs, let, svg, run, shields, build, org, main, install, badge, src",
    "auto_name": "Développement Rust",
    "repo_count": 761
}
```

**Analyse** :
Cette approche entièrement automatique fournit des noms de catégories cohérents et pertinents. Le LLM filtre les mots parasites remontés par TF-IDF (comme "http", "img", "io") et produit un label sémantique propre.

2 graphique de distribution sont générés : 
- `outputs/step1_clustering/cluster_distribution.png`

![alt](/outputs/step1_clustering/cluster_distribution.png)

- `outputs/step1_clustering/theme_distribution.png`.

![alt](/outputs/step1_clustering/theme_distribution.png)

La distribution reste très déséquilibrée. L'étape 2 appliquera donc une évaluation pondérée (*weighted avg*).

**Analyse :**

* Les clusters ne sont pas équilibrés
* Quelques "méga-clusters" regroupent la majorité des dépôts (ex : "Développement Web (Frontend)", "IA / Machine Learning")
* Certains clusters très petits (ex : "Développement Blockchain")
* Cette distribution justifie l'utilisation d'une **moyenne pondérée (weighted avg)** pour évaluer le classifieur de l'étape 2.
