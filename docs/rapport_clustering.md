# Rapport d'Étape 1 : Découverte des Catégories (Clustering)

Cette première étape (step1_create_categories.py) vise à répondre à la question : "Quels sont les 50 thèmes principaux présents dans nos 56 000 dépôts ?"

## 4.1. Vectorisation Sémantique

Nous ne pouvons pas utiliser directement le texte. Nous devons le transformer en vecteurs numériques (embeddings) qui représentent le "sens".

### Choix du modèle : all-MiniLM-L6-v2

Nous avons choisi ce modèle de Sentence-Transformers car il offre un excellent équilibre entre vitesse et performance. Il est conçu pour générer des embeddings de phrases de haute qualité, ce qui est parfait pour capturer la sémantique d'une description ou d'un README.

### Défi de Mémoire : MiniBatchKMeans

Le jeu de données (plusieurs Go) ne tient pas en RAM. Nous ne pouvons pas charger tous les embeddings d'un coup. Pour contourner ce problème, le script lit le CSV par morceaux (chunks) de 2000 lignes.

Sur chaque morceau, nous entraînons de manière incrémentale un modèle MiniBatchKMeans, qui est conçu pour apprendre "par lots" sans nécessiter l'ensemble des données en mémoire.

### Concept du script step1_create_categories.py

```python
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

kmeans = MiniBatchKMeans(n_clusters=50, n_init='auto', batch_size=256)

for chunk in pd.read_csv("github_data_with_readmes.csv", chunksize=2000):
    chunk['full_text'] = chunk['description'].fillna('') + ' ' + chunk['readme_content'].fillna('')
    embeddings = embedding_model.encode(chunk['full_text'].tolist())
    kmeans.partial_fit(embeddings)
```

## 4.2. Nommage des Catégories (Le défi "Non Supervisé")

Après l'étape 4.1, nous avons 50 clusters, mais ce ne sont que des numéros (de 0 à 49). Pour les rendre utiles, nous devons leur donner un nom lisible.

### Analyse TF-IDF

Pour chaque cluster, le script regroupe tous les textes qui lui appartiennent et effectue une analyse TF-IDF. Cela permet d'extraire les mots-clés les plus importants et distinctifs de ce groupe.
Le script sauvegarde ces mots-clés dans outputs/step1_clustering/cluster_top_keywords.json.

Exemple de sortie (pour "Cluster 1") : ["ai", "model", "torch", "llm", "agent"]

### Nommage Manuel

Nous utilisons ce JSON comme un dictionnaire pour remplacer manuellement les noms dans outputs/step1_clustering/github_categories_database.json.

**Action manuelle** : Cluster 1 - ai, model, torch... → "IA / Machine Learning"

Ce processus semi-manuel est crucial : il combine la puissance de la découverte non supervisée (basée sur les données) avec l'intelligence humaine (pour créer des noms intelligibles).

## 4.3. Évaluation (Étape 1)

La sortie principale est github_categories_database.json, qui contient les 50 "vecteurs prototypes" (le centre de chaque cluster) et leurs noms lisibles.

Nous générons aussi un graphique de distribution pour analyser la répartition des données :

### Liste (partielle) des thèmes générés (après nommage manuel) :

* Compression de Données
* Développement Blockchain
* Développement Cloud (AWS)
* Développement Cloud (Containers)
* Développement Logiciel
* Développement Mobile (iOS)
* Développement Rust
* Développement Web (Backend)
* Développement Web (Frontend)
* IA / Machine Learning

### Analyse

Le graphique confirme que les clusters ne sont pas équilibrés. On observe quelques "méga-clusters" (ex: Cluster 5, 12, 30) qui regroupent des thèmes très généraux (comme "outils de développement" ou "projets web"). À l'inverse, de nombreux clusters sont très petits, indiquant des niches sémantiques très spécifiques (ex: "Développement Blockchain").

Cette distribution inégale est normale et justifie l'utilisation d'une moyenne pondérée (weighted avg) lors de l'évaluation du classifieur à l'étape 2.
