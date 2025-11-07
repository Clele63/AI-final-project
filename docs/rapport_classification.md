# Rapport d'Étape 2 : Classification Supervisée (Multi-Label)

Maintenant que nous avons notre "vérité terrain" de l'Étape 1 (un fichier `github_categories_database.json` contenant 50 prototypes de catégories nommés), nous pouvons passer à l'entraînement supervisé.

L'objectif de cette étape (`step2_train_classifier.py`) est d'entraîner un modèle Transformer à prédire ces 50 tags pour n'importe quel dépôt.

## 1. La Logique "Multi-Label" (La Clé du Projet)

La plus grande erreur serait de forcer chaque dépôt dans une seule catégorie (ex: "la catégorie la plus proche"). Un projet est presque toujours multi-thème.

Notre approche résout ce problème en créant des étiquettes (labels) multi-label avant l'entraînement.

### 1.1. Création des Labels (Seuil de Similarité)

Pour chaque dépôt de notre jeu de données, nous effectuons les opérations suivantes :

* Calculer son embedding (avec `all-MiniLM-L6-v2`, le même que l'étape 1).
* Calculer la similarité cosinus entre cet embedding et les 50 "vecteurs prototypes" (centres de cluster) de notre base de données.
* Appliquer un seuil : Si la similarité avec un prototype dépasse 0.6, nous assignons le tag correspondant.

Un dépôt peut donc recevoir 0, 1, ou N tags. C'est ce qui transforme notre problème en classification multi-label.

```python
# Concept de la création de labels (multi-label) dans step2
# repo_embeddings.shape: (56641, 384)
# category_prototypes.shape: (50, 384)

similarity_matrix = cosine_similarity(repo_embeddings, category_prototypes)
# similarity_matrix.shape: (56641, 50)

# Appliquer le seuil (0.6) pour obtenir les "tags"
labels_multi_hot = (similarity_matrix > 0.6).astype(int)

# labels_multi_hot est un tableau [56641, 50]
# ex: [1, 0, 0, 1, 0, ...] signifie que le dépôt a les tags "0" et "3".
```

Ce tableau `labels_multi_hot` devient notre "Y" (nos étiquettes cibles) pour l'entraînement supervisé.

## 2. Fine-Tuning du Modèle Transformer

### 2.1. Choix du Modèle : distilroberta-base

Nous utilisons un modèle Transformer, `distilroberta-base`. C'est un excellent compromis entre performance et ressources : il est plus léger et rapide que `roberta-base` tout en conservant une grande partie de sa performance. Le jeu de données est divisé en train (80%) et test (20%).

### 2.2. La Configuration Technique Essentielle

L'étape la plus importante du script `step2_train_classifier.py` est la configuration du modèle pour qu'il comprenne la nature "multi-label" de notre problème.

```python
from transformers import AutoModelForSequenceClassification

N_LABELS = 50 # Nos 50 catégories

model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=N_LABELS,
    problem_type="multi_label_classification"
)
```

Spécifier `problem_type="multi_label_classification"` n'est pas un détail. Cela change fondamentalement le comportement du modèle :

* **Architecture** : Il n'utilise pas une couche Softmax finale (qui force la somme des probabilités à 1 et ne permet qu'un seul gagnant).
* **Sortie** : Il applique une fonction Sigmoid indépendante sur chacun des 50 neurones de sortie. Chaque tag est traité comme une prédiction binaire (oui/non) indépendante.
* **Fonction de Perte** : Il utilise la perte BCEWithLogitsLoss (Binary Cross-Entropy), qui évalue chaque prédiction de tag (oui/non) individuellement, plutôt que CrossEntropyLoss (utilisée pour la classification mono-classe).

## 3. Évaluation et Analyse des Résultats

Après l'entraînement (3 époques), le Trainer de Hugging Face évalue le modèle sur l'ensemble de test (les 20% de données qu'il n'a jamais vues).

### 3.1. Performance de l'Entraînement

* **Artefact** : `outputs/step2_classification/training_plots.png`

Analyse : Les courbes de perte montrent une convergence saine du modèle. La Training Loss et la Validation Loss diminuent de concert au fil des 3 époques. La Validation Loss ne remonte pas, indiquant que le modèle ne tombe pas significativement en sur-apprentissage.

### 3.2. Rapport de Classification Détaillé

* **Artefact** : `outputs/step2_classification/classification_report_multilabel.txt`

Le rapport détaille la performance pour chaque tag (précision, rappel, F1-score et support).

**Analyse des Métriques** :

* **Performance Globale (F1-Score)** : weighted avg F1 = 0.76, très encourageant, pondéré pour tenir compte du déséquilibre des classes.
* **Haute performance (F1 > 0.80)** : iOS, Web Frontend, IA/Machine Learning.
* **Faible performance (F1 < 0.20)** : Blockchain, Android, Compression de données, Logiciel Windows. Principalement dû au faible nombre d'exemples.
* **Cas Intéressants** : Backend vs Frontend distingués correctement, CMake plus difficile à reconnaître.

## 4. Conclusion

Le passage à une approche multi-label avec 50 catégories est un succès. Le modèle identifie efficacement les thèmes clairs et à fort volume, avec un F1-score moyen de 0.76.

Les faibles performances sont presque exclusivement dues à un faible support dans le jeu de données.

### Prochaines étapes possibles

* Ajustement du seuil (ex: 0.55 ou 0.65) pour optimiser le compromis précision/rappel.
* Modèle plus grand (`roberta-base`) pour améliorer la distinction des catégories proches.
* Nettoyage avancé des README pour réduire le bruit et améliorer la qualité des embeddings.
