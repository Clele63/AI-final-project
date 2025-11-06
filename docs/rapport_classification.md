# Rapport d'Étape 2 : Classification Supervisée (Multi-Label)

Maintenant que nous avons 50 prototypes de catégories (un vecteur central et un nom pour "IA / ML", "Web Dev", etc.), nous entraînons un classifieur (`step2_train_classifier.py`) à les prédire.

## 5.1. La Logique "Multi-Label"

Nous ne voulons pas forcer un dépôt dans une seule catégorie. Un projet peut être à la fois IA et Web Dev.

L'approche "la catégorie la plus proche gagne" (argmax) est trop restrictive. Nous avons donc opté pour un **seuil de similarité**.

### Création des Labels (Multi-Label)

Pour chaque dépôt :

1. Calculer son embedding `all-MiniLM-L6-v2`
2. Calculer la similarité cosinus avec les 50 vecteurs prototypes de l'Étape 1
3. Si la similarité dépasse **0.6**, assigner le tag correspondant

Un dépôt peut recevoir 0, 1 ou N tags, transformant le problème en **classification multi-label**.

#### Concept de création des labels

```python
similarity_matrix = cosine_similarity(repo_embeddings, category_prototypes)
labels_multi_hot = (similarity_matrix > 0.6).astype(int)
# labels_multi_hot est maintenant un tableau [56641, 50]
# ex: [1, 0, 0, 1, 0, ...] signifie que le dépôt a les tags "0" et "3".
```

## 5.2. Fine-Tuning du Modèle (DistilRoBERTa)

Nous utilisons un modèle Transformer, `distilroberta-base`, léger et performant.

* Jeu de données : train (80%) / test (20%)
* Configuration du modèle pour **multi-label**

#### Concept du chargement du modèle

```python
from transformers import AutoModelForSequenceClassification

N_LABELS = 50  # Nos 50 catégories

model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=N_LABELS,
    problem_type="multi_label_classification"  # essentiel pour multi-label
)
```

* `problem_type="multi_label_classification"` :

  * pas de softmax global
  * sigmoïde indépendante sur chaque sortie
* Fonction de perte : **BCEWithLogitsLoss** (Binary Cross-Entropy)

## 6. Évaluation et Résultats

### 6.1. Performance de l'Entraînement

Les courbes de perte montrent une convergence saine : Training Loss et Validation Loss diminuent ensemble sur 3 époques. La validation ne remonte pas, indiquant un faible sur-apprentissage et une bonne généralisation.

### 6.2. Rapport de Classification

Le tableau suivant montre la performance finale du modèle sur l'ensemble de test (extraits) :

| Catégorie                        | Precision | Recall | F1-score | Support |
| -------------------------------- | --------- | ------ | -------- | ------- |
| Compression de Données           | 1.00      | 0.05   | 0.10     | 19      |
| Développement Blockchain         | 0.00      | 0.00   | 0.00     | 7       |
| Développement Cloud (AWS)        | 0.79      | 0.68   | 0.73     | 87      |
| Développement Cloud (Containers) | 0.70      | 0.79   | 0.74     | 86      |
| ...                              | ...       | ...    | ...      | ...     |
| IA / Machine Learning            | 0.84      | 0.81   | 0.82     | 5077    |
| ...                              | ...       | ...    | ...      | ...     |

**Micro avg** : 0.77, **Weighted avg** : 0.76, **Samples avg** : 0.73

> Note : La matrice de confusion N x N n'est pas pertinente en multi-label.

### Analyse des Résultats

* **Performance globale** : F1 weighted = 0.76 → bon équilibre entre précision et rappel
* **Haute performance (F1 > 0.80)** :

  * Développement Mobile (iOS)
  * Développement Web (Frontend)
  * IA / Machine Learning
* **Faible performance (F1 < 0.20)** :

  * Compression de Données (19)
  * Développement Blockchain (7)
  * Développement Mobile (Android) (18)
  * Développement Logiciel (Windows) (71)
* **Cas intéressants** :

  * Développement Web Backend (0.75) vs Frontend (0.83) bien séparés
  * Développement Logiciel (CMake) (0.34) et Programmation Système (Linux) (0.47) plus difficiles à distinguer

## 7. Conclusion

La classification multi-label avec 50 catégories est un succès :

* F1-score moyen = 0.76
* Identification correcte des thèmes sémantiques clairs
* Les faibles performances sont liées au faible support, pas au modèle

### Prochaines étapes possibles

* **Ajustement du seuil** : tester 0.55 ou 0.65 pour optimiser le nombre de tags par dépôt
* **Modèle plus grand** : remplacer `distilroberta-base` par `roberta-base` pour mieux distinguer catégories proches
* **Nettoyage avancé du texte** : retirer code, badges, tables pour améliorer la qualité des embeddings
