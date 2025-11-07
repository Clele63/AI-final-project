# Rapport d'Étape 2 : Classification Supervisée (Multi-Label)

Avec les catégories nommées issues de l'étape 1, nous pouvons entraîner un modèle supervisé capable de prédire plusieurs thèmes pour chaque dépôt.

---

## 1. Logique multi-label

Un dépôt peut appartenir à plusieurs thèmes. Pour capturer cette réalité, nous créons des labels multi-label basés sur la similarité entre chaque dépôt et les prototypes de catégories.

### 1.1. Création des labels (seuil de similarité)

Pour chaque dépôt :

* calcul de l'embedding avec all-MiniLM-L6-v2,
* calcul des similarités cosinus avec les prototypes des catégories,
* application d'un seuil de 0.6.

Si la similarité dépasse 0.6, le tag est attribué. Un dépôt peut recevoir 0, 1 ou plusieurs tags.

```python
similarity_matrix = cosine_similarity(repo_embeddings, category_prototypes)
labels_multi_hot = (similarity_matrix > 0.6).astype(int)
```

Ce tableau devient la cible Y pour l'entraînement.

---

## 2. Fine-tuning du modèle Transformer

### 2.1. Modèle : distilroberta-base

Nous utilisons distilroberta-base, un bon compromis entre vitesse et performance. Les données sont séparées en 80% entraînement et 20% test.

### 2.2. Configuration multi-label

Le modèle est configuré pour la classification multi-label.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=N_LABELS,
    problem_type="multi_label_classification"
)
```

Conséquences :

* pas de Softmax final,
* Sigmoid indépendante par label,
* perte BCEWithLogitsLoss.

---

## 3. Évaluation et analyse

Après 3 époques d'entraînement, les résultats sont analysés.

### 3.1. Courbes de perte

Les courbes montrent une convergence saine. La perte de validation ne remonte pas.

![alt](/outputs/step2_classification/training_plots.png)

### 3.2. Rapport détaillé

Rapport textuel : `classification_report_multilabel.txt`.

```markdown
                                                                precision    recall  f1-score   support

                                        Compression de Données       1.00      0.05      0.10        19
                                      Développement Blockchain       0.00      0.00      0.00         7
                                     Développement Cloud (AWS)       0.79      0.68      0.73        87
                              Développement Cloud (Containers)       0.70      0.79      0.74        86
                          Développement Cloud (Infrastructure)       0.63      0.36      0.46        95
                              Développement Cloud (Kubernetes)       0.84      0.74      0.79       137
                                        Développement Logiciel       0.75      0.76      0.76       958
                                Développement Logiciel (CMake)       0.50      0.26      0.34       114
                                  Développement Logiciel (Git)       0.68      0.63      0.65       161
                              Développement Logiciel (Windows)       0.50      0.07      0.12        71
                                Développement Mobile (Android)       0.00      0.00      0.00        18
                                    Développement Mobile (iOS)       0.86      0.87      0.86       346
                                       Développement Robotique       0.63      0.57      0.60        42
                                            Développement Rust       0.87      0.70      0.78       219
                               Développement Système (Windows)       0.70      0.57      0.62       129
                                        Développement Terminal       0.54      0.30      0.38        50
                                   Développement Web (Backend)       0.76      0.74      0.75      2445
                                  Développement Web (Frontend)       0.84      0.83      0.83      4396
                                             Générique / Autre       0.70      0.59      0.64      1540
                                         IA / Machine Learning       0.84      0.81      0.82      5077
                                                  IA / Musique       0.68      0.84      0.75       140
                                            IA / Réseaux / DNS       0.75      0.55      0.64        49
                                                 IA / Sécurité       0.60      0.49      0.54       168
Ici, les mots-clés sont principalement liés à HTML et CSS, qui       0.66      0.47      0.55       552
                                 Programmation Système (Linux)       0.58      0.40      0.47        94

                                                     micro avg       0.80      0.74      0.77     17000
                                                     macro avg       0.66      0.52      0.56     17000
                                                  weighted avg       0.79      0.74      0.76     17000
                                                   samples avg       0.75      0.75      0.73     17000
```

Graphiques complémentaires :

* F1-score par thème : `f1_score_barplot.png`,

![alt](/outputs/step2_classification/f1_score_barplot.png)

* support par thème : `support_distribution_barplot.png`.

![alt](/outputs/step2_classification/support_distribution_barplot.png)

#### Analyse

Une corrélation forte apparaît entre le support d'un thème et son F1-score.

* **Score pondéré global** : 0.76.
* **F1 élevés** (0.82 à 0.86) pour les thèmes à haut volume : IA / Machine Learning, Développement Web (Frontend), Développement Mobile (iOS).
* **F1 faibles** (< 0.20) pour les catégories à très faible support, comme Blockchain ou Android.
* Un label mal parsé provenant de l'étape 1 apparaît dans le rapport, ce qui souligne la nécessité d'un nettoyage plus strict des sorties LLM.

---

## 4. Conclusion

Le passage au multi-label fonctionne bien. Le modèle identifie avec précision les thèmes bien représentés. Les faiblesses viennent presque toujours du manque de données.

### Pistes d'amélioration

* ajuster le seuil de similarité (0.55 à 0.65),
* tester roberta-base,
* nettoyer davantage les README,
* fusionner ou filtrer les catégories ayant moins de 50 exemples.
* essayer avec des catégories pré-fixé dès le départ

---

## Annexe : Liste des 30 thèmes uniques

```markdown
Liste des thèmes uniques générés :
==============================
- Compression de Données
- Développement Blockchain
- Développement Cloud (AWS)
- Développement Cloud (Containers)
- Développement Cloud (Infrastructure)
- Développement Cloud (Kubernetes)
- Développement Logiciel
- Développement Logiciel (CMake)
- Développement Logiciel (Git)
- Développement Logiciel (Windows)
- Développement Mobile (Android)
- Développement Mobile (iOS)
- Développement Robotique
- Développement Rust
- Développement Système (Windows)
- Développement Terminal
- Développement Web (Backend)
- Développement Web (Frontend)
- Générique / Autre
- IA / Machine Learning
- IA / Musique
- IA / Réseaux / DNS
- IA / Sécurité
- Ici, les mots-clés sont principalement liés à HTML et CSS, qui
- Programmation Système (Linux)
```

► [*Lire le rapport d'étape 1 : Clustering*](/docs/rapport_clustering.md)

► [*Lire le rapport d'étape 3 : Génération*](/docs/rapport_generation.md)

---

► [*Retour*](/README.md)