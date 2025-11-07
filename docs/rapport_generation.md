# Rapport d'Étape 3 : Génération de Description

Cette étape couvre la partie génération du pipeline (`step3_train_generator.py`). Comme expliqué dans le README, elle a été implémentée en fin de projet et reste nettement moins aboutie que les étapes 1 et 2.

L'objectif est de fine-tuner un modèle capable de produire une description de projet cohérente en fonction des tags prédits.

Exemple attendu :

```
TAGS: [IA / Machine Learning, Développement Web (Backend)] DESCRIPTION:
Une API Flask simple pour servir un modèle Pytorch de classification d'images.
```

---

## 1. Stratégie : Fine-tuning d'un modèle causal

Nous utilisons un modèle pré-entraîné de type GPT-2.

* **Modèle de base** : distilgpt2
* **Format d'entraînement** :

```
TAGS: [tag1, tag2] DESCRIPTION: Texte du dépôt<|endoftext|>
```

* **Sources des données** : descriptions courtes du CSV et liste des thèmes (~25) obtenus après fusion des catégories de l'étape 2.

Le modèle apprend à compléter le texte après "DESCRIPTION:" en se basant sur les tags placés avant.

---

## 2. Problèmes rencontrés et limites

### Problème 1 : contamination multilingue

Lors des premiers tests, les générations mélangeaient plusieurs langues : anglais, chinois, etc...

Exemple obtenu :

```
A powerful Python utility to manipulate AI models using deep learning methods. 一个简单生成组件端轻量社器，一款加数据技术持程序构、推及括芘式、编码源论商展、视频、功能从分汇总类、开洁、创微信、安即到等可发支付、号引网变任的比用提例点脚本建础、小平台、无机力模型、基于红链、页爱别古种面楨日、�
```

**Cause** :

* distilgpt2 est un modèle anglophone,
* le dataset contient des descriptions multilingues,
* le tokenizer anglais découpe les caractères non latins en tokens inconnus.

Résultat : le modèle apprend que des tokens <UNK> peuvent suivre "DESCRIPTION:", et les reproduit.

---

### Problème 2 : une étape non aboutie

L'entraînement souffre de plusieurs limites, principalement dues au manque de temps.

* **Données trop courtes** : seules les descriptions du CSV sont utilisées. Elles font souvent moins de 20 mots. L'idéal aurait été d'intégrer le contenu des README, mais cela requiert un pipeline plus complexe (nettoyage, découpe, gestion du contexte long).
* **Modèle très petit** : distilgpt2 est approprié pour un prototype, mais trop limité pour capturer des thèmes variés et produire des descriptions riches.
* **Qualité imparfaite des thèmes** : quelques tags erronés provenant de l'étape 1 entraînent des générations incohérentes.

---

## 3. Limite fondamentale : un compléteur spécialisé

Le modèle entraîné est un compléteur spécialisé, pas un modèle de raisonnement.

Il ne sait gérer que les ~25 tags exacts qu'il a vus. Toute variation inconnue entraîne un échec.

**Exemple (échec)** :

```
TAGS: [devops, infra, k9s] DESCRIPTION:
```

Le modèle n'a jamais vu ces mots et génère du texte aléatoire.

**Exemple (réussi)** :

```
TAGS: [Développement Cloud (Kubernetes), Développement Cloud (Infrastructure)] DESCRIPTION:
```

Les tokens sont connus et la génération suit les associations apprises.

---

## 4. Conclusion

Malgré ses limites, cette étape montre qu'un petit modèle peut être conditionné pour générer des descriptions basées sur des tags.

Les résultats fonctionnent, mais restent bridés par :

* le conflit linguistique initial,
* l'absence d'un dataset riche,
* l'utilisation d'un modèle trop léger pour la tâche.

Une version aboutie demanderait un modèle plus robuste et un pipeline d'entraînement mieux alimenté en texte de qualité.
