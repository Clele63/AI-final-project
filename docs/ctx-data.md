# Contexte, Problématique et Données

## 1. Contexte et Problématique

L’objectif de ce projet est de développer un système capable d’assigner automatiquement des **tags pertinents** à des dépôts GitHub.

La problématique est double :

### Découverte (Non Supervisé)
Comment définir un ensemble pertinent de catégories (tags) ?  
Étiqueter manuellement **50 000 dépôts** est impossible : nous devons **découvrir automatiquement** les thèmes existants.

### Classification (Supervisé)
Comment gérer le fait qu’un dépôt appartient souvent à **plusieurs catégories** ?  
Un projet peut être à la fois *IA*, *Web Dev* et *Python*.  
Un classifieur de type single-label n’est pas adapté.

--> Pour répondre à ces deux défis, nous avons conçu un pipeline en deux étapes :  
1. **Découverte non supervisée** des catégories  
2. **Classification multi-label supervisée**

---

## 2. Les Données et Collecte

La base de l’analyse repose sur le fichier **`github_data_with_readmes.csv`**, généré à partir de notebooks de scraping (`scrapIntoOneRepo.ipynb`).

### 2.1. Source

- **Source initiale** : liste des dépôts issue du projet **GitHubStars** de *vmarkovtsev*, qui maintient un inventaire à jour des dépôts populaires.
- **Filtrage** : plusieurs seuils d’étoiles ont été testés (>50, >100, >200).  
  Le jeu final utilise les dépôts avec **plus de 1000 étoiles** (`repos-min-1000stars.json`), offrant un bon équilibre entre *quantité* et *qualité* (documentation, maturité).

### 2.2. Données Collectées et Prétraitement

Pour chaque dépôt, l’API GitHub a été interrogée afin d’extraire :

- **description** : résumé court, très informatif
- **readme_content** : texte long (détails techniques, exemples, explications)

Les deux champs sont combinés en un **full_text**, permettant d’associer :
- un signal sémantique de haut niveau (description)
- un contexte riche et technique (README)

Après filtrage des dépôts vides ou trop courts, le jeu de données final contient :

**56 641 dépôts** utilisables pour l’entraînement  

► [*Retour*](/README.md)