# step1_create_categories.py
import pandas as pd
import numpy as np
import json
import gc
import logging
import argparse
import os
import matplotlib
import csv
matplotlib.use('Agg') # Mode non-interactif pour SLURM
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(100 * 1024 * 1024)
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name, device):
    """Charge le modèle d'embedding."""
    logging.info(f"Chargement du modèle d'embedding : {model_name} sur {device}")
    return SentenceTransformer(model_name, device=device)

def train_clustering(csv_file, chunk_size, n_categories, embedding_model):
    """
    Phase 1 : Entraînement du MiniBatchKMeans en lisant le CSV par morceaux.
    """
    logging.info("--- Phase 1: Entraînement du modèle de clustering ---")
    kmeans = MiniBatchKMeans(
        n_clusters=n_categories,
        random_state=42,
        batch_size=256,
        n_init='auto'
    )
    
    # csv_reader = pd.read_csv(csv_file, chunksize=chunk_size, iterator=True)
    csv_reader = pd.read_csv(
        csv_file, 
        chunksize=chunk_size, 
        iterator=True, 
        engine='python', 
        on_bad_lines='warn'
    )
    
    for chunk in tqdm(csv_reader, desc="Entraînement du clustering"):
        chunk['description'] = chunk['description'].fillna('')
        chunk['readme_content'] = chunk['readme_content'].fillna('')
        chunk['full_text'] = chunk['description'] + ' ' + chunk['readme_content']
        
        chunk = chunk[chunk['full_text'].str.strip().str.len() > 50]
        if chunk.empty:
            continue

        embeddings = embedding_model.encode(
            chunk['full_text'].tolist(), 
            show_progress_bar=False,
            batch_size=128
        )
        
        kmeans.partial_fit(embeddings)
        del chunk, embeddings
        gc.collect()

    logging.info("✅ Modèle de clustering entraîné.")
    return kmeans

def assign_and_name_categories(csv_file, chunk_size, n_categories, kmeans_model, embedding_model):
    """
    Phase 2 : Assignation des catégories, collecte de textes et nommage par TF-IDF.
    """
    logging.info("--- Phase 2: Assignation et nommage des catégories ---")
    
    texts_per_category = [[] for _ in range(n_categories)]
    repos_per_category_count = [0] * n_categories
    
    # csv_reader_pass2 = pd.read_csv(csv_file, chunksize=chunk_size, iterator=True)
    csv_reader_pass2 = pd.read_csv(
        csv_file, 
        chunksize=chunk_size, 
        iterator=True, 
        engine='python', 
        on_bad_lines='warn'
    )

    for chunk in tqdm(csv_reader_pass2, desc="Assignation & Collecte de texte"):
        chunk['description'] = chunk['description'].fillna('')
        chunk['readme_content'] = chunk['readme_content'].fillna('')
        chunk['full_text'] = chunk['description'] + ' ' + chunk['readme_content']
        chunk = chunk[chunk['full_text'].str.strip().str.len() > 50]
        if chunk.empty:
            continue

        embeddings = embedding_model.encode(chunk['full_text'].tolist(), show_progress_bar=False, batch_size=128)
        labels = kmeans_model.predict(embeddings)
        
        for text, label in zip(chunk['full_text'], labels):
            if repos_per_category_count[label] < 200: # Échantillon pour TF-IDF
                texts_per_category[label].append(text)
            repos_per_category_count[label] += 1
            
        del chunk, embeddings, labels
        gc.collect()

    logging.info("✅ Textes collectés. Nommage des catégories par TF-IDF...")
    
    categories_database = []
    category_keywords = {}
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, max_df=0.8, min_df=2)
    
    for i in tqdm(range(n_categories), desc="Finalisation des catégories"):
        if len(texts_per_category[i]) < 5:
            keywords = ["small_or_generic"]
            category_name = f"Cluster {i} - small_or_generic"
        else:
            tfidf_matrix = vectorizer.fit_transform(texts_per_category[i])
            terms = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            top_indices = mean_tfidf.argsort()[-5:][::-1]
            keywords = [terms[j] for j in top_indices]
            category_name = f"Cluster {i} - {', '.join(keywords)}"
        
        categories_database.append({
            "category_id": i,
            "category_name": category_name,
            "embedding_prototype": kmeans_model.cluster_centers_[i].tolist()
        })
        category_keywords[f"Cluster {i}"] = {
            "name": ", ".join(keywords),
            "repo_count": repos_per_category_count[i]
        }

    return categories_database, category_keywords, repos_per_category_count

def save_cluster_distribution_plot(repo_counts, output_file):
    """
    (Pour le rapport) Sauvegarde un histogramme de la distribution des clusters.
    """
    logging.info(f"Sauvegarde du graphique de distribution des clusters dans {output_file}")
    plt.figure(figsize=(20, 10))
    sns.barplot(x=list(range(len(repo_counts))), y=repo_counts)
    plt.title("Distribution des dépôts par cluster")
    plt.xlabel("ID de Cluster")
    plt.ylabel("Nombre de dépôts")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(args):
    # Création du dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration du device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Utilisation du device : {device}")

    # Modèle d'embedding
    embedding_model = load_model(args.embedding_model, device)
    
    # Phase 1: Entraînement
    kmeans = train_clustering(
        args.input_csv, 
        args.chunk_size, 
        args.n_categories, 
        embedding_model
    )
    
    # Phase 2: Assignation et Nommage
    categories_db, keywords_db, repo_counts = assign_and_name_categories(
        args.input_csv,
        args.chunk_size,
        args.n_categories,
        kmeans,
        embedding_model
    )
    
    # Sauvegarde des artefacts
    db_path = os.path.join(args.output_dir, "github_categories_database.json")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(categories_db, f, ensure_ascii=False, indent=4)
    logging.info(f"Base de {args.n_categories} catégories sauvegardée dans '{db_path}'")
    
    # (Pour le rapport) Mots-clés
    keywords_path = os.path.join(args.output_dir, "cluster_top_keywords.json")
    with open(keywords_path, 'w', encoding='utf-8') as f:
        json.dump(keywords_db, f, ensure_ascii=False, indent=4)
    logging.info(f"Mots-clés des clusters sauvegardés dans '{keywords_path}'")

    # (Pour le rapport) Graphique de distribution
    plot_path = os.path.join(args.output_dir, "cluster_distribution.png")
    save_cluster_distribution_plot(repo_counts, plot_path)

    logging.info("--- Étape 1 terminée avec succès ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 1 : Création de catégories de dépôts GitHub par Clustering.")
    
    # Arguments
    parser.add_argument('--input_csv', type=str, default="data/github_data_with_readmes.csv", help="Chemin vers le CSV d'entrée.")
    parser.add_argument('--output_dir', type=str, default="outputs/step1_clustering", help="Dossier où sauvegarder les résultats.")
    
    # Paramètres du modèle
    parser.add_argument('--n_categories', type=int, default=200, help="Nombre de catégories à créer.")
    parser.add_argument('--chunk_size', type=int, default=2000, help="Taille des morceaux (chunks) pour la lecture CSV.")
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Modèle SentenceTransformer à utiliser.")
    
    args = parser.parse_args()
    main(args)