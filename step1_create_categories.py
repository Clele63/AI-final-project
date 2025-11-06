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

# --- IMPORTS pour PHI-3 ---
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
# (Pas de BitsAndBytesConfig, nous chargeons en 16-bit)
# --- FIN IMPORTS ---

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

# --- FONCTION DE NOMMAGE (PHI-3, PROMPT AMÉLIORÉ, NETTOYAGE) ---
def assign_and_name_categories(csv_file, chunk_size, n_categories, kmeans_model, embedding_model):
    """
    Phase 2 : Assignation et nommage des catégories (via LLM Local Phi-3).
    """
    logging.info("--- Phase 2: Assignation et nommage des catégories ---")
    
    # Étape 2.1 : Collecte des textes (inchangée)
    texts_per_category = [[] for _ in range(n_categories)]
    repos_per_category_count = [0] * n_categories
    
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
            if repos_per_category_count[label] < 200: 
                texts_per_category[label].append(text)
            repos_per_category_count[label] += 1
            
        del chunk, embeddings, labels
        gc.collect()

    logging.info("✅ Textes collectés. Nommage par LLM local (Phi-3)...")

    # Étape 2.2 : Nommage automatique (logique LLM Local)
    
    model_id = "microsoft/Phi-3-mini-4k-instruct" 
    logging.info(f"Chargement du LLM local : {model_id} (en bfloat16, chargement manuel)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Chargement en 16 bits
        # trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.to("cuda")
    logging.info("Modèle déplacé avec succès sur 'cuda'")

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda"
    )
    
    # --- PROMPT AMÉLIORÉ ---
    prompt_template = """<|user|>
Vous êtes un expert en développement logiciel qui catégorise des dépôts GitHub.
Votre tâche est d'analyser une liste de mots-clés et de générer une catégorie technique unique et spécifique.

RÈGLES IMPORTANTES :
1.  Ignorez les mots-clés génériques et bruyants comme: 'http', 'org', 'io', 'md', 'png', 'www', 'br', 'img', 'github', 'file', 'use', 'code', 'license', 'build', 'install', 'docs', 'run', 'data', 'api', 'server'.
2.  Répondez *uniquement* avec le nom de la catégorie (ex: "Développement Web (Frontend)") et rien d'autre.

EXEMPLES :
- Mots-clés: "model, py, pytorch, tensorflow, training, neural"
  Catégorie: IA / Machine Learning
- Mots-clés: "js, react, npm, webpack, css, html, vue"
  Catégorie: Développement Web (Frontend)
- Mots-clés: "docker, kubernetes, helm, aws, compose, container, terraform"
  Catégorie: DevOps & Infrastructure Cloud
- Mots-clés: "swift, ios, cocoapods, xcode, app, objective-c"
  Catégorie: Développement Mobile (iOS)
- Mots-clés: "rust, cargo, dll, process, windows, systems"
  Catégorie: Programmation Système (Rust)
- Mots-clés: "django, flask, api, server, py, backend, node, spring"
  Catégorie: Développement Web (Backend)
- Mots-clés: "android, app, kotlin, java, gradle, mobile"
  Catégorie: Développement Mobile (Android)

MAINTENANT, CATÉGORISEZ CECI :
Mots-clés: "{keywords}"
<|end|>
<|assistant|>
"""

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, max_df=0.8, min_df=2) 
    categories_database = []
    category_keywords = {}

    for i in tqdm(range(n_categories), desc="Nommage par LLM local"):
        if len(texts_per_category[i]) < 5:
            keywords_str = "small_or_generic"
            category_name = "Générique / Autre" # Nom de fusion
            auto_name = "Générique / Autre"
        else:
            # 6. Obtenir les mots-clés
            tfidf_matrix = vectorizer.fit_transform(texts_per_category[i])
            terms = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            top_indices = mean_tfidf.argsort()[-15:][::-1] # 15 keywords
            keywords_str = ", ".join([terms[j] for j in top_indices])
            
            prompt = prompt_template.format(keywords=keywords_str)
            
            # 7. Générer le nom
            try:
                outputs = text_generator(
                    prompt, 
                    max_new_tokens=20,  # <-- Augmenté pour éviter la troncature
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
                
                # 8. Parser la sortie
                generated_text = outputs[0]['generated_text']
                assistant_response = generated_text.split("<|assistant|>")[-1].strip()
                # category_name = assistant_response.split('\n')[0].strip().replace('"', '')
                first_line = assistant_response.split('\n')[0].strip().replace('"', '')
                if ":" in first_line:
                    category_name = first_line.split(":")[-1].strip()
                else:
                    category_name = first_line

                if category_name == "" or "désolé" in category_name.lower():
                    logging.warning(...)
                    category_name = "Générique / Autre"
                # # --- AUTO-NETTOYAGE ---
                # if category_name == "" or "désolé" in category_name.lower() or "cluster" in category_name.lower():
                #     logging.warning(f"Échec du nommage pour Cluster {i}. Réponse LLM: '{category_name}'. Rejeté comme 'Générique'.")
                #     category_name = "Générique / Autre"
                # # --- FIN AUTO-NETTOYAGE ---

            except Exception as e:
                logging.warning(f"Erreur parsing LLM pour Cluster {i}: {e}. Fallback sur 'Générique'.")
                category_name = "Générique / Autre"
            
            auto_name = category_name
            logging.info(f"Cluster {i} -> Mots-clés: '{keywords_str}' -> Nommé: '{category_name}'")

        
        categories_database.append({
            "category_id": i,
            "category_name": category_name,
            "embedding_prototype": kmeans_model.cluster_centers_[i].tolist()
        })
        category_keywords[f"Cluster {i}"] = {
            "name": keywords_str, 
            "auto_name": auto_name,
            "repo_count": repos_per_category_count[i]
        }

    return categories_database, category_keywords, repos_per_category_count
# --- FIN DE LA FONCTION ---


def save_cluster_distribution_plot(repo_counts, output_file):
    """(Pour le rapport) Sauvegarde un histogramme de la distribution des clusters."""
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
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Utilisation du device : {device}")

    embedding_model = load_model(args.embedding_model, device)
    
    kmeans = train_clustering(
        args.input_csv, 
        args.chunk_size, 
        args.n_categories, 
        embedding_model
    )
    
    categories_db, keywords_db, repo_counts = assign_and_name_categories(
        args.input_csv,
        args.chunk_size,
        args.n_categories,
        kmeans,
        embedding_model
    )
    
    db_path = os.path.join(args.output_dir, "github_categories_database.json")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(categories_db, f, ensure_ascii=False, indent=4)
    logging.info(f"Base de {args.n_categories} catégories sauvegardée dans '{db_path}'")
    
    keywords_path = os.path.join(args.output_dir, "cluster_top_keywords.json")
    with open(keywords_path, 'w', encoding='utf-8') as f:
        json.dump(keywords_db, f, ensure_ascii=False, indent=4)
    logging.info(f"Mots-clés des clusters sauvegardés dans '{keywords_path}'")

    plot_path = os.path.join(args.output_dir, "cluster_distribution.png")
    save_cluster_distribution_plot(repo_counts, plot_path)

    logging.info("--- Étape 1 terminée avec succès (nommage automatique par LLM local) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 1 : Création de catégories de dépôts GitHub par Clustering.")
    
    parser.add_argument('--input_csv', type=str, default="data/github_data_with_readmes.csv", help="Chemin vers le CSV d'entrée.")
    parser.add_argument('--output_dir', type=str, default="outputs/step1_clustering", help="Dossier où sauvegarder les résultats.")
    
    # --- MODIFICATION IMPORTANTE ---
    parser.add_argument('--n_categories', type=int, default=70, help="Nombre de catégories à créer (ex: 70).")
    # --- FIN MODIFICATION ---
    parser.add_argument('--chunk_size', type=int, default=2000, help="Taille des morceaux (chunks) pour la lecture CSV.")
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Modèle SentenceTransformer à utiliser.")
    
    args = parser.parse_args()
    main(args)