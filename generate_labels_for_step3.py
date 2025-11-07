import pandas as pd
import numpy as np
import json
import logging
import os
import gc
import csv
import sys
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_categories_prototypes(categories_db_file):
    """Charge les prototypes d'embedding depuis le fichier JSON de l'étape 1."""
    try:
        with open(categories_db_file, 'r', encoding='utf-8') as f:
            categories_db = json.load(f)
        
        categories_db.sort(key=lambda x: x['category_id'])
        
        prototypes = [cat['embedding_prototype'] for cat in categories_db]
        theme_names = [cat['category_name'] for cat in categories_db]
        
        logging.info(f"Chargé {len(prototypes)} prototypes de catégories.")
        return np.array(prototypes), theme_names
    except FileNotFoundError:
        logging.error(f"Fichier de base de données des catégories non trouvé: {categories_db_file}")
        return None, None
    except Exception as e:
        logging.error(f"Erreur lors du chargement des prototypes: {e}")
        return None, None

def main(args):
    """Fonction principale orchestrant la génération des labels."""
    logging.info("--- Démarrage de la génération des labels pour l'Étape 3 ---")
    logging.info(f"Paramètres: {vars(args)}")

    # --- CORRECTIF POUR 'field larger than field limit' ---
    logging.info("Augmentation de la limite de champ CSV...")
    max_int = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            max_int = int(max_int / 10)
            decrement = True
    
    logging.info(f"Limite de champ CSV augmentée à {max_int}")
    # --- FIN DU CORRECTIF ---
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Dossier de sortie assuré d'exister: {output_dir}")

    # 1. Charger les prototypes de l'Étape 1
    prototypes, theme_names = load_categories_prototypes(args.categories_db_file)
    if prototypes is None:
        return

    # 2. Charger le modèle d'embedding
    logging.info(f"Chargement du modèle d'embedding: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)

    # 3. Traiter le CSV par chunks pour calculer tous les embeddings
    all_embeddings = []
    total_rows = 0
    
    logging.info(f"Lecture du CSV ({args.data_file}) par chunks de {args.chunk_size}...")
    try:
        csv_reader = pd.read_csv(
            args.data_file, 
            chunksize=args.chunk_size, 
            iterator=True,
            engine='python',
            on_bad_lines='warn'
        )

        for chunk in tqdm(csv_reader, desc="Calcul des Embeddings"):
            chunk['description'] = chunk['description'].fillna('')
            chunk['readme_content'] = chunk['readme_content'].fillna('')
            chunk['full_text'] = chunk['description'] + ' ' + chunk['readme_content']
            
            # --- SUPPRESSION DU FILTRAGE ---
            # chunk = chunk[chunk['full_text'].str.strip().str.len() > 50] # <-- LIGNE SUPPRIMÉE
            # Nous encodons TOUTES les lignes, même vides. L'embedding d'un
            # texte vide n'aura de similarité avec aucun prototype,
            # ce qui donnera [0, 0, ..., 0] (correct).
            # --- FIN DE LA CORRECTION ---
            
            if chunk.empty:
                continue

            embeddings = embedding_model.encode(
                chunk['full_text'].tolist(), 
                show_progress_bar=False, 
                batch_size=128
            )
            all_embeddings.append(embeddings)
            
            total_rows += len(chunk)
            
            del chunk, embeddings
            gc.collect()

    except FileNotFoundError:
        logging.error(f"Fichier de données non trouvé: {args.data_file}")
        return
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du CSV: {e}")
        return

    if not all_embeddings:
        logging.error("Aucun embedding n'a été généré. Vérifiez le fichier CSV.")
        return

    logging.info(f"Combiné {len(all_embeddings)} chunks en un seul array numpy.")
    all_embeddings_np = np.vstack(all_embeddings)
    logging.info(f"Shape finale des embeddings: {all_embeddings_np.shape} (doit correspondre au CSV)")

    # 4. Calculer la similarité cosinus
    logging.info("Calcul de la matrice de similarité cosinus...")
    similarity_matrix = cosine_similarity(all_embeddings_np, prototypes)
    logging.info(f"Shape de la matrice de similarité: {similarity_matrix.shape}")

    # 5. Appliquer le seuil pour obtenir les labels multi-hot
    logging.info(f"Application du seuil de similarité ({args.similarity_threshold})...")
    labels_multi_hot = (similarity_matrix > args.similarity_threshold).astype(int)

    # 6. Sauvegarder l'array numpy
    try:
        logging.info(f"Sauvegarde des labels multi-hot (pour étape 3) dans {args.output_file}")
        np.save(args.output_file, labels_multi_hot)
        logging.info("✅ Sauvegarde terminée!")
    except Exception as e:
        logging.error(f"Échec de la sauvegarde de '{args.output_file}': {e}")

    logging.info("--- Script de génération de labels terminé ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère le fichier de labels 'labels_multi_hot.npy' requis pour step3.")
    
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="data/github_data_with_readmes.csv",
        help="Chemin vers le CSV original des dépôts."
    )
    parser.add_argument(
        "--categories_db_file", 
        type=str, 
        default="outputs/step1_clustering/github_categories_database.json",
        help="Chemin vers le JSON des prototypes de l'étape 1."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="outputs/step2_classification/labels_multi_hot.npy",
        help="Chemin du fichier .npy de sortie pour les labels."
    )
    parser.add_argument(
        "--embedding_model", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Nom du modèle d'embedding SentenceTransformer."
    )
    parser.add_argument(
        "--similarity_threshold", 
        type=float, 
        default=0.6,
        help="Seuil de similarité cosinus pour assigner un label."
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000,
        help="Taille des chunks pour la lecture du CSV."
    )
    
    args = parser.parse_args()
    main(args)