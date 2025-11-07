import pandas as pd
import numpy as np
import torch
import logging
import os
import argparse
import csv
import sys
import json # Ajouté pour lire la DB des catégories
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)
from sklearn.model_selection import train_test_split

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(data_file, labels_file, categories_db_file):
    """
    Charge les données, fusionne les labels (clusters) en thèmes (noms uniques)
    et prépare le jeu de données pour le fine-tuning.
    """
    logging.info("--- Phase 1: Chargement et Préparation des Données ---")
    
    # 1. Charger la base de données des catégories (de l'étape 1)
    # C'est notre table de correspondance "cluster_id" -> "theme_name"
    try:
        with open(categories_db_file, 'r', encoding='utf-8') as f:
            categories_db = json.load(f)
        # S'assurer qu'ils sont triés par ID pour correspondre à l'index (0-99)
        categories_db.sort(key=lambda x: x['category_id'])
        # Liste des 100 noms (avec doublons)
        all_theme_names = [cat['category_name'] for cat in categories_db]
        # Liste des ~25 noms (uniques)
        unique_themes_list = sorted(list(set(all_theme_names)))
        # Map pour la fusion : "IA / ML" -> 0, "Web" -> 1...
        theme_to_index_map = {name: i for i, name in enumerate(unique_themes_list)}
        # Map de cluster (100) vers index de thème (25)
        # ex: [0, 1, 0, 2, 1, ...]
        cluster_id_to_theme_index = [theme_to_index_map[name] for name in all_theme_names]
        
        logging.info(f"Chargé {len(all_theme_names)} clusters mappés sur {len(unique_themes_list)} thèmes uniques.")
        
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier de base de données des catégories non trouvé à {categories_db_file}")
        return None
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de {categories_db_file}: {e}")
        return None

    # 2. Charger les labels (Y) - Shape (56641, 100)
    try:
        labels_multi_hot_clusters = np.load(labels_file)
        logging.info(f"Chargé les labels des clusters (shape: {labels_multi_hot_clusters.shape}).")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier de labels non trouvé à {labels_file}")
        return None
        
    # Vérification de cohérence (100 labels vs 100 noms de clusters)
    if labels_multi_hot_clusters.shape[1] != len(all_theme_names):
        logging.error(f"Incohérence: {labels_multi_hot_clusters.shape[1]} labels vs {len(all_theme_names)} noms de clusters. Arrêt.")
        return None
    
    # 3. --- FUSION DES LABELS ---
    # Créer le nouvel array de labels basé sur les thèmes uniques
    num_rows = labels_multi_hot_clusters.shape[0]
    num_unique_themes = len(unique_themes_list)
    
    logging.info(f"Fusion de {labels_multi_hot_clusters.shape[1]} colonnes (clusters) en {num_unique_themes} colonnes (thèmes)...")
    labels_multi_hot_themes = np.zeros((num_rows, num_unique_themes), dtype=int)

    for cluster_id in range(len(all_theme_names)):
        theme_index = cluster_id_to_theme_index[cluster_id]
        
        # Utiliser np.logical_or pour fusionner les colonnes
        # Si la colonne du thème (ex: "Web") est déjà à 1, elle le reste.
        # Si elle est à 0, elle prend la valeur de la colonne du cluster (0 ou 1).
        labels_multi_hot_themes[:, theme_index] = np.logical_or(
            labels_multi_hot_themes[:, theme_index], 
            labels_multi_hot_clusters[:, cluster_id]
        )
        
    logging.info(f"Fusion terminée. Shape des labels thèmes: {labels_multi_hot_themes.shape}")


    # 4. Charger les données textuelles (X)
    try:
        # Nous n'avons besoin que de la description
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
                
        df = pd.read_csv(data_file, usecols=['description'], engine='python', on_bad_lines='warn')
        df = df.dropna(subset=['description'])
        df['description'] = df['description'].str.strip()
        df = df[df['description'].str.len() > 10] 
        logging.info(f"Chargé {len(df)} descriptions depuis {data_file}.")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier de données non trouvé à {data_file}")
        return None
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du CSV: {e}")
        return None

    # 5. Aligner les données (labels et descriptions)
    valid_indices = df.index
    
    # Utiliser le nouvel array de labels fusionnés
    if len(df) != len(labels_multi_hot_themes):
        logging.warning(f"Alignement des labels nécessaire : {len(df)} descriptions valides vs {len(labels_multi_hot_themes)} labels bruts.")
        try:
            labels_multi_hot_themes = labels_multi_hot_themes[valid_indices]
            logging.info(f"Alignement terminé. Nouvelle shape des labels thèmes: {labels_multi_hot_themes.shape}")
            if len(df) != len(labels_multi_hot_themes):
                 logging.error(f"Échec de l'alignement: {len(df)} != {len(labels_multi_hot_themes)}. Stop.")
                 return None
        except IndexError:
             logging.error(f"Erreur d'indexation lors de l'alignement. Le fichier labels_multi_hot.npy est peut-être corrompu.")
             return None
    else:
        logging.info("Descriptions et labels thèmes sont déjà alignés.")


    # 6. Créer le jeu de données formaté (Prompt -> Complétion)
    formatted_texts = []
    for i in range(len(df)):
        description = df.iloc[i]['description']
        
        # Utiliser les labels thèmes (shape 25)
        row_labels_indices = np.where(labels_multi_hot_themes[i] == 1)[0]
        
        if len(row_labels_indices) == 0:
            continue
            
        # Utiliser la liste de thèmes uniques (length 25)
        tags_str = ", ".join([unique_themes_list[idx] for idx in row_labels_indices])
        
        formatted_text = f"TAGS: [{tags_str}] DESCRIPTION: {description}<|endoftext|>"
        formatted_texts.append(formatted_text)

    logging.info(f"Créé {len(formatted_texts)} exemples formatés pour l'entraînement (dépôts avec au moins 1 tag).")
    
    dataset = Dataset.from_dict({"text": formatted_texts})
    return dataset

def main(args):
    """Fonction principale orchestrant le fine-tuning."""
    
    logging.info(f"--- Démarrage de l'Étape 3: Fine-Tuning du Générateur ({args.model_name}) ---")
    logging.info(f"Paramètres: {vars(args)}")

    CHECKPOINT_DIR = os.path.join(args.output_dir, "checkpoints")

    # 1. Préparer les données
    # Appel modifié pour inclure le fichier DB
    dataset = load_and_prepare_data(
        args.data_file, 
        args.labels_file, 
        args.categories_db_file # Nouvel argument
    )
    if dataset is None:
        logging.error("Échec de la préparation des données. Arrêt.")
        return

    # Diviser en train/test
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # 2. Initialiser le Tokenizer et le Modèle
    logging.info(f"Chargement du tokenizer et du modèle: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Tokeniser le jeu de données
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Configurer le Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False 
    )

    # 5. Configurer les Arguments d'Entraînement
    logging.info("Configuration des arguments d'entraînement...")
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), 
        load_best_model_at_end=True,
        logging_steps=100,
    )

    # 6. Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    # 7. Lancer l'entraînement
    logging.info("--- Début du Fine-Tuning ---")
    trainer.train()
    logging.info("--- Fine-Tuning Terminé ---")

    # 8. Sauvegarder le modèle final
    logging.info(f"Sauvegarde du modèle final dans {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 9. Tester le modèle fine-tuné
    logging.info("--- Test de Génération ---")
    
    test_tags = "[IA / Machine Learning, Développement Web (Backend)]"
    prompt = f"TAGS: {test_tags} DESCRIPTION:"
    
    logging.info(f"Prompt: {prompt}")

    device_num = 0 if torch.cuda.is_available() else -1
    generator = pipeline('text-generation', model=args.output_dir, tokenizer=args.output_dir, device=device_num)
    
    outputs = generator(
        prompt,
        max_length=60, 
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        no_repeat_ngram_size=2
    )
    
    generated_text = outputs[0]['generated_text']
    generated_description = generated_text.split("DESCRIPTION:")[1].split("<|endoftext|>")[0].strip()
    
    logging.info(f"Description Générée: {generated_description}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 3: Fine-Tuning d'un modèle de génération de description GitHub.")
    
    # --- Arguments pour les Fichiers ---
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="data/github_data_with_readmes.csv",
        help="Chemin vers le CSV original des dépôts."
    )
    parser.add_argument(
        "--labels_file", 
        type=str, 
        default="outputs/step2_classification/labels_multi_hot.npy",
        help="Chemin vers le fichier .npy des labels multi-hot (sortie step2/generate_labels)."
    )
    parser.add_argument(
        "--categories_db_file", 
        type=str, 
        default="outputs/step1_clustering/github_categories_database.json",
        help="Chemin vers le JSON des prototypes/noms de l'étape 1 (requis pour la fusion)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="distilgpt2",
        help="Nom du modèle de base à fine-tuner (ex: distilgpt2, gpt2)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/step3_generator/distilgpt2_github_generator",
        help="Dossier où sauvegarder le modèle fine-tuné."
    )
    
    # --- Arguments pour l'Entraînement ---
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=2,
        help="Nombre d'époques d'entraînement."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Taille du batch par appareil (train et eval)."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Taux d'apprentissage initial."
    )

    args = parser.parse_args()
    
    # Supprimer l'argument --themes_file s'il est passé par erreur (maintenant inutile)
    if hasattr(args, 'themes_file'):
        delattr(args, 'themes_file')

    main(args)