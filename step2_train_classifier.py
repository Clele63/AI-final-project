# step2_train_classifier.py
import pandas as pd
import numpy as np
import json
import gc
import logging
import argparse
import os
import matplotlib
matplotlib.use('Agg') # Mode non-interactif
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(csv_file, categories_file):
    """Charge le CSV et la base de catégories JSON."""
    logging.info(f"Chargement des données depuis {csv_file}")
    df = pd.read_csv(csv_file)
    df['description'] = df['description'].fillna('')
    df['readme_content'] = df['readme_content'].fillna('')
    df['full_text'] = df['description'] + ' ' + df['readme_content']
    df = df[df['full_text'].str.strip().str.len() > 50].reset_index(drop=True)

    logging.info(f"Chargement des catégories depuis {categories_file}")
    with open(categories_file, 'r', encoding='utf-8') as f:
        categories_db = json.load(f)
    
    id2label = {cat['category_id']: cat['category_name'] for cat in categories_db}
    label2id = {v: k for k, v in id2label.items()}
    n_labels = len(categories_db)
    
    logging.info(f"{len(df)} dépôts et {n_labels} catégories chargés.")
    return df, categories_db, id2label, label2id, n_labels

def assign_labels(df, categories_db, embedding_model_name, device):
    """Assign_labels : Assigne les étiquettes (labels) supervisées en utilisant le clustering de l'étape 1."""
    logging.info("Création des étiquettes pour l'entraînement supervisé...")
    
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    repo_embeddings = embedding_model.encode(
        df['full_text'].tolist(), 
        show_progress_bar=True,
        batch_size=32
    )
    
    category_embeddings = np.array([cat['embedding_prototype'] for cat in categories_db])
    
    logging.info("Assignation de la catégorie la plus proche...")
    similarity_matrix = cosine_similarity(repo_embeddings, category_embeddings)
    df['label'] = np.argmax(similarity_matrix, axis=1)
    
    df.rename(columns={'full_text': 'text'}, inplace=True)
    df_final = df[['text', 'label']]
    
    # Nettoyage mémoire
    del embedding_model, repo_embeddings, category_embeddings, similarity_matrix, df
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    logging.info("Étiquettes assignées et mémoire nettoyée.")
    return df_final

def tokenize_data(df, tokenizer_name, test_size):
    """Convertit en Dataset HF, divise et tokenise."""
    logging.info("Conversion en Dataset Hugging Face et tokenisation...")
    full_dataset = Dataset.from_pandas(df)
    hf_datasets = full_dataset.train_test_split(test_size=test_size, stratify_by_column="label")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = hf_datasets.map(tokenize_function, batched=True)
    logging.info(f"Datasets tokenisés : {hf_datasets}")
    return tokenized_datasets, tokenizer

def compute_metrics(pred):
    """Fonction pour calculer les métriques pendant l'entraînement."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def save_training_plots(log_history, output_file):
    """(Pour le rapport) Sauvegarde les courbes d'entraînement et de validation."""
    logging.info(f"Sauvegarde des courbes d'entraînement dans {output_file}")
    train_loss = [log['loss'] for log in log_history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    epochs_train = [log['epoch'] for log in log_history if 'loss' in log]
    epochs_eval = [log['epoch'] for log in log_history if 'eval_loss' in log]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_train, train_loss, label='Training Loss')
    plt.plot(epochs_eval, eval_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def save_classification_report_and_matrix(trainer, test_dataset, id2label, output_dir):
    """(Pour le rapport) Évalue le modèle final et sauvegarde le rapport et la matrice de confusion."""
    logging.info("Évaluation finale sur le jeu de test...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    label_names = [id2label[i] for i in range(len(id2label))]
    
    # 1. Rapport de classification (texte)
    report = classification_report(labels, preds, target_names=label_names, zero_division=0)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    logging.info(f"Rapport de classification sauvegardé dans {report_path}")
    
    # 2. Matrice de confusion (image)
    # On ne garde que les labels présents pour alléger la matrice
    present_labels_ids = sorted(list(set(labels) | set(preds)))
    present_label_names = [id2label[i] for i in present_labels_ids]
    
    if len(present_labels_ids) > 50:
        logging.warning("Plus de 50 labels, la matrice de confusion sera illisible. Sauvegarde tronquée.")
        # On pourrait tronquer, ou simplement ne pas la générer. Ici, on tente.
        figsize = (50, 40)
    else:
        figsize = (max(15, len(present_labels_ids)//2), max(12, len(present_labels_ids)//2))

    cm = confusion_matrix(labels, preds, labels=present_labels_ids)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=present_label_names, 
                yticklabels=present_label_names)
    plt.title("Matrice de Confusion")
    plt.ylabel('Vrai Label')
    plt.xlabel('Label Prédit')
    plt.tight_layout()
    matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(matrix_path)
    plt.close()
    logging.info(f"Matrice de confusion sauvegardée dans {matrix_path}")

def main(args):
    os.makedirs(args.output_model_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Utilisation du device : {device}")

    # 1. Chargement et préparation des données
    df, categories_db, id2label, label2id, n_labels = load_data(args.input_csv, args.categories_file)
    
    # 2. Assignation des labels (étape coûteuse)
    df_labeled = assign_labels(df, categories_db, args.embedding_model, device)
    
    # 3. Tokenisation
    tokenized_datasets, tokenizer = tokenize_data(df_labeled, args.base_model, args.test_size)
    
    # 4. Chargement du modèle de classification
    logging.info(f"Chargement du modèle de base : {args.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, 
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # 5. Configuration de l'entraînement (optimisé RAM)
    training_args = TrainingArguments(
        output_dir=args.output_model_dir,
        
        # Optimisations Mémoire
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        fp16=True if device == 'cuda' else False,

        # Paramètres classiques
        num_train_epochs=args.epochs,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none" # Désactive WandB/etc.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )
    
    # 6. Lancement de l'entraînement
    logging.info("--- Lancement du fine-tuning optimisé ---")
    trainer.train()
    logging.info("--- Fine-tuning terminé ---")
    
    # 7. Sauvegarde du modèle final
    trainer.save_model(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)
    logging.info(f"Modèle fine-tuné sauvegardé dans '{args.output_model_dir}'")
    
    # 8. Génération des artefacts pour le rapport
    output_parent_dir = os.path.dirname(args.output_model_dir) # outputs/step2_classification/
    
    # Graphiques de perte
    plot_path = os.path.join(output_parent_dir, "training_plots.png")
    save_training_plots(trainer.state.log_history, plot_path)
    
    # Rapport de classification et Matrice de confusion
    save_classification_report_and_matrix(
        trainer, 
        tokenized_datasets['test'], 
        id2label, 
        output_parent_dir
    )
    
    logging.info("--- Étape 2 terminée avec succès ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 2 : Fine-tuning d'un classifieur de dépôts GitHub.")

    # Fichiers
    parser.add_argument('--input_csv', type=str, default="inputs/github_data_with_readmes.csv")
    parser.add_argument('--categories_file', type=str, default="outputs/step1_clustering/github_categories_database.json")
    parser.add_argument('--output_model_dir', type=str, default="outputs/step2_classification/distilroberta_github_classifier")
    
    # Modèles
    parser.add_argument('--base_model', type=str, default='distilroberta-base')
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    
    # Hyperparamètres
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4, help="Per-device batch size")
    parser.add_argument('--grad_accum_steps', type=int, default=8, help="Gradient accumulation steps")

    args = parser.parse_args()
    main(args)