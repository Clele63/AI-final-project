# step2_train_classifier.py
import pandas as pd
import numpy as np
import json
import gc
import logging
import argparse
import os
import matplotlib
import csv
matplotlib.use('Agg') # Mode non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # <--- CORRECTION : IMPORT AJOUTÉ
import torch
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

csv.field_size_limit(100 * 1024 * 1024)
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FONCTION MODIFIÉE (pour gérer la fusion de noms) ---
def load_data(csv_file, categories_file):
    """Charge le CSV et la base de catégories JSON."""
    logging.info(f"Chargement des données depuis {csv_file}")
    df = pd.read_csv(csv_file, engine='python', on_bad_lines='warn')
    df['description'] = df['description'].fillna('')
    df['readme_content'] = df['readme_content'].fillna('')
    df['full_text'] = df['description'] + ' ' + df['readme_content']
    df = df[df['full_text'].str.strip().str.len() > 50].reset_index(drop=True)

    logging.info(f"Chargement des {args.n_categories} clusters depuis {categories_file}")
    with open(categories_file, 'r', encoding='utf-8') as f:
        categories_db = json.load(f) # Base de N clusters (ex: 70)
    
    # --- LOGIQUE DE FUSION ---
    all_cluster_names = [cat['category_name'] for cat in categories_db]
    unique_themes = sorted(list(set(all_cluster_names)))
    
    n_labels = len(unique_themes) # Le nombre de thèmes uniques (ex: ~20-30)
    id2label = {i: theme for i, theme in enumerate(unique_themes)}
    label2id = {theme: i for i, theme in enumerate(unique_themes)}
    
    cluster_name_to_theme_id = {name: label2id[name] for name in all_cluster_names}

    logging.info(f"{len(df)} dépôts chargés.")
    logging.info(f"{len(categories_db)} clusters ont été lus et fusionnés en {n_labels} thèmes uniques.")
    
    # --- AJOUT POUR LE RAPPORT ---
    themes_list_path = os.path.join(args.output_dir, "themes_list.txt")
    with open(themes_list_path, 'w', encoding='utf-8') as f:
        f.write("Liste des thèmes uniques générés :\n")
        f.write("="*30 + "\n")
        for theme in unique_themes:
            f.write(f"- {theme}\n")
    logging.info(f"Liste des thèmes uniques sauvegardée dans '{themes_list_path}'")
    # --- FIN AJOUT ---
    
    return df, categories_db, id2label, label2id, n_labels, cluster_name_to_theme_id
# --- FIN MODIFICATION ---

# --- FONCTION MODIFIÉE (pour mapper les clusters aux thèmes) ---
def assign_labels(df, categories_db, embedding_model_name, device, similarity_threshold, cluster_name_to_theme_id, n_themes):
    """Assign_labels : Assigne les étiquettes (multi-label) en mappant les clusters aux thèmes."""
    logging.info("Création des étiquettes pour l'entraînement supervisé (Multi-Label)...")
    
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    repo_embeddings = embedding_model.encode(
        df['full_text'].tolist(), 
        show_progress_bar=True,
        batch_size=32
    )
    
    # 1. Obtenir les 70 prototypes de clusters
    category_embeddings = np.array([cat['embedding_prototype'] for cat in categories_db])
    
    logging.info(f"Assignation des étiquettes (multi-label) avec un seuil de similarité de {similarity_threshold}")
    # 2. Calculer la similarité avec les 70 clusters
    similarity_matrix = cosine_similarity(repo_embeddings, category_embeddings)
    
    # 3. Trouver quels clusters (parmi 70) dépassent le seuil
    labels_bool_clusters = similarity_matrix > similarity_threshold
    
    # --- Fallback pour les repos sans label ---
    no_label_mask = ~labels_bool_clusters.any(axis=1)
    if np.any(no_label_mask):
        logging.warning(f"{np.sum(no_label_mask)} dépôts n'ont atteint le seuil pour aucun cluster. Assignation au plus proche.")
        best_cluster_indices = np.argmax(similarity_matrix[no_label_mask], axis=1)
        for i, best_cluster_idx in zip(np.where(no_label_mask)[0], best_cluster_indices):
            labels_bool_clusters[i, best_cluster_idx] = True
            
    # 4. Conversion des 70 clusters en ~30 thèmes
    logging.info("Fusion des labels de clusters en labels de thèmes uniques...")
    
    labels_multi_hot_themes = np.zeros((len(df), n_themes), dtype=float)
    cluster_names = [cat['category_name'] for cat in categories_db]
    
    for repo_idx, cluster_matches in enumerate(tqdm(labels_bool_clusters, desc="Fusion des labels")):
        theme_ids_for_repo = set()
        for cluster_idx, did_match in enumerate(cluster_matches):
            if did_match:
                cluster_name = cluster_names[cluster_idx]
                theme_id = cluster_name_to_theme_id[cluster_name]
                theme_ids_for_repo.add(theme_id)
        
        for theme_id in theme_ids_for_repo:
            labels_multi_hot_themes[repo_idx, theme_id] = 1.0

    df['label'] = list(labels_multi_hot_themes)
    
    df.rename(columns={'full_text': 'text'}, inplace=True)
    df_final = df[['text', 'label']]
    
    # Nettoyage mémoire
    del embedding_model, repo_embeddings, category_embeddings, similarity_matrix, df, labels_bool_clusters, labels_multi_hot_themes
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    logging.info("Étiquettes (multi-label) fusionnées et assignées.")
    return df_final
# --- FIN MODIFICATION ---

def tokenize_data(df, tokenizer_name, test_size_arg, n_labels):
    """Convertit en Dataset HF, divise et tokenise (pour Multi-Label)."""
    logging.info("Conversion en Dataset Hugging Face et tokenisation (Multi-Label)...")
    
    df['label'] = df['label'].apply(lambda x: list(x))
    full_dataset = Dataset.from_pandas(df)

    new_features = Features({
        'text': full_dataset.features['text'],
        'label': Sequence(feature=Value(dtype='float32'), length=n_labels)
    })
    full_dataset = full_dataset.cast(new_features)

    logging.warning("La stratification par colonne est désactivée pour le multi-label.")
    hf_datasets = full_dataset.train_test_split(test_size=test_size_arg)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = hf_datasets.map(tokenize_function, batched=True)
    logging.info(f"Datasets tokenisés : {hf_datasets}")
    return tokenized_datasets, tokenizer

def compute_metrics(pred):
    """Fonction pour calculer les métriques (Multi-Label)."""
    labels = pred.label_ids
    logits = pred.predictions
    
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    subset_accuracy = accuracy_score(labels, preds)
    
    return {
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'subset_accuracy': subset_accuracy
    }

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

# --- MODIFICATION : Renommage de la fonction ---
def save_classification_outputs(trainer, test_dataset, id2label, output_dir):
    """(Pour le rapport) Évalue le modèle final et sauvegarde le rapport et les graphiques."""
    logging.info("Évaluation finale sur le jeu de test (Multi-Label)...")
    predictions = trainer.predict(test_dataset)
    
    logits = predictions.predictions
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = predictions.label_ids
    
    all_label_ids = list(id2label.keys()) 
    label_names = [id2label[i] for i in all_label_ids]

    # 1. Sauvegarder le rapport texte (inchangé)
    report_text = classification_report(
        labels, 
        preds, 
        labels=all_label_ids,
        target_names=label_names, 
        zero_division=0
    )
    report_path = os.path.join(output_dir, "classification_report_multilabel.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logging.info(f"Rapport de classification (multi-label) sauvegardé dans {report_path}")
    
    # 2. Sauvegarder le rapport en dictionnaire (pour les graphiques)
    report_dict = classification_report(
        labels, 
        preds, 
        labels=all_label_ids,
        target_names=label_names, 
        zero_division=0,
        output_dict=True
    )
    
    # --- AJOUT : SAUVEGARDE DES GRAPHIQUES POUR LE RAPPORT ---
    save_report_graphics(report_dict, output_dir)
    # --- FIN AJOUT ---

# --- NOUVELLE FONCTION POUR LES GRAPHIQUES DU RAPPORT ---
def save_report_graphics(report_dict, output_dir):
    """Sauvegarde des graphiques (F1, Support) basés sur le rapport de classification."""
    logging.info("Génération des graphiques pour le rapport...")
    
    # 1. Extraire les données dans un DataFrame
    data = []
    for label, metrics in report_dict.items():
        if label not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            data.append({
                "theme": label,
                "f1-score": metrics['f1-score'],
                "support": metrics['support']
            })
    
    df = pd.DataFrame(data).sort_values('f1-score', ascending=False)
    
    # 2. Graphique des F1-Scores
    # Augmentation de la taille de la figure basée sur le nombre de labels
    plt.figure(figsize=(15, max(10, len(df) * 0.5)))
    sns.barplot(x="f1-score", y="theme", data=df, palette="viridis")
    plt.title("Performance (F1-Score) par Thème")
    plt.xlabel("F1-Score")
    plt.ylabel("Thème")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_barplot.png"))
    plt.close()

    # 3. Graphique de Distribution du Support
    df_support = df.sort_values('support', ascending=False)
    plt.figure(figsize=(15, max(10, len(df) * 0.5))) # Taille dynamique
    sns.barplot(x="support", y="theme", data=df_support, palette="plasma")
    plt.title("Distribution des Échantillons (Support) par Thème")
    plt.xlabel("Nombre d'échantillons (Support)")
    plt.ylabel("Thème")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "support_distribution_barplot.png"))
    plt.close()
    
    logging.info(f"Graphiques de rapport sauvegardés dans '{output_dir}'")
# --- FIN NOUVELLE FONCTION ---

# --- FONCTION MAIN MODIFIÉE ---
def main(args):
    # --- MODIFICATION ---
    # Le dossier de sortie est maintenant partagé
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    # --- FIN MODIFICATION ---
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Utilisation du device : {device}")

    # 1. Chargement et préparation des données (avec fusion)
    df, categories_db, id2label, label2id, n_labels, cluster_name_to_theme_id = load_data(args.input_csv, args.categories_file)
    
    # 2. Assignation des labels (avec fusion)
    df_labeled = assign_labels(
        df, 
        categories_db, 
        args.embedding_model, 
        device, 
        args.similarity_threshold,
        cluster_name_to_theme_id,
        n_labels # n_labels est maintenant le nombre de thèmes uniques
    )
    
    # 3. Tokenisation
    tokenized_datasets, tokenizer = tokenize_data(df_labeled, args.base_model, args.test_size, n_labels)
    
    # 4. Chargement du modèle de classification
    logging.info(f"Chargement du modèle de base : {args.base_model} (pour {n_labels} thèmes uniques)")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, 
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )
    
    # 5. Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=model_dir, # Sauvegarde dans le sous-dossier
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        fp16=True if device == 'cuda' else False,
        num_train_epochs=args.epochs,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )
    
    # 6. Lancement de l'entraînement
    logging.info("--- Lancement du fine-tuning optimisé (Multi-Label) ---")
    trainer.train()
    logging.info("--- Fine-tuning terminé ---")
    
    # 7. Sauvegarde du modèle final
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    logging.info(f"Modèle fine-tuné sauvegardé dans '{model_dir}'")
    
    # 8. Génération des artefacts pour le rapport
    plot_path = os.path.join(args.output_dir, "training_plots.png")
    save_training_plots(trainer.state.log_history, plot_path)
    
    # --- MODIFICATION ---
    # Appel de la fonction de sauvegarde (qui inclut les graphiques)
    save_classification_outputs(
        trainer, 
        tokenized_datasets['test'], 
        id2label, 
        args.output_dir # Sauvegarde dans le dossier parent
    )
    # --- FIN MODIFICATION ---
    
    logging.info("--- Étape 2 terminée avec succès ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Étape 2 : Fine-tuning d'un classifieur (Multi-Label) de dépôts GitHub.")

    # Fichiers
    parser.add_argument('--input_csv', type=str, default="data/github_data_with_readmes.csv")
    parser.add_argument('--categories_file', type=str, default="outputs/step1_clustering/github_categories_database.json")
    
    # --- MODIFICATION ---
    # Séparation du dossier de sortie global et du nom du modèle
    parser.add_argument('--output_dir', type=str, default="outputs/step2_classification")
    parser.add_argument('--model_name', type=str, default="distilroberta_github_classifier")
    # --- FIN MODIFICATION ---
    
    # Modèles
    parser.add_argument('--base_model', type=str, default='distilroberta-base')
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    
    # Hyperparamètres
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=3)
    # --- CORRECTION FAUTE DE FRAPPE ---
    parser.add_argument('--batch_size', type=int, default=4, help="Per-device batch size")
    # --- FIN CORRECTION ---
    parser.add_argument('--grad_accum_steps', type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument('--similarity_threshold', type=float, default=0.6, help="Seuil de similarité cosinus pour l'assignation multi-label.")
    
    # --- AJOUT IMPORTANT ---
    # Cet argument doit correspondre au n_categories de l'étape 1
    parser.add_argument('--n_categories', type=int, default=70, help="Nombre de clusters générés par l'étape 1 (ex: 70).")
    # --- FIN AJOUT ---

    args = parser.parse_args()
    main(args)
