import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertModel, BertTokenizerFast, XLMRobertaModel
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

RANDOM_STATE = 13
# 1. Load the Dataset
print("Loading Dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": "HebSentiment_train.jsonl",
        "validation": "HebSentiment_val.jsonl",
        "test": "HebSentiment_test.jsonl"
    },
    cache_dir="./huggingface_cache",
    download_mode="force_redownload",
    data_dir="data"
)

# Train test split
texts = dataset["train"]["text"]
labels = dataset["train"]["tag_ids"]
labels = [label.strip() for label in labels]

texts_test = dataset["test"]["text"]
labels_test = dataset["test"]["tag_ids"]
labels_test = [label_test.strip() for label_test in labels_test]

# redownload alephbert since it doesn't work without it
snapshot_download("onlplab/alephbert-base")

# 2. Load Pre-trained Models and Tokenizers
print("Loading AlephBert...")
alephbert_tokenizer = BertTokenizerFast.from_pretrained("onlplab/alephbert-base")
alephbert_model = BertModel.from_pretrained("onlplab/alephbert-base")

print("Loading AlephBertGimel...")
alephbertgimel_tokenizer = AutoTokenizer.from_pretrained("imvladikon/alephbertgimmel-base-512")
alephbertgimel_model = BertModel.from_pretrained("imvladikon/alephbertgimmel-base-512")

print("Loading XLM-RoBERTa...")
xlm_roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
xlm_roberta_model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

alephbert_model.to(device)
alephbertgimel_model.to(device)
xlm_roberta_model.to(device)

# 3. Generate Embeddings
def get_embeddings(texts, tokenizer, model, device, embeddings_path=None):
    if embeddings_path and os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}")
        return np.load(embeddings_path)

    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Creating Embeddings", miniters=10):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    if embeddings_path:
        print(f"Saving embeddings to {embeddings_path}")
        np.save(embeddings_path, embeddings)

    return embeddings

alephbert_embeddings_path = "embeddings/alephbert_embeddings.npy"
alephbert_embeddings = get_embeddings(texts, alephbert_tokenizer, alephbert_model, device, alephbert_embeddings_path)

alephbertgimel_embeddings_path = "embeddings/alephbertgimel_embeddings.npy"
alephbertgimel_embeddings = get_embeddings(texts, alephbertgimel_tokenizer, alephbertgimel_model, device, alephbertgimel_embeddings_path)

xlm_roberta_embeddings_path = "embeddings/xlm_roberta_embeddings.npy"
xlm_roberta_embeddings = get_embeddings(texts, xlm_roberta_tokenizer, xlm_roberta_model, device, xlm_roberta_embeddings_path)

alephbert_embeddings_test_path = "embeddings/alephbert_embeddings_test.npy"
alephbert_embeddings_test = get_embeddings(texts_test, alephbert_tokenizer, alephbert_model, device, alephbert_embeddings_test_path)

alephbertgimel_embeddings_test_path = "embeddings/alephbertgimel_embeddings_test.npy"
alephbertgimel_embeddings_test = get_embeddings(texts_test, alephbertgimel_tokenizer, alephbertgimel_model, device, alephbertgimel_embeddings_test_path)

xlm_roberta_embeddings_test_path = "embeddings/xlm_roberta_embeddings_test.npy"
xlm_roberta_embeddings_test = get_embeddings(texts_test, xlm_roberta_tokenizer, xlm_roberta_model, device, xlm_roberta_embeddings_test_path)


# 4. Prepare Data for Training (Separate Splits)
print("Prepare alephbert for Training (data split)...")
X_train_alephbert, X_test_alephbert, y_train_alephbert, y_test_alephbert = alephbert_embeddings, alephbert_embeddings_test, labels, labels_test

print("Prepare alephbertgimmel for Training (data split)...")
X_train_alephbertgimel, X_test_alephbertgimel, y_train_alephbertgimel, y_test_alephbertgimel = alephbertgimel_embeddings, alephbertgimel_embeddings_test, labels, labels_test

print("Prepare xlm roberta for Training (data split)...")
X_train_xlm_roberta, X_test_xlm_roberta, y_train_xlm_roberta, y_test_xlm_roberta = xlm_roberta_embeddings, xlm_roberta_embeddings_test, labels, labels_test


# 5. Encode String Labels
print("Encode string labels")
label_encoder = LabelEncoder()
y_train_alephbert_encoded = label_encoder.fit_transform(y_train_alephbert)
y_test_alephbert_encoded = label_encoder.transform(y_test_alephbert)
y_train_alephbertgimel_encoded = label_encoder.fit_transform(y_train_alephbertgimel)
y_test_alephbertgimel_encoded = label_encoder.transform(y_test_alephbertgimel)
y_train_xlm_roberta_encoded = label_encoder.fit_transform(y_train_xlm_roberta)
y_test_xlm_roberta_encoded = label_encoder.transform(y_test_xlm_roberta)

# 6. Train and Evaluate Classifiers

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)}

print("Evaluation:")

trained_classifiers_dir = "trained_classifiers"
os.makedirs(trained_classifiers_dir, exist_ok=True) # Create the directory if it does not exist

for name, clf in classifiers.items():
    alephbert_model_path = os.path.join(trained_classifiers_dir, f"alephbert_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.pkl")
    alephbertgimel_model_path = os.path.join(trained_classifiers_dir, f"alephbertgimel_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.pkl")
    xlm_roberta_model_path = os.path.join(trained_classifiers_dir, f"xlm_roberta_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.pkl")
    llm2vec_model_path = os.path.join(trained_classifiers_dir, f"llm2vec_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.pkl")

    if os.path.exists(alephbert_model_path):
        print(f"Loading {name} on AlephBert embeddings...")
        with open(alephbert_model_path, 'rb') as f:
            clf_alephbert = pickle.load(f)
    else:
        print(f"Training {name} on AlephBert embeddings...")
        clf_alephbert = clf
        clf_alephbert.fit(X_train_alephbert, y_train_alephbert_encoded)
        with open(alephbert_model_path, 'wb') as f:
            pickle.dump(clf_alephbert, f)

    y_pred_alephbert = clf_alephbert.predict(X_test_alephbert)
    print("AlephBert Classification Report:")
    print(classification_report(y_test_alephbert_encoded, y_pred_alephbert, target_names=label_encoder.classes_))

    if os.path.exists(alephbertgimel_model_path):
        print(f"Loading {name} on AlephBertGimel embeddings...")
        with open(alephbertgimel_model_path, 'rb') as f:
            clf_alephbertgimel = pickle.load(f)
    else:
        print(f"Training {name} on AlephBertGimel embeddings...")
        clf_alephbertgimel = clf
        clf_alephbertgimel.fit(X_train_alephbertgimel, y_train_alephbertgimel_encoded)
        with open(alephbertgimel_model_path, 'wb') as f:
            pickle.dump(clf_alephbertgimel, f)

    y_pred_alephbertgimel = clf_alephbertgimel.predict(X_test_alephbertgimel)
    print("AlephBertGimel Classification Report:")
    print(classification_report(y_test_alephbertgimel_encoded, y_pred_alephbertgimel, target_names=label_encoder.classes_))

    if os.path.exists(xlm_roberta_model_path):
        print(f"Loading {name} on XLM-RoBERTa embeddings...")
        with open(xlm_roberta_model_path, 'rb') as f:
            clf_xlm_roberta = pickle.load(f)
    else:
        print(f"Training {name} on XLM-RoBERTa embeddings...")
        clf_xlm_roberta = clf
        clf_xlm_roberta.fit(X_train_xlm_roberta, y_train_xlm_roberta_encoded)
        with open(xlm_roberta_model_path, 'wb') as f:
            pickle.dump(clf_xlm_roberta, f)

    y_pred_xlm_roberta = clf_xlm_roberta.predict(X_test_xlm_roberta)
    print("XLM-RoBERTa Classification Report:")
    print(classification_report(y_test_xlm_roberta_encoded, y_pred_xlm_roberta, target_names=label_encoder.classes_))