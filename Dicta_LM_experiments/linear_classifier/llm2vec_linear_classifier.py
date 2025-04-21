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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from llm2vec import LLM2Vec
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

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

#define the llm2vec model
l2v = LLM2Vec.from_pretrained(
    "/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/output/mntp/dictalm2.0-instruct/checkpoint-1000",
    peft_model_name_or_path="/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/output/mntp_simcse/dictalm2.0-instruct/checkpoint-1000",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)


# 3.1. Generate llm2vec Embeddings
def load_or_create_embeddings(texts, embeddings_path):
    if embeddings_path and os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}")
        return np.load(embeddings_path)
    else:
        llm2vec_embeddings = np.array(l2v.encode(texts))
        print(f"Saving embeddings to {embeddings_path}")
        np.save(embeddings_path, llm2vec_embeddings)
        return llm2vec_embeddings

#create the embeddings
print("Generating llm2vec embeddings...")
llm2vec_embeddings_path = "embeddings/llm2vec_embeddings_train.npy"
llm2vec_embeddings_test_path = "embeddings/llm2vec_embeddings_test.npy"
llm2vec_embeddings = load_or_create_embeddings(texts, llm2vec_embeddings_path)
llm2vec_embeddings_test = load_or_create_embeddings(texts_test, llm2vec_embeddings_test_path)

print("embeddings dim: ", llm2vec_embeddings.shape)

print("Prepare llm2vec for Training (data split)...")
X_train_llm2vec, X_test_llm2vec, y_train_llm2vec, y_test_llm2vec = llm2vec_embeddings, llm2vec_embeddings_test, labels, labels_test

# 5. Encode String Labels
print("Encode string labels")
label_encoder = LabelEncoder()
y_train_llm2vec_encoded = label_encoder.fit_transform(y_train_llm2vec)
y_test_llm2vec_encoded = label_encoder.transform(y_test_llm2vec)

# 6. Train and Evaluate Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)}

print("Evaluation:")

trained_classifiers_dir = "trained_classifiers"
os.makedirs(trained_classifiers_dir, exist_ok=True) # Create the directory if it does not exist

for name, clf in classifiers.items():
    llm2vec_model_path = os.path.join(trained_classifiers_dir, f"llm2vec_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.pkl")

    if os.path.exists(llm2vec_model_path):
        print(f"Loading {name} on llm2vec embeddings...")
        with open(llm2vec_model_path, 'rb') as f:
            clf_llm2vec = pickle.load(f)
    else:
        print(f"Training {name} on llm2vec embeddings...")
        clf_llm2vec = clf
        clf_llm2vec.fit(X_train_llm2vec, y_train_llm2vec_encoded)
        with open(llm2vec_model_path, 'wb') as f:
            pickle.dump(clf_llm2vec, f)

    y_pred_llm2vec = clf_llm2vec.predict(X_test_llm2vec)
    print("llm2vec Classification Report:")
    print(classification_report(y_test_llm2vec_encoded, y_pred_llm2vec, target_names=label_encoder.classes_))