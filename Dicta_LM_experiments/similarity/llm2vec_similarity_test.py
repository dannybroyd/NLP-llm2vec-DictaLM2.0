from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizerFast, XLMRobertaModel, AutoTokenizer
import torch
import numpy as np
import pickle
from llm2vec import LLM2Vec
from tqdm import tqdm

# Load the translated dataset
print("Loading translated data")
with open("hebrew_sts_pairs.pkl", "rb") as f:
    hebrew_pairs = pickle.load(f)
assert all(len(item)==3 for item in hebrew_pairs), "Each pair must be in the format(s1, s2, score)"

# Load Models and Tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading AlephBert")
alephbert_tokenizer = BertTokenizerFast.from_pretrained("onlplab/alephbert-base")
alephbert_model = BertModel.from_pretrained("onlplab/alephbert-base")
alephbert_model.to(device).eval()

print("Loading AlephBertGimel")
alephbertgimel_tokenizer = AutoTokenizer.from_pretrained("imvladikon/alephbertgimmel-base-512")
alephbertgimel_model = BertModel.from_pretrained("imvladikon/alephbertgimmel-base-512")
alephbertgimel_model.to(device).eval()

print("Loading XLM-RoBERTa")
xlm_roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
xlm_roberta_model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
xlm_roberta_model.to(device).eval()

print("Loading LLM2Vec‑DictaLM2.0")
l2v = LLM2Vec.from_pretrained(
    "/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/output/mntp/dictalm2.0-instruct/checkpoint-1000",
    peft_model_name_or_path="/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/output/mntp_simcse/dictalm2.0-instruct/checkpoint-1000",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
).to(device).eval()

# Get the tokenaizer for our moodel from checkpoint
l2v_tokenizer = AutoTokenizer.from_pretrained("/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/output/mntp/dictalm2.0-instruct/checkpoint-1000")


def get_embeddings(texts, tokenizer, model, device):
    """
    Purpose: Given a list of strings (texts), produce one vector per string of size of the last hidden_state.
    Returns: A NumPy array shaped (batch_size, hidden_size).
    """
    # check if the model has ".encode" option (our LLM2Vec has) and get encoding if possible automaticly
    if hasattr(model, "encode"):
        #get embeddings
        embeds = model.encode(texts)  
        if isinstance(embeds, torch.Tensor):
            # if type is torch.Tensor move to GPU before converting to numpy
            return embeds.detach().cpu().numpy()
        return np.asarray(embeds)

    # Otherwise, use Hugging Face standard way
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest"
    ).to(device)

    # doing inference no need for gradient tracking
    with torch.no_grad():
        #outputs returns BaseModelOutput from Hugging Face with the hidden_state data we need
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
        #get 1 vector for the sentence so take the mean
        pooled = hidden.mean(dim=1)         # (batch, dim)
    return pooled.cpu().numpy()


def evaluate_model(model, tokenizer, model_name):
    cosine_scores = []
    ground_truth_scores   = []

    print(f"\nEncoding & computing similarity using {model_name}:")
    for s1, s2, ground_truth in tqdm(hebrew_pairs, desc=model_name):
        emb1, emb2 = get_embeddings([s1, s2], tokenizer, model, device)
        sim = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0, 0]

        cosine_scores.append(sim)
        ground_truth_scores.append(ground_truth)

    # Step 4: Spearman & Pearson correlation
    spearman_corr, _ = spearmanr(cosine_scores, ground_truth_scores)
    pearson_corr, _  = pearsonr (cosine_scores, ground_truth_scores)

    print(f"   Spearman: {spearman_corr:.4f}")
    print(f"   Pearson : {pearson_corr:.4f}")


# Evaluate each model
evaluate_model(alephbert_model,     alephbert_tokenizer,     "AlephBert")
evaluate_model(alephbertgimel_model,alephbertgimel_tokenizer,"AlephBertGimel")
evaluate_model(xlm_roberta_model,   xlm_roberta_tokenizer,   "XLM-RoBERTa")
evaluate_model(l2v,                 l2v_tokenizer,           "DictaLM2Vec")