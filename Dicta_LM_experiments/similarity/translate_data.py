# File: translate_and_save.py
from googletrans import Translator
from datasets import load_dataset
import time
import pickle
from tqdm import tqdm

# Step 1: Load the STS dataset (small dev subset for speed)
print("Loading STS dataset...")
sts = load_dataset("glue", "stsb", split="validation")

english_pairs = list(zip(sts["sentence1"], sts["sentence2"], sts["label"]))
english_pairs = [(s1, s2, score / 5.0) for s1, s2, score in english_pairs]  # normalize to 0–1

# Step 2: Translate to Hebrew
translator = Translator()
hebrew_pairs = []
print("Translating to Hebrew...")

for s1, s2, score in tqdm(english_pairs, desc="Translating pairs"):
    try:
        t1 = translator.translate(s1, src='en', dest='he').text
        time.sleep(0.3)  # avoid hitting Google's rate limits
        t2 = translator.translate(s2, src='en', dest='he').text
        time.sleep(0.3)  # avoid hitting Google's rate limits
        hebrew_pairs.append((t1, t2, score))
    except Exception as e:
        print("Translation failed:", e)

# Step 3: Save the translated data
print("Saving translated data...")
with open("hebrew_sts_pairs.pkl", "wb") as f:
    pickle.dump(hebrew_pairs, f)

print("Translation and saving complete.")