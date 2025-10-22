
import re
import random
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def text_normalize(text):
    text = re.sub(r"^RT[\s]+", "", text)
    text = re.sub(r"https|:\/\/.*[\r\n]*", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_stopwords(text, stopwords=ENGLISH_STOP_WORDS):
    tokens = text.split()
    tokens = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(tokens)

def MLM(text, mask_prob=0.3):
    tokens = text.split()
    if not tokens:
        return text

    num_to_mask = max(1, int(len(tokens) * mask_prob))
    mask_indices = random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))

    for idx in mask_indices:
        tokens[idx] = "[MASK]"

    return " ".join(tokens)
