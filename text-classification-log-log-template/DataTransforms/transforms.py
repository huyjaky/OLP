import re
import random
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def text_normalize(text):
    text = text.lower()  # lowercase
    text = re.sub(r"^RT[\s]+", "", text)  # remove RT in tweet
    text = re.sub(r"https|:\/\/.*[\r\n]*", "", text)  # remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # remove special characters
    text = re.sub(r"<.*?>", "", str(text)) # remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text)) # remove special characters
    text = re.sub(r"\s+", " ", str(text)).strip() # remove extra spaces
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
