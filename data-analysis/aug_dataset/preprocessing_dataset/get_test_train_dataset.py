import pandas as pd
from spacy.lang.vi import Vietnamese
import string
from sklearn.model_selection import train_test_split
# import env variables

from dotenv import load_dotenv
import os 
load_dotenv('../.env')

# from DataTransforms.test import create_tokenizer
import re

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

for enc in encodings:
    try:
        df = pd.read_csv(os.getenv('RAW_TRAIN_DATASET_PATH'), encoding=enc).dropna()
        # df = pd.read_csv("./Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

from spacy.lang.vi import Vietnamese

nlp = Vietnamese()


def text_normalize(
    text,
    remove_urls=True,
    remove_emails=True,
    remove_mentions=True,
    remove_hashtags=True,
    remove_numbers=False,
):
    """
    Normalize and clean Vietnamese text with additional regex steps:
    - remove/normalize URLs, emails, mentions, hashtags, phone numbers, currency symbols
    - remove non-printable characters, reduce repeated punctuation and elongated letters
    - optional removal of digits
    Returns cleaned string (tokens joined by space) after spaCy tokenization and stopword filtering.
    """
    try:
        text = str(text)
        # basic normalization
        text = text.strip().lower()

        # remove urls
        if remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # remove emails
        if remove_emails:
            text = re.sub(r"\S+@\S+", " ", text)
        # remove html tags
        text = re.sub(r"<.*?>", " ", text)
        # mentions (@username)
        if remove_mentions:
            text = re.sub(r"@\w+", " ", text)
        # hashtags: keep the word but drop the #
        if remove_hashtags:
            text = re.sub(r"#(\w+)", r"\1", text)
        # remove phone numbers (simple patterns)
        text = re.sub(r"\b0\d{8,}\b|\b\d{9,}\b", " ", text)
        # replace common currency symbols with space
        text = re.sub(r"[\$€£¥₫]", " ", text)

        # remove punctuation (translate to space to avoid joining words)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)

        # remove non-printable characters
        text = "".join(ch for ch in text if ch.isprintable())

        # collapse repeated punctuation (e.g., '!!!' -> '!')
        text = re.sub(r"([!?.]){2,}", r"\1", text)
        # reduce elongated characters (>2 repeats -> 2 repeats) e.g. heyyyy -> heyy
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # optionally remove digits entirely
        if remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # tokenize with spaCy (Vietnamese) and filter stopwords/punct/space
        doc = nlp(text)
        cleaned_tokens = []
        for token in doc:
            # Keep tokens that are not stopwords/punctuation/space
            if not token.is_stop and not token.is_punct and not token.is_space:
                # filter out tokens that contain digits (optional already handled above)
                if not any(char.isdigit() for char in token.text):
                    cleaned_tokens.append(token.lower_)

        return " ".join(cleaned_tokens)

    except Exception as e:
        print("Error normalizing text:", e)
        return ""


def Age2Vect(age: str):
    start_age, end_age = age.strip().split("-")
    return [x for x in range(int(start_age), int(end_age) + 1)]


sentiments2id = {v: k for k, v in enumerate(df_train["sentiment"].unique())}
time2id = {v: k for k, v in enumerate(df_train["Time of Tweet"].unique())}
country2id = {v: k for k, v in enumerate(df_train["Country"].unique())}

id2sentiments = {k: v for k, v in enumerate(df_train["sentiment"].unique())}
id2time = {k: v for k, v in enumerate(df_train["Time of Tweet"].unique())}
id2country = {k: v for k, v in enumerate(df_train["Country"].unique())}


df_train["text"] = df_train["text"].apply(text_normalize)
df_train["sentiment"] = df_train["sentiment"].map(lambda x: sentiments2id[x])
df_train["Time of Tweet"] = df_train["Time of Tweet"].map(lambda x: time2id[x])
df_train["Country"] = df_train["Country"].map(lambda x: country2id[x])
df_train["Age of User"] = df_train["Age of User"].map(lambda x: Age2Vect(x))

df_test["text"] = df_test["text"].apply(text_normalize)
df_test["sentiment"] = df_test["sentiment"].map(lambda x: sentiments2id[x])
df_test["Time of Tweet"] = df_test["Time of Tweet"].map(lambda x: time2id[x])
df_test["Country"] = df_test["Country"].map(lambda x: country2id[x])
df_test["Age of User"] = df_test["Age of User"].map(lambda x: Age2Vect(x))

df_train.to_csv("../dataset/cleaned/train_cleaned.csv", index=False)
df_test.to_csv("../dataset/cleaned/test_cleaned.csv", index=False)


import json 

mappings = {
    'sentiments2id': sentiments2id,
    'time2id': time2id, 
    'country2id': country2id,
    'id2sentiments': id2sentiments,
    'id2time': id2time,
    'id2country': id2country
}

# Save từng file riêng
for name, mapping_dict in mappings.items():
    filename = f"{name}.json"
    with open(f"../dataset/utils/{filename}", 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=4, ensure_ascii=False)
    print(f"Saved {filename}")
