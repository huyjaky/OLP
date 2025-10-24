import pandas as pd

# from DataTransforms.test import create_tokenizer
import re

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]


for enc in encodings:
    try:
        df_train = pd.read_csv("../Attachments/train.csv", encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df_train.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

for enc in encodings:
    try:
        df_test = pd.read_csv("../Attachments/test.csv", encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df_test.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")


def text_normalize(text):
    try:
        text = str(text)
        text = text.lower()  # lowercase
        text = re.sub(r"^RT[\s]+", "", text)  # remove RT in tweet
        # text = re.sub(r"https|:\/\/.*[\r\n]*", "", text)  # remove URLs
        # text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"https?://[^\s]+(?:\s*[-:])?", "", text)
        text = re.sub(r"[^\w\s]", "", text)  # remove special characters
        text = re.sub(r"<.*?>", "", str(text))  # remove HTML tags
        text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text))  # remove special characters
        text = re.sub(r"\s+", " ", str(text)).strip()  # remove extra spaces
    except Exception as e:
        print("Error normalizing text:", e)
    return text


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

df_train.to_csv("../Attachments/train_cleaned.csv", index=False)
df_test.to_csv("../Attachments/test_cleaned.csv", index=False)
