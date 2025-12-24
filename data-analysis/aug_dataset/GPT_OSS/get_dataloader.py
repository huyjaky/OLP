import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import random
from transformers import T5Tokenizer
from dotenv import load_dotenv
import os

load_dotenv("../.env")

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]
tokenizer = T5Tokenizer.from_pretrained("./tokenizer_model/spm_bpe_tokenizer.model")
tokenizer.add_special_tokens(
    {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
)

for enc in encodings:
    try:
        df = pd.read_csv(os.getenv("CLEANED_TRAIN_DATASET_PATH"), encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")


for enc in encodings:
    try:
        df_test = pd.read_csv(
            os.getenv("CLEANED_TEST_DATASET_PATH"), encoding=enc
        ).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")


def MLM(text, mask_prob=0.2):
    tokens = text.split()
    if not tokens:
        return text

    num_to_mask = max(1, int(len(tokens) * mask_prob))
    mask_indices = random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))

    for idx in mask_indices:
        tokens[idx] = "[MASK]"

    return " ".join(tokens)


class CustomCollator(Dataset):
    def __init__(self, input_data, label, tokenizer, max_length, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text = input_data["text"].tolist()
        self.sentiment = label.tolist()
        self.time = input_data["Time of Tweet"].tolist()
        self.age = input_data["Age of User"].tolist()
        self.country = input_data["Country"].tolist()
        self.is_train = is_train

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = "<s>" + self.text[idx] + "</s>"

        sentiment = [0 for _ in range(3)]
        sentiment[self.sentiment[idx]] = 1

        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=64
        )

        return (
            torch.tensor(encoding.input_ids, dtype=torch.long),  # 64x64
            torch.tensor(encoding.attention_mask, dtype=torch.long),  # 64x64
            torch.tensor(sentiment, dtype=torch.float),  # 64x3
        )


# bpe_tokenizer = CharBPETokenizer(
#     vocab="../Attachments/NER_tokenizer/vocab.json",
#     merges="../Attachments/NER_tokenizer/merges.txt",
# )

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


x_train, x_val, y_train, y_val = train_test_split(
    df[["text", "Age of User", "Time of Tweet", "Country"]],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
)
print(x_train)


# x_val, x_test, y_val, y_test = train_test_split(
#     x_val, y_val, test_size=0.5, random_state=42
# )

training_dataset = CustomCollator(
    input_data=x_train,
    label=y_train,
    tokenizer=tokenizer,
    max_length=64,
    is_train=True,
)
validation_dataset = CustomCollator(
    input_data=df[["text", "Age of User", "Time of Tweet", "Country"]],
    label=df["sentiment"],
    tokenizer=tokenizer,
    max_length=64,
    is_train=True,
)

testing_dataset = CustomCollator(
    input_data=x_val,
    label=y_val,
    tokenizer=tokenizer,
    max_length=64,
    is_train=True,
)

train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(
    validation_dataset, batch_size=64, shuffle=False, drop_last=True
)
test_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False, drop_last=False)
# test_loader = DataLoader(testing_dataset, shuffle=False)
