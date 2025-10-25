import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from DataTransforms.get_tokenizer import create_tokenizer
from sklearn.model_selection import train_test_split
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]


for enc in encodings:
    try:
        df = pd.read_csv("Attachments/train_cleaned.csv", encoding=enc).dropna()
        # df = pd.read_csv("../Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")


def MLM(text, mask_prob=0.3):
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
        text = self.text[idx]
        # text = MLM(text) 

        sentiment = [0 for _ in range(3)]
        sentiment[self.sentiment[idx]] = 1

        time = [0 for _ in range(3)]
        time[self.time[idx]] = 1

        age = [0 for _ in range(100)]
        age_ids = self.age[idx]
        for age_id in json.loads(age_ids):
            age[int(age_id)-1] = 1

        country = [0 for _ in range(195)]
        country[self.country[idx]] = 1

        encoding = self.tokenizer.encode(text).ids

        return (
            torch.tensor(encoding, dtype=torch.long), # 64x64
            torch.tensor(sentiment, dtype=torch.float), # 64x3
            torch.tensor(time, dtype=torch.long), # 64x3
            torch.tensor(age, dtype=torch.long), # 64x100
            torch.tensor(country, dtype=torch.long), # 64x195
        )


# bpe_tokenizer = CharBPETokenizer(
#     vocab="../Attachments/NER_tokenizer/vocab.json",
#     merges="../Attachments/NER_tokenizer/merges.txt",
# )

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tokenizer = create_tokenizer()

x_train, x_val, y_train, y_val = train_test_split(
    df[["text", "Age of User", "Time of Tweet", "Country"]],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
)

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
    input_data=x_val,
    label=y_val,
    tokenizer=tokenizer,
    max_length=64,
    is_train=True,
)

# testing_dataset = CustomCollator(
#     sentences=x_test,
#     pos_ids=y_test,
#     tokenizer=tokenizer,
#     max_length=64,
#     is_train=False,
# )

train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(
    validation_dataset, batch_size=64, shuffle=False, drop_last=True
)
# test_loader = DataLoader(testing_dataset, shuffle=False)


