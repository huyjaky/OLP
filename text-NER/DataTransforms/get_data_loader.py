import pandas as pd
from tokenizers import CharBPETokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import json
from transformers import BertTokenizer

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]


for enc in encodings:
    try:
        df = pd.read_csv("Attachments/NER_dataset_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")


class CustomCollator(Dataset):
    def __init__(self, sentences, pos_ids, tokenizer, max_length, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = sentences
        self.pos_ids = pos_ids
        self.is_train = is_train

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.pos_ids[idx]
        label = json.loads(label)



        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.is_train:
            if len(label) > self.max_length:
                label = label[: self.max_length]
            else:
                pad_len = self.max_length - len(label)
                label = label + ([-100] * pad_len)  # -100 is the default ignore_index
            return encoding["input_ids"].squeeze(0), torch.tensor(label, dtype=torch.long)
        else:
            return encoding["input_ids"].squeeze(0)


# bpe_tokenizer = CharBPETokenizer(
#     vocab="../Attachments/NER_tokenizer/vocab.json",
#     merges="../Attachments/NER_tokenizer/merges.txt",
# )

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

x_train, x_val, y_train, y_val = train_test_split(
    df["Sentence"].tolist(), df["POS_id"].tolist(), test_size=0.4, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_val, y_val, test_size=0.5, random_state=42
)

training_dataset = CustomCollator(
    sentences=x_train,
    pos_ids=y_train,
    tokenizer=tokenizer,
    max_length=64,
    is_train=True,
)
validation_dataset = CustomCollator(
    sentences=x_val,
    pos_ids=y_val,
    tokenizer=tokenizer,
    max_length=64,
    is_train=True,
)

testing_dataset = CustomCollator(
    sentences=x_test,
    pos_ids=y_test,
    tokenizer=tokenizer,
    max_length=64,
    is_train=False,
)

train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, drop_last=True)
test_loader = DataLoader(testing_dataset, shuffle=False)
