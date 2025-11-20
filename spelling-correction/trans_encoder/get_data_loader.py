import pandas as pd
from norm_text import text_normalize
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from get_tokenizer import create_tokenizer

df = pd.read_json("./Attachments/VSEC.jsonl", lines=True)



annotations = list(df["annotations"])


def get_full_right_sequence(sample: list, get_wrong_indices=False):
    full_sequence = ""

    for idx, character in enumerate(sample):
        if character["alternative_syllables"] != []:
            full_sequence += (
                " " + character["alternative_syllables"][0]
            )  # ignore multiple case for now
        else:
            full_sequence += " " + character["current_syllable"]
    return text_normalize(full_sequence.strip())


def get_full_wrong_sequence(sample: list):
    full_sequence = ""
    for character in sample:
        full_sequence += " " + character["current_syllable"]
    return text_normalize(full_sequence.strip())


df["right_sequence"] = [get_full_right_sequence(sample) for sample in annotations]
df["wrong_sequence"] = [get_full_wrong_sequence(sample) for sample in annotations]


class CustomCollator(Dataset):
    def __init__(self, right_sequences, wrong_sequences, tokenizer):
        self.right_sequences = right_sequences
        self.wrong_sequences = wrong_sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.right_sequences)

    def __getitem__(self, idx) -> dict:
        right_text = self.right_sequences[idx] 
        wrong_text = self.wrong_sequences[idx]
        target_text = self.right_sequences[idx]

        right_encoding = self.tokenizer.encode(right_text)
        wrong_encoding = self.tokenizer.encode(wrong_text)
        target_encoding = self.tokenizer.encode(target_text)

        item = {
            "right_input_ids": torch.tensor(
                right_encoding.ids, dtype=torch.long
            ),  # 32x64
            "right_attention_mask": torch.tensor(
                right_encoding.attention_mask, dtype=torch.float
            ),  # 32x64
            "wrong_input_ids": torch.tensor(
                wrong_encoding.ids, dtype=torch.long
            ),  # 32x64
            "wrong_attention_mask": torch.tensor(
                wrong_encoding.attention_mask, dtype=torch.float
            ),  # 32x64
            "target_input_ids": torch.tensor(
                target_encoding.ids, dtype=torch.long
            ),  # 32x64
            "target_attention_mask": torch.tensor(
                target_encoding.attention_mask, dtype=torch.float
            ),  # 32x64
        }
        return item


X_train, X_val, y_train, y_val = train_test_split(
    df[["wrong_sequence"]],
    df[["right_sequence"]],
    test_size=0.4,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val, test_size=0.5, random_state=42
)


tokenizer = create_tokenizer(corpus=df["right_sequence"].tolist())

train_dataset = CustomCollator(
    right_sequences=y_train["right_sequence"].tolist(),
    wrong_sequences=X_train["wrong_sequence"].tolist(),
    tokenizer=tokenizer,
)

val_dataset = CustomCollator(
    right_sequences=y_val["right_sequence"].tolist(),
    wrong_sequences=X_val["wrong_sequence"].tolist(),
    tokenizer=tokenizer,
)

test_dataset = CustomCollator(
    right_sequences=y_test["right_sequence"].tolist(),
    wrong_sequences=X_test["wrong_sequence"].tolist(),
    tokenizer=tokenizer,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
