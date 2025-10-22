from tokenizers import CharBPETokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from .transforms import text_normalize, MLM
import pandas as pd

# Define target names for reference
id2label = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology",
}

# Define dataset class
class MathDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256, is_train=True):
        # pass
        ## TO-DO
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # pass
        ## TO-DO
        label = [0, 0, 0, 0, 0, 0, 0, 0]
        text = self.texts[idx]

        if self.is_train:
            text = text_normalize(text)
            text = MLM(text)

        encoding = self.tokenizer.encode(text)

        input_ids = encoding.ids[: self.max_len]
        attention_mask = encoding.attention_mask[: self.max_len]
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        if self.is_train:
            label[self.labels[idx]] = 1
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(label, dtype=torch.float),
                f"{self.texts[idx]}"
            )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            f"{self.texts[idx]}",
        )


# Define paths
TRAIN_PATH = "Attachments/train.csv"
TEST_PATH = "Attachments/test.csv"
SAMPLE_SUBMISSION_PATH = "Attachments/sample_submission.csv"

df = pd.read_csv(TRAIN_PATH).sample(frac=1).reset_index(drop=True)
train_df = df[:8000]
val_df = df[8000:]
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

def get_data_loaders(batch_size):
    bpe_tokenizer = CharBPETokenizer(
        "Attachments/tokenizer/vocab.json", "Attachments/tokenizer/merges.txt"
    )

    training_dataset = MathDataset(
        texts=train_df["Question"].values,
        labels=train_df["label"].values,
        tokenizer=bpe_tokenizer,
    )

    validating_dataset = MathDataset(
        texts=val_df["Question"].values,
        labels=val_df["label"].values,
        tokenizer=bpe_tokenizer,
    )

    testing_dataset = MathDataset(
        texts=test_df["Question"].values, tokenizer=bpe_tokenizer, is_train=False
    )

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validating_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(testing_dataset, shuffle=False)

    return train_loader, val_loader, test_loader
