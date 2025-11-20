from torch.utils.data import DataLoader, Dataset
import pandas as pd
from datasets import load_dataset
from get_tokenizer import create_tokenizer
import torch
from norm_text import text_normalize

# ds = load_dataset("thainq107/iwslt2015-en-vi", cache_dir="./data_cache")
ds = load_dataset("./Attachments/")

# convert train_dataset to pandas dataframe
# train_df = pd.DataFrame(ds["train"]).map(text_normalize)
# val_df = pd.DataFrame(ds["validation"]).map(text_normalize)
# test_df = pd.DataFrame(ds["test"]).map(text_normalize)
train_df = pd.DataFrame(ds["train"])
val_df = pd.DataFrame(ds["validation"])
test_df = pd.DataFrame(ds["test"])

vi_tokenizer = create_tokenizer(
    corpus=train_df["vi"].tolist(), cache_path="./vi_tokenizer.json"
)
en_tokenizer = create_tokenizer(
    corpus=train_df["en"].tolist(), cache_path="./en_tokenizer.json"
)




class CustomCollator(Dataset):
    def __init__(self, source_texts, target_texts, source_tokenizer, target_tokenizer):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        source_encoding = self.source_tokenizer.encode(source_text)

        if self.target_texts is None:  # for test case
            return torch.tensor(source_encoding, dtype=torch.long)

        target_text = "[SOS]" + self.target_texts[idx] + "[EOS]"
        target_encoding = self.target_tokenizer.encode(target_text)

        return (
            torch.tensor(source_encoding.ids, dtype=torch.long),
            torch.tensor(source_encoding.attention_mask, dtype=torch.float),
            torch.tensor(target_encoding.ids, dtype=torch.long),
            torch.tensor(target_encoding.attention_mask, dtype=torch.float),
        )


train_dataset = CustomCollator(
    train_df["en"].tolist(),
    train_df["vi"].tolist(),
    en_tokenizer,
    vi_tokenizer,
)
val_dataset = CustomCollator(
    val_df["en"].tolist(),
    val_df["vi"].tolist(),
    en_tokenizer,
    vi_tokenizer,
)

test_dataset = CustomCollator(
    source_texts=test_df["en"].tolist(),
    target_texts=None,
    source_tokenizer=en_tokenizer,
    target_tokenizer=vi_tokenizer,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    pin_memory_device="cuda",
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    pin_memory_device="cuda",
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    pin_memory_device="cuda",
)

for sample in train_loader:
    break
