from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, WordPiece
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import re

encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

for enc in encodings:
    try:
        df = pd.read_csv("Attachments/train_cleaned.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

corpus = df["text"].apply(lambda x: str(x)).tolist()

data_size = len(corpus)

vocab_size = 31000
sequence_length = 64

def create_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)
    trainer = BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[SOS]"]
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer

