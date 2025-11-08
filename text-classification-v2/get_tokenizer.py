from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, WordPiece, Unigram
from tokenizers.trainers import (
    WordLevelTrainer,
    BpeTrainer,
    WordPieceTrainer,
    UnigramTrainer,
)
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd

encodings = ["utf-7", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

for enc in encodings:
    try:
        df = pd.read_csv("./Attachments/train.csv", encoding=enc)
        print(f"Read finish with: {enc}")
        print(df.head())
        break
    except UnicodeDecodeError:
        print(f"Cant read with: {enc}")

corpus = df["question_text"].apply(lambda x: str(x)).tolist()  # type: ignore

data_size = len(corpus)


def create_tokenizer(vocab_size: int = 31000, sequence_length: int = 64):
    # tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))  # type: ignore
    tokenizer = Tokenizer(Unigram()) # type: ignore
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)
    trainer = UnigramTrainer(
        vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]"],
        unk_token="[UNK]",
    )
    # trainer = WordPieceTrainer(
    #     vocab_size=vocab_size,
    #     special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]"],
    #     unk_token="[UNK]",
    # )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save("./Attachments/tokenizer-uni.json")
    return tokenizer
