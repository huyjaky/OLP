from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel, WordPiece, Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import (
    BpeTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
    UnigramTrainer,
)

def create_tokenizer(corpus: list, vocab_size: int = 3000, sequence_length: int = 64):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    # tokenizer = Tokenizer(Unigram())  # type: ignore
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)
    # trainer = UnigramTrainer(
    #     vocab_size=vocab_size,
    #     special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]"],
    #     unk_token="[UNK]",
    # )
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[RS]", "[WS]", "[MASK]", "[CLS]"],
        unk_token="[UNK]",
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save("./Attachments/tokenizer-level.json")
    return tokenizer
