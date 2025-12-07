from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel, WordPiece, Unigram
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.trainers import (
    BpeTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
    UnigramTrainer,
)
from tokenizers.normalizers import BertNormalizer, ByteLevel, NFC, Sequence


def create_tokenizer(
    corpus: list[str],
    cache_path: str,
    vocab_size: int = 9000,
    sequence_length: int = 128,
) -> Tokenizer:
    # tokenizer = Tokenizer(Unigram())
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([BertNormalizer()])
    tokenizer.pre_tokenizer = Punctuation()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)
    # trainer = UnigramTrainer(
    #     vocab_size=vocab_size,
    #     unk_token="[UNK]",
    #     special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]", "[CLS]"],
    # )
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        unk_token="[UNK]",
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SEP]", "[MASK]", "[CLS]"],
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save(cache_path)
    return tokenizer


