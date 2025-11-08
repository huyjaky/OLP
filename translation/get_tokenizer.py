from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, WordPiece, Unigram
from tokenizers.trainers import (
    WordLevelTrainer,
    BpeTrainer,
    WordPieceTrainer,
    UnigramTrainer,
)
from tokenizers.pre_tokenizers import Whitespace



def create_tokenizer(corpus, vocab_size=3000, sequence_length=64, cache_path=None):
    tokenizer = Tokenizer(Unigram())
    # tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=sequence_length)
    tokenizer.enable_truncation(max_length=sequence_length)
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="[UNK]",
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]", "[CLS]"],
    )
    # trainer = WordPieceTrainer(
    #     vocab_size=vocab_size,
    #     unk_token="[UNK]",
    #     special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]", "[CLS]"],
    # )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save(cache_path) if cache_path else None
    return tokenizer
