VOCAB_SIZE = 50257
import sentencepiece as spm


def train_spm(input_file, model_prefix, vocab_size=VOCAB_SIZE):
    """Train a SentencePiece BPE model."""
    args = (
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--model_type=bpe --character_coverage=1.0 "
        "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
        "--num_subworkers=3"
    )
    spm.SentencePieceTrainer.Train(args)
    print(f"Trained SentencePiece model: {model_prefix}.model")


def load_sp(model_path):
    """Load a trained SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

from dotenv import load_dotenv
import os
load_dotenv('../.env')

tokenizer_save_path = "./tokenizer_model/spm_bpe_tokenizer"
with open(os.getenv('CLEANED_UNSUPERVISED_DATASET_PATH'), "r", encoding="utf-8") as f:
    train_spm(
        input_file=os.getenv('CLEANED_UNSUPERVISED_DATASET_PATH'),
        model_prefix=tokenizer_save_path,
        vocab_size=VOCAB_SIZE,
    )

