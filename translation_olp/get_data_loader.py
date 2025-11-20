from torch.utils.data import DataLoader, Dataset
from get_tokenizer import create_tokenizer
from tokenizers import Tokenizer
import torch
from transformers import T5Tokenizer


def read_lines(path):
    """Read lines from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


train_src = read_lines("./dataset/train/train.zh")
train_tgt = read_lines("./dataset/train/train.vi")
test_src = read_lines("./dataset/public_test/public_test.zh")


class CustomCollator(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = "[SOS]" + self.src_sentences[idx].replace("_", " ") + "[EOS]"

        decoder_input_sentence =  "[SOS]" + self.tgt_sentences[idx].replace("_", " ")
        tgt_sentence = self.tgt_sentences[idx].replace("_", " ") + "[EOS]"

        src_encoded = self.src_tokenizer.encode(src_sentence)
        dec_input_encoded = self.tgt_tokenizer.encode(decoder_input_sentence)
        tgt_encoded = self.tgt_tokenizer.encode(tgt_sentence)

        return {
            "encoder_input": torch.tensor(src_encoded.ids),
            "decoder_input": torch.tensor(dec_input_encoded.ids),
            "tgt": torch.tensor(tgt_encoded.ids),

            "encoder_input_mask": torch.tensor(src_encoded.attention_mask),
            "decoder_input_mask": torch.tensor(dec_input_encoded.attention_mask),
            "tgt_mask": torch.tensor(tgt_encoded.attention_mask),
        }

src_tokenzier = create_tokenizer(train_src, cache_path="./Attachments/src_tokenizer.json", sequence_length=128)
tgt_tokenzier = create_tokenizer(train_tgt, cache_path="./Attachments/tgt_tokenizer.json", sequence_length=128)

train_dataset = CustomCollator(train_src[:28000], train_tgt[:28000], src_tokenzier, tgt_tokenzier)
val_dataset = CustomCollator(train_src[28000:], train_tgt[28000:], src_tokenzier, tgt_tokenzier)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)





