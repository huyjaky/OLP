from torch import nn
import torch
import pytorch_lightning as L
import math
from torchmetrics.functional.text import sacre_bleu_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)  # getting buffer, not a parameter

    def forward(self, x):
        # X is embedded input of shape 32x64x64
        x = self.dropout(
            x + self.pe[:, : x.size(1), :]
        )  # add positional encoding to input (32x64x64)
        return x


class MyModel(L.LightningModule):
    def __init__(self, input_dim, output_dim, vocab_size, tokenizer):
        super().__init__()
        self.PAD_IDX = 0
        self.SOS_IDX = 2
        self.EOS_IDX = 3

        self.tokenizer = tokenizer

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        # NOTE: input is 32x64
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=input_dim, padding_idx=self.PAD_IDX
        )  # 32x64x64

        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=input_dim, padding_idx=self.PAD_IDX
        )  # 32x64x64

        self.pos_encoding = PositionalEncoding(
            d_model=input_dim, dropout=0.2, max_len=64
        )

        self.trans = nn.Transformer(
            d_model=input_dim,
            nhead=8,
            activation="gelu",
            num_encoder_layers=5,
            num_decoder_layers=5,
            # dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )  # 32x64x64

        self.tail = nn.Linear(64, vocab_size)  # 32x64x3000

        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.criterion = nn.CrossEntropyLoss()
        self.blue = sacre_bleu_score

    def forward(
        self,
        source_encoding,  # [w1, w2, w3, EOS, PAD, PAD,...]
        target_encoding,  # [SOS, w1, w2, w3, EOS, PAD, PAD,...]
        encoder_attn_mask,
        decoder_attn_mask,
    ):

        tgt_mask = self.trans.generate_square_subsequent_mask(self.input_dim).to("cuda")

        src_embedded = self.pos_encoding(
            self.src_embedding(source_encoding) * math.sqrt(self.input_dim)
        )
        tgt_embedded = self.pos_encoding(
            self.tgt_embedding(target_encoding) * math.sqrt(self.input_dim)
        )

        logits = self.trans(
            src=src_embedded,
            tgt=tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=encoder_attn_mask,
            tgt_key_padding_mask=decoder_attn_mask,
        )

        logits = self.tail(logits)  # 32x64x3000

        return logits


    def predict_step(self, sentence: str):
            max_len = self.pos_encoding.pe.size(1)

            # self.tokenizer.enable_truncation(max_length=max_len)
            # self.tokenizer.enable_padding(
            #     length=max_len, pad_id=self.PAD_IDX, pad_token="[PAD]"
            # )

            encoding = self.tokenizer.encode(sentence)

            source_encoding = torch.tensor([encoding.ids], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long).to(
                self.device
            )
            encoder_attn_mask = (attention_mask == 0).to(self.device)

            batch_size = 1

            src_embedded = self.pos_encoding(
                self.src_embedding(source_encoding) * math.sqrt(self.input_dim)
            )

            memory = self.trans.encoder(
                src_embedded, src_key_padding_mask=encoder_attn_mask
            )

            tgt_tokens = (
                torch.ones((batch_size, 1), dtype=torch.long)
                * self.SOS_IDX
            ).to(self.device)

            for _ in range(max_len - 1):
                current_seq_len = tgt_tokens.size(1)

                tgt_embedded = self.pos_encoding(
                    self.tgt_embedding(tgt_tokens) * math.sqrt(self.input_dim)
                )

                tgt_mask = self.trans.generate_square_subsequent_mask(current_seq_len).to(
                    self.device
                )

                decoder_output = self.trans.decoder(
                    tgt=tgt_embedded,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=encoder_attn_mask,
                )

                logits = self.tail(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1)

                if next_token.item() == self.EOS_IDX:
                    break

                tgt_tokens = torch.cat((tgt_tokens, next_token.unsqueeze(1)), dim=1)


            return tgt_tokens
