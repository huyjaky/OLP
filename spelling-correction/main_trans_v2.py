import math
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchmetrics.functional.text import sacre_bleu_score
from get_data_loader import train_loader, val_loader, test_loader
from tokenizers import Tokenizer


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
            d_model=input_dim, dropout=0.2, max_len=input_dim
        )

        self.trans = nn.Transformer(
            d_model=input_dim,
            nhead=8,
            activation="gelu",
            num_encoder_layers=20,
            num_decoder_layers=0,
            # dim_feedforward=1024,
            dropout=0.2,
            batch_first=True,
        )  # 32x64x64

        self.tail = nn.Sequential(
            nn.Linear(self.input_dim, vocab_size)
        )  # 32x64x3000

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        # self.criterion = nn.CrossEntropyLoss()
        self.blue = sacre_bleu_score

    def forward(
        self,
        source_encoding,  # [w1, w2, w3, PAD, PAD,...]
        encoder_attn_mask,
    ):

        src_embedded = self.pos_encoding(
            self.src_embedding(source_encoding) * math.sqrt(self.input_dim)
        )

        print("src_embedded shape:", src_embedded.tolist())

        logits = self.trans.encoder(
            src=src_embedded, src_key_padding_mask=~encoder_attn_mask
        ) # 32x64x64


        logits = self.tail(logits)  # 32x64x3000
        return logits

    def train_val_step(self, batch, batch_idx):
        (
            wrong_encoding,  # [w1, w2, w3, PAD, PAD,...]
            wrong_attention_mask,
            right_encoding,
            right_attention_mask,
            target_encoding,  # [w1, w2, w3, PAD, PAD,...]
            target_attention_mask,
        ) = (
            batch["wrong_input_ids"],
            batch["wrong_attention_mask"],
            batch["right_input_ids"],
            batch["right_attention_mask"],
            batch["target_input_ids"],
            batch["target_attention_mask"],
        )


        logits: torch.Tensor = self(
            source_encoding=wrong_encoding,
            encoder_attn_mask=wrong_attention_mask.bool(),
        )  # 32x64x3000

        loss = self.criterion(
            logits.view(-1, self.vocab_size), target_encoding.view(-1)
        )

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.train_val_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        loss, pred_id = self.train_val_step(batch, batch_idx)
        target_id = batch["right_input_ids"]  # tgt_encoding_4_encoder [w1, w2, w3, EOS]
        input_id = batch["wrong_input_ids"]

        pred_id = pred_id.argmax(dim=-1)  # 32x64

        pred_texts = []
        target_texts = []
        input_texts = []

        for pred, target, input_ in zip(pred_id, target_id, input_id):
            pred_list = pred.tolist()  # [w1, w2, w3, EOS, PAD, PAD, PAD,...]
            target_list = target.tolist()  # [SOS, w1, w2, w3, EOS, PAD, PAD, PAD,...]
            input_list = input_.tolist()

            input_texts.append(
                self.tokenizer.decode(input_list, skip_special_tokens=True)
            )

            target_texts.append(
                self.tokenizer.decode(target_list, skip_special_tokens=True)
            )
            pred_texts.append(
                self.tokenizer.decode(pred_list, skip_special_tokens=True)
            )

        print("----validation example----")
        print("Target:\n", target_texts[0])
        print("Input:\n", input_texts[0])
        print("Pred:\n", pred_texts[0])
        score = self.blue(pred_texts, [[ref] for ref in target_texts])
        self.log("val_acc", score, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, logits = self.train_val_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)

        pred_id = logits.argmax(dim=-1)  # pred_id: [w1, w2, w3, EOS]
        target_id = batch["right_input_ids"]  # tgt_encoding_4_encoder [w1, w2, w3, EOS]

        pred_texts = []
        target_texts = []
        for pred, target in zip(pred_id, target_id):
            pred_list = pred.tolist()  # [w1, w2, w3, EOS, PAD, PAD, PAD,...]
            target_list = target.tolist()  # [SOS, w1, w2, w3, EOS, PAD, PAD, PAD,...]

            target_texts.append(
                self.tokenizer.decode(target_list, skip_special_tokens=True)
            )
            pred_texts.append(
                self.tokenizer.decode(pred_list, skip_special_tokens=True)
            )

        score = self.blue(pred_texts, [[ref] for ref in target_texts])
        self.log("test_acc", score, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3
            ),
            "monitor": "val_acc",
            "interval": "epoch",
            "frequency": 1,
        }


def train(model):
    model_name = "spelling_correction_pretrain"
    logger = TensorBoardLogger("tb_logs", name=model_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"checkpoints/{model_name}",
        filename=model_name + "-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    trainer = L.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    src_tokenizer = Tokenizer.from_file("./Attachments/tokenizer-level.json")

    model = MyModel(
        input_dim=256,
        output_dim=2,
        vocab_size=31000,
        tokenizer=src_tokenizer,
    )
    train(model)
