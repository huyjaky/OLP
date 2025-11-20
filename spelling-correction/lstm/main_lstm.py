import math
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchmetrics.functional.text import sacre_bleu_score
from get_data_loader import train_loader, val_loader, test_loader
from tokenizers import Tokenizer


# NOTE: input has shape 32x64
class Encoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, padding_idx=0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            # bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embedding(src))  # 32x64x256
        outputs, (hidden, cell) = self.lstm(
            embedded
        )  # hidden: 4x32x256, cell: 4x32x256
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, output_dim, embed_dim, hidden_dim, n_layers, dropout, padding_idx=0
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            # bidirectional=True,
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, hidden, cell):
        embedded = self.dropout(self.embedding(tgt))  # 32x1x256
        outputs, (hidden, cell) = self.lstm(
            embedded, (hidden, cell)
        )  # outputs: 32x1x512
        predictions = self.fc_out(outputs)  # 32x1x3000
        return predictions, hidden, cell


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

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=input_dim,
            hidden_dim=input_dim,
            n_layers=2,
            dropout=0.1,
            padding_idx=self.PAD_IDX,
        )
        self.decoder = Decoder(
            output_dim=vocab_size,
            embed_dim=input_dim,
            hidden_dim=input_dim,
            n_layers=2,
            dropout=0.1,
            padding_idx=self.PAD_IDX,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        # self.criterion = nn.CrossEntropyLoss()
        self.blue = sacre_bleu_score

    def forward(
        self,
        source_encoding,  # [w1, w2, w3, EOS, PAD, PAD,...]
        target_encoding,  # [w1, w2, w3, EOS, PAD, PAD,...]
        encoder_attn_mask,
        decoder_attn_mask,
    ):
        tgt_len = target_encoding.shape[1]
        batch_size = target_encoding.shape[0]

        hidden, cell = self.encoder(
            source_encoding,
        )  # 4x32x256

        logits = torch.full(
            (batch_size, 1), self.SOS_IDX, dtype=torch.long, device=self.device
        ) # 32x1 (SOS)

        # NOTE: teacher forcing
        for _ in range(63):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                logits,
                hidden,
                cell,
            )  # decoder_output: 32x1x3000
            logits = torch.cat(
                (logits, decoder_output.argmax(dim=-1)), dim=1
            )  # 32xseq_len

        return logits

    def train_val_step(self, batch, batch_idx):
        (
            wrong_encoding,
            wrong_attention_mask,
            right_encoding,
            right_attention_mask,
            target_encoding,
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
            wrong_encoding,
            right_encoding,
            encoder_attn_mask=wrong_attention_mask.bool(),
            decoder_attn_mask=right_attention_mask.bool(),
        )  # 32x64x3000

        loss = self.criterion(
            logits.permute(0,2,1),  # 32x3000x64
            target_encoding  # 32x64
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
        pred_id = pred_id.argmax(dim=-1)  # 32x64

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
        self.log("val_acc", score, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode="min", patience=3
            # ),
            "monitor": "val_acc",
            "interval": "epoch",
            "frequency": 1,
        }


def train(model):
    model_name = "spelling_correction"
    logger = TensorBoardLogger("tb_logs", name=model_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"checkpoints/{model_name}",
        filename=model_name + "-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    src_tokenizer = Tokenizer.from_file("./Attachments/tokenizer-level.json")

    model = MyModel(
        input_dim=64,
        output_dim=2,
        vocab_size=3000,
        tokenizer=src_tokenizer,
    )
    train(model)
