import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from DataTransforms.get_data_loader import train_loader, val_loader
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
import math

from torchmetrics.functional.classification import accuracy


class Sentiment_Analysis_LSTM_model(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.embedding = nn.Embedding(31000, 256, padding_idx=1)
        self.transformer = nn.Transformer(
            d_model=256,
            activation="gelu",
            batch_first=True,
            dropout=0.2,
            num_encoder_layers=4,
            num_decoder_layers=2,
            nhead=4,
            # dim_feedforward=64*256,
            norm_first=True,
        )
        self.output_head = nn.Linear(256, self.num_classes)

        # Loss and Acc
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = accuracy  
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)


    def forward(self, x):

        # x = torch.concat([x, time, age, country], dim=1)  # 64x(128+3+100+195)=64x426

        x = self.embedding(x)  # 64x64x256
        transformer_out = self.transformer.encoder(x)  # 64x64x256

        transformer_out = self.output_head(
            transformer_out[:, 0, :]
        )  # get [CLS] token  64xnum_classes

        return transformer_out  # 64xnum_classes

    def training_step(self, batch, batch_idx: int):

        # torch.tensor(encoding, dtype=torch.long), # 64x64
        # torch.tensor(time, dtype=torch.long), # 64x3
        # torch.tensor(age, dtype=torch.long), # 64x100
        # torch.tensor(country, dtype=torch.long), # 64x195

        x, y, time, age, country = batch[0], batch[1], batch[2], batch[3], batch[4]
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.accuracy(
            logits.argmax(dim=-1),
            y.argmax(dim=-1),
            num_classes=self.num_classes,
            task="multiclass",
            ignore_index=self.num_classes,
        )

        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return {
            "loss": loss,
            "accuracy": acc,
        }

    def validation_step(self, batch, batch_idx: int):

        x, y, time, age, country = batch[0], batch[1], batch[2], batch[3], batch[4]
        logits = self(x)

        loss = self.criterion(logits, y)

        acc = self.accuracy(
            logits.argmax(dim=-1),
            y.argmax(dim=-1),
            num_classes=self.num_classes,
            task="multiclass",
            ignore_index=self.num_classes,
        )

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x, y = batch[0], batch[1]
        # Note: test dataloader in this repo isn't wired-up; keep behavior consistent
        # with training/validation steps: convert y to indices then compute loss/acc.
        logits = self(x, None, None, None)
        y_idx = y.argmax(dim=-1).long()
        loss = self.criterion(logits, y_idx)
        acc = self.accuracy(
            logits.argmax(dim=-1),
            y_idx,
            num_classes=self.num_classes,
            task="multiclass",
            ignore_index=self.num_classes,
        )
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        return {
            "optimizer": optimizer,
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }


def train(model):
    logger = TensorBoardLogger("tb_logs", name="Sentiment_Analysis_LSTM_model")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/Sentiment_Analysis_LSTM_model",
        filename="sentiment-analysis-lstm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="max",
    )
    trainer = L.Trainer(
        max_epochs=7,
        accelerator="auto",
        devices="auto",
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model, dataloaders=test_loader)



if __name__ == "__main__":

    model = Sentiment_Analysis_LSTM_model(num_classes=3)
    train(model)
