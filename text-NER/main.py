
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from DataTransforms.get_data_loader import train_loader, val_loader
from torch import nn
# from torchmetrics.functional.detection


class NER_LSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(31000, 256, padding_idx=1)
        self.proj = nn.Linear(256, 128)
        self.lstm_1 = nn.LSTM(
            128, 256, num_layers=4, batch_first=True, bidirectional=True
        )
        self.dropout_1 = nn.Dropout(0.2)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.lstm_2 = nn.LSTM(
            512, 256, num_layers=4, batch_first=True, bidirectional=False
        )
        self.dropout_2 = nn.Dropout(0.2)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        # Project to class logits per token
        self.atn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fc = nn.Linear(256, 64)
        # Ignore padded label positions (-100) in loss
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        x = self.embedding(x)
        x = self.proj(x)
        x, _ = self.lstm_1(x)
        x = self.dropout_1(x)
        x = self.batch_norm_1(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm_2(x)
        x = self.dropout_2(x)
        x = self.batch_norm_2(x.transpose(1, 2)).transpose(1, 2)
        # x: (batch, seq_len, hidden*2) -> (batch, seq_len, num_classes)
        x, _ = self.atn(x, x, x)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx: int):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx: int):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
    logger = TensorBoardLogger("tb_logs", name="NER_LSTM_model")
    trainer = L.Trainer(
        max_epochs=9,
        accelerator="auto",
        devices="auto",
        logger=logger,
        gradient_clip_val=1.0,
        # log_every_n_steps=25,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    model = NER_LSTM()
    train(model)
