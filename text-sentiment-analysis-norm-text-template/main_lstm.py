import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from DataTransforms.get_data_loader import train_loader, val_loader
from torch import nn
from torchmetrics.functional.classification import accuracy

class Sentiment_Analysis_LSTM_model(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.embedding = nn.Embedding(31000, 256, padding_idx=1)
        self.proj = nn.Linear(256, 128) # 64x64x128

        self.lstm_1 = nn.LSTM(
            128,
            256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,  # Dropout giữa các lớp LSTM
        )  # 64x64x512

        self.dropout_1 = nn.Dropout(0.4)
        self.layer_norm_1 = nn.LayerNorm(512)

        self.lstm_2 = nn.LSTM(
            512,
            256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.3,  # Dropout giữa các lớp LSTM
        )  # Output: (batch, seq_len, 256)

        self.dropout_2 = nn.Dropout(0.4)
        self.layer_norm_2 = nn.LayerNorm(256)

        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        self.tail = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes),
        )

        # Loss and Acc
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = accuracy  # Đảm bảo bạn đã import hàm này

    def forward(self, x, time=None, age=None, country=None):
        text_model = self.embedding(x)
        text_model = self.proj(text_model)  # 64x64x128
        lstm_out, _ = self.lstm_1(text_model)  # 64x64x512
        lstm_out = self.dropout_1(lstm_out)
        lstm_out = self.layer_norm_1(lstm_out)
        lstm_out, (hidden, _) = self.lstm_2(lstm_out)  # 64x256
        lstm_out = self.dropout_2(hidden.squeeze(0))  # Sử dụng hidden state cuối cùng
        lstm_out = self.layer_norm_2(lstm_out)
        attn_output, _ = self.attn(lstm_out, lstm_out, lstm_out)  # 64x256
        logits = self.tail(attn_output)  # 64xnum_classes

        return logits

    def training_step(self, batch, batch_idx: int):
        x, y, time, age, country = batch[0], batch[1], batch[2], batch[3], batch[4]
        logits = self(x, time, age, country)

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
        # torch.tensor(encoding, dtype=torch.long), # 64x64
        # torch.tensor(sentiment, dtype=torch.float), # 64x3
        # torch.tensor(time, dtype=torch.long), # 64x3
        # torch.tensor(age, dtype=torch.long), # 64x100
        # torch.tensor(country, dtype=torch.long), # 64x195

        x, y, time, age, country = batch[0], batch[1], batch[2], batch[3], batch[4]
        logits = self(x, time, age, country)
        y_idx = y.argmax(dim=-1).long()
        loss = self.criterion(logits, y_idx)

        acc = self.accuracy(
            logits.argmax(dim=-1),
            y_idx,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
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
    trainer = L.Trainer(
        max_epochs=12,
        accelerator="auto",
        devices="auto",
        logger=logger,
        gradient_clip_val=1.0,
        # log_every_n_steps=25,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    model = Sentiment_Analysis_LSTM_model(num_classes=3)
    train(model)
