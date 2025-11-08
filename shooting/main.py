import pytorch_lightning as L
from torch import nn
from torchmetrics.functional.classification import accuracy
from get_data_loader import training_dataloader, validation_dataloader
import torch


class SimpleFC(L.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim  # 32x4
        self.output_dim = output_dim  # 32x1

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )

        self.criterion = nn.BCEWithLogitsLoss()
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
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features).squeeze()  # (batch, 1) -> (batch,)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels, task="binary")

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)

        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features).squeeze()  # (batch, 1) -> (batch,)
        # logits = logits.argmax(dim=-1)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels, task="binary")
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": acc}

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
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        # log_every_n_steps=25,
    )
    trainer.fit(
        model,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )


if __name__ == "__main__":
    model = SimpleFC(4, 1)
    train(model)
