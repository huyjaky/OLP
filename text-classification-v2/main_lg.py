import pytorch_lightning as L
from get_data_loader import (
    training_loader,
    validation_loader,
    testing_loader,
    X_train,
    y_train,
    X_val,
    y_val,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional.classification import accuracy
import torch
from torch import nn

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import GradientBoostingRegressor


class TextClassifier(L.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            num_embeddings=31000, embedding_dim=input_dim, padding_idx=1
        )  # 32x64x64
        self.trans = nn.Transformer(
            d_model=input_dim,
            activation="gelu",
            batch_first=True,
            dropout=0.2,
            num_encoder_layers=4,
            num_decoder_layers=2,
            nhead=8,
            # dim_feedforward=64*256,
            # norm_first=True,
        )  # 32x64x64
        self.dense = nn.Linear(input_dim, output_dim)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = accuracy

        # Define your model architecture here

    def forward(self, x):
        x = self.embedding(x)  # 32x64x64
        transformer_out = self.trans.encoder(x)  # 32x64x64
        transformer_out = self.dense(transformer_out[:, 0, :])  # get [CLS] token  32x2
        return transformer_out  # 32x2

    def test_val_step(self, batch, batch_idx):
        x, y = batch
        logits: torch.Tensor = self(x)  # 32x2
        loss = self.criterion(logits, y)
        acc = self.accuracy(
            logits.argmax(dim=-1),
            y.argmax(dim=-1),
            num_classes=self.output_dim,
            task="multiclass",
        )  # type: ignore
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.test_val_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return {
            "loss": loss,
            "accuracy": acc,
        }

    def validation_step(self, batch, batch_idx):
        loss, acc = self.test_val_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

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
    lr = LogisticRegression()
    vector = TfidfVectorizer()
    x = vector.fit_transform(X_train).toarray()  # type: ignore

    x_val1 = vector.transform(X_val).toarray()  # type: ignore
    lr.fit(x, y_train)

    pred = lr.predict(x_val1)
    score = accuracy_score(y_val, pred)
    print("Accuracy:", score)


if __name__ == "__main__":
    model = TextClassifier(input_dim=64, output_dim=2)
    train(model)
