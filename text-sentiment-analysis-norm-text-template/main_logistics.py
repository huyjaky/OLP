import torch
import pytorch_lightning as L
from DataTransforms.get_data_loader import x_train, y_train, x_val, y_val, train_loader, val_loader
from torch import nn
from torchmetrics.functional.classification import accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.metrics import accuracy_score,classification_report, ConfusionMatrixDisplay
from pytorch_lightning.loggers import TensorBoardLogger


class LogisticRegressionModel(nn.Module):
    def __init__(self, inum_features, output_dim):
        super().__init__()
        self.linear = nn.Linear(inum_features, output_dim)

    def forward(self, x):
        return self.linear(x)


class Sentiment_Analysis_LSTM_model(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.embedding = nn.Embedding(31000, 256, padding_idx=1)  # 64x64x256
        self.fc_1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.logistic_1 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Loss and Acc
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = accuracy  # Đảm bảo bạn đã import hàm này

    def forward(self, x, time=None, age=None, country=None):
        logits = self.embedding(x)  
        logits = self.fc_1(logits)
        logits = self.logistic_1(logits.mean(dim=1))
        print("Logits shape:", logits.shape)
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
    # logger = TensorBoardLogger("tb_logs", name="Sentiment_Analysis_LSTM_model")
    # trainer = L.Trainer(
    #     max_epochs=12,
    #     accelerator="auto",
    #     devices="auto",
    #     logger=logger,
    #     gradient_clip_val=1.0,
    #     # log_every_n_steps=25,
    # )
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model, dataloaders=test_loader)

    lr = LogisticRegression()
    vector = TfidfVectorizer()
    x = x_train.drop(columns=["Age of User", "Time of Tweet", "Country"])
    x = vector.fit_transform(x["text"]).toarray()

    x_val1 = x_val.drop(columns=["Age of User", "Time of Tweet", "Country"])
    x_val1 = vector.transform(x_val1["text"]).toarray()

    lr.fit(x, y_train)

    pred = lr.predict(x_val1)
    score = accuracy_score(y_val, pred)
    print("Accuracy:", score)


if __name__ == "__main__":
    model = Sentiment_Analysis_LSTM_model(num_classes=3)
    train(model)
