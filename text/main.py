import torch
from torchmetrics.functional.classification import accuracy
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from DataTransforms.get_data_loader import get_data_loaders, id2label
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=8
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_results = {"logits": [], "labels": [], "texts": []}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, attention_mask, y = batch[0], batch[1], batch[2]
        logits = self(x).logits
        loss = self.criterion(logits, y)
        train_accuracy = accuracy(logits, y, num_classes=8, task="multiclass")

        return {
            "loss": loss,
            "accuracy": train_accuracy,
        }

    # def test_step(self, batch, batch_idx: int):
    #     x, attention_mask = batch[0], batch[1]
    #     logits = self(x).logits
    #     return logits.argmax(dim=-1).detach().cpu()

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return {
            "optimizer": optimizer,
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=5
            ),
            "monitor": "train_loss, train_accuracy",
            "interval": "epoch",
            "frequency": 1,
        }

    def validation_step(self, batch):
        x, attention_mask, y, text = batch[0], batch[1], batch[2], batch[3]
        logits = self(x).logits

        self.validation_results["logits"].extend(
            logits.argmax(dim=-1).detach().cpu().numpy()
        )
        self.validation_results["labels"].extend(
            y.argmax(dim=-1).detach().cpu().numpy()
        )
        self.validation_results["texts"].extend(text)

    def on_validation_epoch_end(self):
        all_logits = self.validation_results["logits"]
        all_labels = self.validation_results["labels"]

        ## NOTE: Plot confusion matrix to TensorBoard
        cm = confusion_matrix(
            all_labels,
            all_logits,
            labels=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        fig, ax = plt.subplots(figsize=(6, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[id2label[idx] for idx in range(8)]
        )
        disp.plot(ax=ax)

        f1_micro = f1_score(
            self.validation_results["logits"],
            self.validation_results["labels"],
            average="micro",
        )

        self.logger.experiment.add_figure(
            "validation/confusion_matrix", fig, self.current_epoch
        )
        self.log("validation/f1_micro", f1_micro, prog_bar=True)

        # NOTE: log incorrect predictions to TensorBoard

        incorrect_indices = [
            sample
            for sample in range(len(self.validation_results["logits"]))
            if self.validation_results["logits"][sample]
            != self.validation_results["labels"][sample]
        ]

        count = 0
        for idx in incorrect_indices:
            if count >= 10:
                break
            log_text = f"{self.validation_results['texts'][idx]}"
            self.logger.experiment.add_text(
                "validation/incorrect_predictions", log_text, self.current_epoch
            )
            count += 1

        self.validation_results["logits"].clear()
        self.validation_results["labels"].clear()
        self.validation_results["texts"].clear()
        print(f"F1 Micro: {f1_micro}")


def train(model):
    logger = TensorBoardLogger("tb_logs", name="Bert_model")
    trainer = L.Trainer(max_epochs=5, accelerator="auto", devices="auto", logger=logger)
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    model = MyModel()
    train(model)
