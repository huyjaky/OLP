from transformers import T5Tokenizer, GptOssConfig, GptOssForSequenceClassification

import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from tokenizers import Tokenizer
from get_dataloader import train_loader, val_loader
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F


class MyModel(L.LightningModule):
    def __init__(
        self, input_dim, output_dim, vocab_size, tokenizer: T5Tokenizer, model_name: str
    ):
        super().__init__()
        self.PAD_IDX = 0
        self.SOS_IDX = 2
        self.EOS_IDX = 3
        self.BEST_WEIGHT = 0.0

        self.model_name = model_name

        self.tokenizer = tokenizer

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        config = GptOssConfig(
            vocab_size=self.vocab_size,
            pad_token_id=self.PAD_IDX,
            decoder_start_token_id=self.SOS_IDX,
            bos_token_id=self.SOS_IDX,
            eos_token_id=self.EOS_IDX,
            num_labels=3,
            
            hidden_size=512,           # ↓ from 768 (saves ~40% memory)
            num_hidden_layers=8,       # ↓ from 12
            num_attention_heads=8,     # Keep 8 (512/8=64 head_dim ✓)
            num_key_value_heads=4,     # GQA: saves KV cache memory
            intermediate_size=2048,    # ↓ from 3072 (4x hidden_size)
            num_local_experts=4,       # ↓ from 8 (MOE memory heavy)
            
            use_cache=True,
            pretraining_tp=1,          # No tensor parallelism
        )
        self.model = GptOssForSequenceClassification(config=config)
        self.tail = nn.Sequential(nn.Softmax(dim=-1))

        self.criterion = nn.CrossEntropyLoss()
        self.acc = MulticlassAccuracy(num_classes=3)
        # self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        encoder_input: torch.Tensor,  # [ w1, w2, w3, PAD, PAD,...]
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        logits = self.model(
            input_ids=encoder_input,
            attention_mask=attention_mask,
            labels=labels.argmax(dim=-1),
        )

        return logits

    def train_val_step(self, batch, batch_idx):
        encoder_input, attention_mask, labels = batch

        logits: torch.Tensor = self(encoder_input, attention_mask, labels)

        loss = logits.loss
        logits = logits.logits

        probs = F.softmax(logits, dim=-1)
        acc = self.acc(probs, labels.argmax(dim=-1))

        return loss, logits, acc

    def training_step(self, batch, batch_idx):
        loss, logits, acc = self.train_val_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        loss, logits, acc = self.train_val_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.trainer.logged_metrics.get("val_acc")
        if val_acc and val_acc > self.BEST_WEIGHT:
            self.BEST_WEIGHT = val_acc
            self.save_pretrain(f"./weights/{self.model_name}_acc-{val_acc}")

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        return {
            "optimizer": optimizer,
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3
            ),
            "monitor": "val_acc",
            "interval": "epoch",
            "frequency": 1,
        }

    def save_pretrain(self, path: str):
        self.model.save_pretrained(path)


def train(model):
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=CSVLogger("logs/", name=model.model_name),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("./tokenizer_model/spm_bpe_tokenizer.model")
    tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }
    )
    model_name = "sentiment-bert-model"

    model = MyModel(
        input_dim=64,
        output_dim=2,
        vocab_size=len(tokenizer.get_vocab()),
        tokenizer=tokenizer,
        model_name=model_name,
    ).to("cpu")
    train(model)
