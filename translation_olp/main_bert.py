import math
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchmetrics.functional.text import sacre_bleu_score
from get_data_loader import train_loader, val_loader
from tokenizers import Tokenizer
from transformers import T5ForConditionalGeneration, T5Config



class MyModel(L.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        vocab_size,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
    ):
        super().__init__()
        self.PAD_IDX = 0
        self.SOS_IDX = 2
        self.EOS_IDX = 3

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        # NOTE: input is 32x64
        self.src_embedding = nn.Embedding(
            num_embeddings=self.src_tokenizer.get_vocab_size(), embedding_dim=input_dim, padding_idx=self.PAD_IDX
        )  # 64x64x128

        self.tgt_embedding = nn.Embedding(
            num_embeddings=self.tgt_tokenizer.get_vocab_size(), embedding_dim=input_dim, padding_idx=self.PAD_IDX
        )  # 64x64x128


        config = T5Config(
            vocab_size=self.src_tokenizer.get_vocab_size(), 
            pad_token_id=self.PAD_IDX,
            eos_token_id=self.EOS_IDX, 
            bos_token_id=self.SOS_IDX
        )
        self.model = T5ForConditionalGeneration(config=config)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)

        # self.criterion = nn.CrossEntropyLoss()
        self.blue = sacre_bleu_score

    def forward(
        self,
        encoder_input: torch.Tensor,  # [ w1, w2, w3, PAD, PAD,...]
        decoder_input: torch.Tensor,  # [SOS, w1, w2, w3, PAD, PAD,...]
        label: torch.Tensor # [w1, w2, w3, EOS, PAD, PAD,...]
    ):
        logits = self.model(encoder_input, decoder_input_ids=decoder_input, labels=label)
        return logits.logits



    def train_val_step(self, batch, batch_idx):
        (
            encoder_input,
            decoder_input,
            tgt,
            encoder_input_mask,
            decoder_input_mask,
            tgt_mask,
        ) = (
            batch["encoder_input"],
            batch["decoder_input"],
            batch["tgt"],
            batch["encoder_input_mask"],
            batch["decoder_input_mask"],
            batch["tgt_mask"],
        )

        logits: torch.Tensor = self(
            encoder_input,
            decoder_input,
            tgt
        )  # 64xsequence_lenx3000

        loss = self.criterion(
            logits.view(-1, self.tgt_tokenizer.get_vocab_size()), tgt.view(-1)
        )  # flatten to (batch_size*seq_len, vocab_size)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.train_val_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        pred_id = self.model.generate(batch['encoder_input'])

        target_id = batch["tgt"]  # tgt_encoding_4_encoder [w1, w2, w3, EOS]

        pred_texts = []
        target_texts = []

        for pred, target in zip(pred_id, target_id):
            pred_list = pred.tolist()  # [w1, w2, w3, EOS, PAD, PAD, PAD,...]
            target_list = target.tolist()  # [SOS, w1, w2, w3, EOS, PAD, PAD, PAD,...]

            target_texts.append(
                self.tgt_tokenizer.decode(target_list, skip_special_tokens=True)
            )
            pred_texts.append(
                self.tgt_tokenizer.decode(pred_list, skip_special_tokens=True)
            )

        score = self.blue(pred_texts, [[ref] for ref in target_texts])
        self.log("val_acc", score, prog_bar=True, on_epoch=True)

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
    model_name = "translation_olp_trans"
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
    # trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    src_tokenizer = Tokenizer.from_file("./Attachments/src_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("./Attachments/tgt_tokenizer.json")

    model = MyModel(
        input_dim=128,
        output_dim=2,
        vocab_size=9000,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    ).to("cpu")
    train(model)
