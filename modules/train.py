import subprocess

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_scheduler


class SentimentClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes=3, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        acc = self.accuracy(logits.softmax(dim=-1), batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        wandb.log({"train_loss": loss.item(), "train_acc": acc.item()})

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        acc = self.accuracy(logits.softmax(dim=-1), batch["labels"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        wandb.log({"val_loss": loss.item(), "val_acc": acc.item()})

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        acc = self.accuracy(logits.softmax(dim=-1), batch["labels"])
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        wandb.log({"test_loss": loss.item(), "test_acc": acc.item()})

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train():
    wandb.init(
        project="stock-news-analysis",
        entity="stepanovprogramming-HSE",
        name="my-legit-run-3",
    )

    df = pd.read_csv(
        "all-data.csv",
        names=["sentiment", "text"],
        encoding="utf-8",
        encoding_errors="replace",
    ).dropna()

    label_encoder = LabelEncoder()
    df["sentiment_encoded"] = label_encoder.fit_transform(df["sentiment"])

    X = df["text"].tolist()
    y = df["sentiment_encoded"].tolist()

    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    max_len = 128

    def tokenize_texts(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return encodings, torch.tensor(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_encodings, train_labels = tokenize_texts(X_train, y_train)
    test_encodings, test_labels = tokenize_texts(X_test, y_test)

    class SentimentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    wandb_logger = WandbLogger()

    model_name = "bert-base-uncased"
    model = SentimentClassifier(model_name=model_name)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    model_path = "sentiment_model.pth"
    torch.save(model.state_dict(), model_path)

    subprocess.run(["dvc", "add", model_path])
    subprocess.run(["git", "add", model_path + ".dvc"])
    subprocess.run(["git", "commit", "-m", "Add trained model to DVC"])
    subprocess.run(["dvc", "push"])
