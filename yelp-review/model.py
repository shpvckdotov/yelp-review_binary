from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchmetrics import Accuracy, F1Score, Precision, Recall


class LSTMClassifierLightning(pl.LightningModule):
    """Lightning модуль для классификатора текстов"""

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer: str = "adam",
    ):
        """
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_dim: Размерность скрытого состояния LSTM
            output_dim: Количество классов
            n_layers: Количество слоев LSTM
            dropout: Dropout вероятность
            pad_idx: Индекс padding токена
            learning_rate: Скорость обучения
            weight_decay: Коэффициент L2 регуляризации
            optimizer: Оптимизатор ('adam' или 'adamw')
        """
        super().__init__()
        self.save_hyperparameters()

        # Модель
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        # Метрики

        self._setup_metrics()
        # Функция потерь
        self.loss_fn = nn.CrossEntropyLoss()

    def _setup_metrics(self):
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.train_f1 = F1Score(task="multiclass", num_classes=2)
        self.val_f1 = F1Score(task="multiclass", num_classes=2)
        self.test_f1 = F1Score(task="multiclass", num_classes=2)

        self.train_precision = Precision(task="multiclass", num_classes=2)
        self.val_precision = Precision(task="multiclass", num_classes=2)
        self.test_precision = Precision(task="multiclass", num_classes=2)

        self.train_recall = Recall(task="multiclass", num_classes=2)
        self.val_recall = Recall(task="multiclass", num_classes=2)
        self.test_recall = Recall(task="multiclass", num_classes=2)

    def forward(self, text: torch.Tensor, attention_mask) -> torch.Tensor:
        """Forward pass"""
        return self.model(text, attention_mask)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Шаг обучения"""
        texts, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        # Forward pass
        logits = self(texts, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Логирование
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Расчет accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.train_f1(preds, labels)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.train_precision(preds, labels)
        self.log(
            "train_precision",
            self.train_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.train_recall(preds, labels)
        self.log(
            "train_recall",
            self.train_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Шаг валидации"""
        texts, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        # Forward pass
        logits = self(texts, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Логирование
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Расчет accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_f1(preds, labels)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.val_precision(preds, labels)
        self.log(
            "val_precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_recall(preds, labels)
        self.log(
            "val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Шаг тестирования"""
        texts, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        # Forward pass
        logits = self(texts, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Логирование
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Расчет accuracy
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_f1(preds, labels)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.test_precision(preds, labels)
        self.log(
            "test_precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.test_recall(preds, labels)
        self.log(
            "test_recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True
        )

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Шаг предсказания"""
        texts, atention_mask = batch["input_ids"], batch["attention_mask"]

        # Forward pass
        logits = self(texts, atention_mask)
        probs = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "predictions": torch.argmax(probs, dim=1),
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Настройка оптимизатора и scheduler"""
        # Выбор оптимизатора
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        return {"optimizer": optimizer}
