from typing import Any
import dvc.api
import pandas as pd
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pyarrow as pa
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer



class YelpPolarityDataset(Dataset):
    """Dataset для Yelp Polarity данных из Arrow файлов"""
    
    def __init__(self, df, transform=None):
        """
        Args:
            data_bytes: байты, прочитанные через dvc.api.read()
            transform: преобразования для данных
        """
        
        self.texts = df['text'].values
        self.labels = df['label'].values
        
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        if self.transform:
            text = self.transform(text)
            
        return text, label

class YelpPolarityTokenizedDataset(YelpPolarityDataset):
    """Dataset с токенизацией для трансформеров"""
    
    def __init__(self, df, tokenizer=None, max_length=512):
        super().__init__(df)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Токенизация
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Убираем batch dimension для отдельных примеров
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class YelpPolarityDataModule(LightningDataModule):
    """Lightning DataModule для Yelp Polarity"""
    
    def __init__(
        self,
        train_path='data/yelp-train.csv',
        test_path='data/yelp-test.csv',
        batch_size=32,
        num_workers=4,
        seed = 42,
        tokenizer=None,
        max_length=512,
        val_split=0.1
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_split = val_split
        self.seed=seed
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        """Создаем датасеты (вызывается на каждом GPU)"""
        with dvc.api.open(
            'data/yelp_train.csv',
        ) as f:
            train_df = pd.read_csv(f)
        
        with dvc.api.open(
            'data/yelp_test.csv',
        ) as f:
            test_df = pd.read_csv(f)
        
        # Создаем полный тренировочный датасет
        full_train_dataset = YelpPolarityTokenizedDataset(
            train_df,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Разделяем на train/val
        val_size = int(len(full_train_dataset) * self.val_split)
        train_size = len(full_train_dataset) - val_size
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size], generator
        )
        # Тестовый датасет
        self.test_dataset = YelpPolarityTokenizedDataset(
            test_df,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )