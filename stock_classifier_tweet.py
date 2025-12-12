import os
import shutil
import warnings
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    """Hyperparameters and configuration settings."""
    TRAIN_PATH = 'data/train_data.csv'
    VAL_PATH = 'data/val_data.csv'
    TEST_PATH = 'data/test_data.csv'
    
    # Model Saving
    CHECKPOINT_DIR = 'checkpoints'
    MODEL_NAME = 'bert_stock_classifier'
    
    # Hyperparameters
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-05
    NUM_CLASSES = 3  # Assuming [Negative, Neutral, Positive]
    DROPOUT_RATE = 0.3
    
    # Device Configuration
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Pretrained Model
    BERT_PATH = 'bert-base-uncased'

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)


# ==========================================
# DATASET CLASS
# ==========================================
class StockTweetDataset(Dataset):
    """Custom Dataset for Stock Tweet Sentiment Analysis."""
    
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df
        
        # Pre-clean text once during init to save time during training
        # column names are 'body' (text) and 'target' (label)
        self.texts = [self._clean_text(text) for text in df['body']]
        self.targets = df['target'].values

    def _clean_text(self, text: str) -> str:
        """Basic whitespace cleaning."""
        return " ".join(str(text).split())

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.texts[index]
        target = self.targets[index]

        # Tokenization
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class BERTClassifier(nn.Module):
    """BERT-based classifier for sentiment analysis."""
    
    def __init__(self, num_classes: int, dropout_rate: float):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_PATH, return_dict=True)
        self.dropout = nn.Dropout(dropout_rate)
        # 768 is the hidden size of bert-base-uncased
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Feed input to BERT
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # We use the pooler_output (representation of [CLS] token)
        pooled_output = output.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def save_checkpoint(state: Dict, is_best: bool, checkpoint_dir: str):
    """Saves model checkpoint and updates the best model if applicable."""
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        shutil.copyfile(f_path, best_path)
        print(f"--> New best model saved at {best_path}")

def load_data(path: str) -> pd.DataFrame:
    """Robust data loading with error checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)


# ==========================================
# TRAINING ENGINE
# ==========================================
def train_epoch(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    epoch_idx: int
) -> float:
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0
    
    print(f'\n[Epoch {epoch_idx}] Training...')
    
    for batch_idx, data in enumerate(data_loader):
        # Move data to device
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        # Forward pass
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f"   Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    return avg_loss


def validate_epoch(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_fn: nn.Module, 
    device: torch.device
) -> float:
    """Runs validation step."""
    model.eval()
    running_loss = 0.0
    
    print('Validating...')
    with torch.no_grad():
        for data in data_loader:
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    return avg_loss


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"Using Device: {Config.DEVICE}")
    
    # 1. Load Data
    try:
        print("Note: Attempting to load files. Ensure Config.TRAIN_PATH points to real CSVs.")
        train_df = load_data(Config.TRAIN_PATH)
        val_df = load_data(Config.VAL_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please fix the file paths in the Config class.")
        return

    # 2. Tokenizer & Datasets
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)
    
    train_dataset = StockTweetDataset(train_df, tokenizer, Config.MAX_LEN)
    val_dataset = StockTweetDataset(val_df, tokenizer, Config.MAX_LEN)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.TRAIN_BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.VAL_BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )

    # 3. Model Setup
    model = BERTClassifier(Config.NUM_CLASSES, Config.DROPOUT_RATE)
    model.to(Config.DEVICE)

    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.LEARNING_RATE)
    
    # CRITICAL FIX: Use CrossEntropyLoss for multi-class targets (0, 1, 2)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, epoch)
        val_loss = validate_epoch(model, val_loader, criterion, Config.DEVICE)

        print(f"\nResult Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  Validation loss decreased. Saving best model...")
        
        checkpoint_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        save_checkpoint(checkpoint_state, is_best, Config.CHECKPOINT_DIR)

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
