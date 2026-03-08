import os
import shutil
import warnings
import random
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)

warnings.filterwarnings("ignore")


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Data Paths
    TRAIN_PATH = 'data/train_data.csv'
    VAL_PATH = 'data/val_data.csv'
    TEST_PATH = 'data/test_data.csv'

    # Save Paths
    CHECKPOINT_DIR = 'checkpoints'
    MODEL_NAME = 'stock_sentiment_classifier'

    # Model Selection
    # Choose from: 'bert', 'roberta', 'modified_roberta', 'hybrid_gru'
    MODEL_TYPE = 'bert'

    # Pretrained backbones
    PRETRAINED_MODELS = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'modified_roberta': 'roberta-base',
        'hybrid_gru': 'bert-base-uncased'
    }

    # Hyperparameters
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 3
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 0.01
    RANDOM_SEED = 42

    # Hybrid GRU params
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 1
    BIDIRECTIONAL = True

    # Modified RoBERTa params
    HIDDEN_FC = 256

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)


# ==========================================
# UTILITIES
# ==========================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    return pd.read_csv(path)


def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        shutil.copyfile(checkpoint_path, best_path)
        print(f"--> New best model saved to: {best_path}")


def get_model_name(model_type: str) -> str:
    if model_type not in Config.PRETRAINED_MODELS:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")
    return Config.PRETRAINED_MODELS[model_type]


# ==========================================
# DATASET
# ==========================================
class StockTweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if 'body' not in self.df.columns or 'target' not in self.df.columns:
            raise ValueError("CSV must contain 'body' and 'target' columns.")

        self.texts = [self._clean_text(text) for text in self.df['body'].astype(str).tolist()]
        self.targets = self.df['target'].astype(int).tolist()

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.texts[index]
        target = self.targets[index]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

        # Some models/tokenizers (like RoBERTa) may not return token_type_ids
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        else:
            item['token_type_ids'] = torch.zeros(self.max_len, dtype=torch.long)

        return item


# ==========================================
# MODEL DEFINITIONS
# ==========================================
class BERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if 'bert' in self.encoder.config.model_type else None
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        return logits


class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        return logits


class ModifiedRoBERTaClassifier(nn.Module):
    """
    Modified RoBERTa:
    - RoBERTa backbone
    - CLS token representation
    - Two fully connected layers
    - LayerNorm + GELU
    """
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float, hidden_fc: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_fc)
        self.norm = nn.LayerNorm(hidden_fc)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_fc, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout1(cls_embedding)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


class HybridGRUClassifier(nn.Module):
    """
    Hybrid GRU model:
    - Transformer encoder embeddings
    - BiGRU on top of token embeddings
    - Mean pooling + max pooling + last hidden state
    - Dense classifier
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout_rate: float,
        gru_hidden_size: int,
        gru_num_layers: int,
        bidirectional: bool = True
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        gru_output_dim = gru_hidden_size * 2 if bidirectional else gru_hidden_size

        # mean pool + max pool + final hidden
        combined_dim = gru_output_dim * 3

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(combined_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if 'bert' in self.encoder.config.model_type else None
        )

        sequence_output = outputs.last_hidden_state  # [B, L, H]
        gru_output, hidden = self.gru(sequence_output)  # gru_output: [B, L, G]

        mean_pool = torch.mean(gru_output, dim=1)
        max_pool, _ = torch.max(gru_output, dim=1)

        if self.gru.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        x = torch.cat([mean_pool, max_pool, final_hidden], dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


def build_model(model_type: str) -> nn.Module:
    model_name = get_model_name(model_type)

    if model_type == 'bert':
        return BERTClassifier(
            model_name=model_name,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE
        )

    if model_type == 'roberta':
        return RoBERTaClassifier(
            model_name=model_name,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE
        )

    if model_type == 'modified_roberta':
        return ModifiedRoBERTaClassifier(
            model_name=model_name,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE,
            hidden_fc=Config.HIDDEN_FC
        )

    if model_type == 'hybrid_gru':
        return HybridGRUClassifier(
            model_name=model_name,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE,
            gru_hidden_size=Config.GRU_HIDDEN_SIZE,
            gru_num_layers=Config.GRU_NUM_LAYERS,
            bidirectional=Config.BIDIRECTIONAL
        )

    raise ValueError(f"Invalid model type: {model_type}")


# ==========================================
# TRAIN / EVAL FUNCTIONS
# ==========================================
def compute_metrics(preds, labels) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch_idx: int
) -> Tuple[float, Dict[str, float]]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"\n[Epoch {epoch_idx}] Training...")

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(targets.detach().cpu().numpy())

        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f"   Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


def eval_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    mode: str = "Validation"
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"{mode}...")

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = loss_fn(outputs, targets)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(targets.detach().cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics, np.array(all_preds), np.array(all_labels)


# ==========================================
# MAIN
# ==========================================
def main():
    set_seed(Config.RANDOM_SEED)

    print("=" * 60)
    print(f"Using Device: {Config.DEVICE}")
    print(f"Selected Model: {Config.MODEL_TYPE}")
    print(f"Backbone: {get_model_name(Config.MODEL_TYPE)}")
    print("=" * 60)

    # Load data
    try:
        train_df = load_data(Config.TRAIN_PATH)
        val_df = load_data(Config.VAL_PATH)
        test_df = load_data(Config.TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Validate labels
    valid_labels = set(range(Config.NUM_CLASSES))
    if not set(train_df['target'].unique()).issubset(valid_labels):
        raise ValueError(f"Train labels must be in {valid_labels}")
    if not set(val_df['target'].unique()).issubset(valid_labels):
        raise ValueError(f"Val labels must be in {valid_labels}")
    if not set(test_df['target'].unique()).issubset(valid_labels):
        raise ValueError(f"Test labels must be in {valid_labels}")

    # Tokenizer
    tokenizer_name = get_model_name(Config.MODEL_TYPE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Datasets
    train_dataset = StockTweetDataset(train_df, tokenizer, Config.MAX_LEN)
    val_dataset = StockTweetDataset(val_df, tokenizer, Config.MAX_LEN)
    test_dataset = StockTweetDataset(test_df, tokenizer, Config.MAX_LEN)

    # Loaders
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = build_model(Config.MODEL_TYPE)
    model.to(Config.DEVICE)

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    total_training_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(0.1 * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.DEVICE, epoch
        )

        val_loss, val_metrics, _, _ = eval_epoch(
            model, val_loader, criterion, Config.DEVICE, mode="Validation"
        )

        print(f"\nEpoch {epoch} Results")
        print("-" * 40)
        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Train Accuracy:   {train_metrics['accuracy']:.4f}")
        print(f"Train Macro F1:   {train_metrics['macro_f1']:.4f}")
        print(f"Val Loss:         {val_loss:.4f}")
        print(f"Val Accuracy:     {val_metrics['accuracy']:.4f}")
        print(f"Val Macro F1:     {val_metrics['macro_f1']:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print("Validation loss improved. Saving best model...")

        checkpoint_state = {
            'epoch': epoch,
            'model_type': Config.MODEL_TYPE,
            'backbone_name': get_model_name(Config.MODEL_TYPE),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss
        }

        save_checkpoint(checkpoint_state, is_best, Config.CHECKPOINT_DIR)

    print("\nTraining complete.")

    # Load best model for testing
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['state_dict'])

    # Final test evaluation
    test_loss, test_metrics, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, Config.DEVICE, mode="Testing"
    )

    print("\nFinal Test Results")
    print("=" * 60)
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print("\nClassification Report")
    print("=" * 60)
    print(classification_report(
        test_labels,
        test_preds,
        digits=4,
        target_names=['Negative', 'Neutral', 'Positive']
    ))


if __name__ == "__main__":
    main()
