import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models.model import SequenceRNNClassifier, SequenceRNNConfig

with open("configs/train.yaml", encoding="utf-8") as f:
    TRAIN_CFG = yaml.safe_load(f)["train"]

def load_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).long()
    
    label_map = json.loads(str(data["label_map"]))
    return X, y, label_map

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path("data/processed/train.npz")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run dataset pipeline first.")
        return

    X, y, label_map = load_dataset(data_path)
    num_classes = len(label_map)
    input_dim = X.shape[2] # (N, T, Features)

    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * TRAIN_CFG["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"])

    config = SequenceRNNConfig(input_dim=input_dim, num_classes=num_classes)
    model = SequenceRNNClassifier(config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])

    print(f"Training on {device} with {len(train_ds)} samples, {num_classes} classes.")

    best_acc = 0
    for epoch in range(TRAIN_CFG["epochs"]):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = correct / val_size if val_size > 0 else 0
        print(f"Epoch [{epoch+1}/{TRAIN_CFG['epochs']}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save(Path("models/trained/best_model.pt"), extra={"label_map": label_map})

if __name__ == "__main__":
    train()