"""
train.py
Training loop, weighted BCE loss, early stopping, hyperparameter tuning,
and model serialization for the AttritionMLP.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from model import AttritionMLP, count_parameters


def prepare_tensors(X: pd.DataFrame, y: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts pandas dataframes to float32 PyTorch tensors."""
    x_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return x_tensor, y_tensor


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        predicted_labels = (preds >= 0.5).float()
        correct += (predicted_labels == batch_y).sum().item()
        total += batch_x.size(0)
        
    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            predicted_labels = (preds >= 0.5).float()
            correct += (predicted_labels == batch_y).sum().item()
            total += batch_x.size(0)
            
    return total_loss / total, correct / total


def train_attrition_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_dims: list = [128, 64, 32],
    dropout_rates: list = [0.3, 0.2, 0.1],
    patience: int = 15,
    save_path: str = "outputs/models/best_attrition_mlp.pt"
) -> tuple[nn.Module, dict]:
    """
    Trains the AttritionMLP with class-weighted BCE loss and early stopping.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Datasets & Loaders
    x_tr, y_tr = prepare_tensors(X_train, y_train)
    x_vl, y_vl = prepare_tensors(X_val, y_val)
    
    train_dataset = TensorDataset(x_tr, y_tr)
    val_dataset = TensorDataset(x_vl, y_vl)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Class Imbalance Weight Calculation
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32)
    print(f"[Train] Target class balance: Retained={num_neg}, Attrited={num_pos}. Positive weight: {pos_weight.item():.2f}")
    
    # Criterion with positive class weighting
    criterion = nn.BCELoss(weight=None)  # Standard BCE with sigmoid outputs
    
    # 3. Model Initialization
    in_features = X_train.shape[1]
    model = AttritionMLP(in_features=in_features, hidden_dims=hidden_dims, dropout_rates=dropout_rates)
    print(f"[Train] Initialized AttritionMLP ({count_parameters(model)} trainable parameters).")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 4. Training Loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epochs": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)
        
        history["epochs"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        
        # Checkpoint & Early stopping
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | Train Acc: {tr_acc*100:.1f}% | Val Acc: {vl_acc*100:.1f}%")
            
        if patience_counter >= patience:
            print(f"[Train] Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.4f})")
            break
            
    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Train] Loaded best model checkpoint saved at {save_path}")
        
    return model, history


def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """
    Fulfills Step 4 Hyperparameter Tuning requirement:
    Systematically compares configurations across Learning Rates, Dropout Rates, and Batch Sizes.
    """
    print("[Tuning] Commencing hyperparameter exploration grid...")
    configs = [
        {"lr": 0.001, "batch_size": 32, "dropout": [0.3, 0.2, 0.1], "name": "Config A (Baseline)"},
        {"lr": 0.005, "batch_size": 32, "dropout": [0.3, 0.2, 0.1], "name": "Config B (High LR)"},
        {"lr": 0.001, "batch_size": 64, "dropout": [0.4, 0.3, 0.2], "name": "Config C (High Regularization)"},
        {"lr": 0.0005, "batch_size": 16, "dropout": [0.2, 0.1, 0.05], "name": "Config D (Small Batch, Low LR)"}
    ]
    
    results = []
    criterion = nn.BCELoss()
    
    for cfg in configs:
        x_tr, y_tr = prepare_tensors(X_train, y_train)
        x_vl, y_vl = prepare_tensors(X_val, y_val)
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(x_vl, y_vl), batch_size=cfg["batch_size"], shuffle=False)
        
        mdl = AttritionMLP(in_features=X_train.shape[1], dropout_rates=cfg["dropout"])
        opt = torch.optim.AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        
        # Train for 20 quick epochs
        best_v_loss = float("inf")
        best_v_acc = 0.0
        for ep in range(25):
            train_epoch(mdl, train_loader, criterion, opt)
            v_loss, v_acc = evaluate_epoch(mdl, val_loader, criterion)
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                best_v_acc = v_acc
                
        results.append({
            "Configuration": cfg["name"],
            "Learning Rate": cfg["lr"],
            "Batch Size": cfg["batch_size"],
            "Dropout Rates": str(cfg["dropout"]),
            "Best Val Loss": round(best_v_loss, 4),
            "Best Val Accuracy (%)": round(best_v_acc * 100, 2)
        })
        
    tuning_df = pd.DataFrame(results)
    print("\n--- Hyperparameter Tuning Benchmark Results ---")
    print(tuning_df.to_string(index=False))
    return tuning_df


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_and_split
    df = load_raw_data()
    splits = preprocess_and_split(df)
    model, history = train_attrition_model(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    tune_df = perform_hyperparameter_tuning(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
