# train_utils.py
"""
train_utils.py

This module contains:
1. PTBDataset: a PyTorch Dataset wrapper for Penn Treebank next-word prediction.
2. train_model: function to train and validate a PyTorch model.

Notes / Explanation:
- PTBDataset wraps input-output arrays into PyTorch Dataset objects for mini-batching.
- train_model handles both training and validation loops.
- CrossEntropyLoss is used because this is a multi-class next-word prediction task.
- Adam optimizer is used for faster convergence.
- Only the first next word is predicted for simplicity; you could expand to full sequence.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class PTBDataset(Dataset):
    """Custom dataset for Penn Treebank next-word prediction"""
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs, 
                batch_size=32, epochs=5, lr=0.001):
    """
    Train a PyTorch model with given training and validation data.

    Args:
        model: PyTorch nn.Module model
        train_inputs: numpy array of training inputs
        train_outputs: numpy array of training targets
        val_inputs: numpy array of validation inputs
        val_outputs: numpy array of validation targets
        batch_size: mini-batch size
        epochs: number of training epochs
        lr: learning rate

    Returns:
        None
    """
    train_loader = DataLoader(PTBDataset(train_inputs, train_outputs), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(PTBDataset(val_inputs, val_outputs), batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training loop
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            y_batch = y_batch[:,0]  # predict the first next word
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                outputs = model(x_val)
                y_val = y_val[:,0]
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
