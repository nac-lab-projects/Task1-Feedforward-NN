"""
model.py — Feedforward Neural Network Architecture for Next-Word Prediction

This script defines a simple Feedforward Neural Network (FNN) for next-word prediction tasks
using the Penn Treebank dataset.

Architecture:
    - Input Layer
    - Two Hidden Layers with ReLU activation
    - Output Layer with raw logits (Softmax applied during training/inference)

The model learns relationships between consecutive words to predict the next one.
"""

import torch
import torch.nn as nn


class FeedforwardNN(nn.Module):
    """
    Simple Feedforward Neural Network for next-word prediction.

    Args:
        input_size (int): Size of the input layer (vocabulary embedding dimension).
        hidden_size1 (int): Number of units in the first hidden layer.
        hidden_size2 (int): Number of units in the second hidden layer.
        output_size (int): Size of the output layer (number of vocabulary words).

    Architecture:
        Input → Linear → ReLU → Linear → ReLU → Linear → Output
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        """Defines forward propagation of data through the network."""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
