# data_utils.py
"""
Data Preprocessing Utilities for Next-Word Prediction (Penn Treebank)
---------------------------------------------------------------------
This module handles:
1. Loading and tokenizing text files
2. Building a vocabulary (word ↔ index mappings)
3. Preparing padded input-output pairs for training
"""

import numpy as np
from pathlib import Path
from nltk.tokenize import word_tokenize



def load_and_tokenize(file_path):
    """
    Reads a text file and splits it into tokenized sentences.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: A list of tokenized sentences (each sentence = list of tokens).
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = word_tokenize(line.strip())
            if tokens:
                sentences.append(tokens)
    return sentences


def build_vocab(sentences):
    """
    Builds a mapping between words and numerical indices.

    Args:
        sentences (list): List of tokenized sentences.

    Returns:
        tuple: (word_to_index, index_to_word)
    """
    all_tokens = [token for sent in sentences for token in sent]
    vocab = set(all_tokens)
    word_to_index = {"<PAD>": 0, "<UNK>": 1}

    for idx, word in enumerate(sorted(vocab), start=2):
        word_to_index[word] = idx

    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word


def prepare_data(sentences, word_to_index, max_len=None):
    """
    Converts tokenized sentences into padded numerical sequences.

    Args:
        sentences (list): Tokenized sentences.
        word_to_index (dict): Word to index mapping.
        max_len (int, optional): Maximum sequence length. Computed if None.

    Returns:
        tuple: (inputs, outputs) as NumPy arrays.
    """
    if max_len is None:
        max_len = max(len(s) for s in sentences)

    inputs, outputs = [], []
    for sentence in sentences:
        indices = [word_to_index.get(token, 1) for token in sentence]  # 1 = <UNK>
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [0] * (max_len - len(indices))  # padding with <PAD>

        inputs.append(indices[:-1])
        outputs.append(indices[1:])

    return np.array(inputs), np.array(outputs)
