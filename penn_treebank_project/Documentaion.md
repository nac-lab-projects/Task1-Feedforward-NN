

**Epochs** are one of the most important concepts in training neural networks.

---

### 🔹 Definition

An **epoch** means **one full pass through the entire training dataset**.

* If you have 10,000 training sentences,
* and you train for **1 epoch**, the model has seen each of those 10,000 sentences **once**.
* If you train for **10 epochs**, the model has seen the dataset **10 times**.

---

### 🔹 Why multiple epochs?

Neural networks usually need many passes over the data to learn patterns properly.

* **1 epoch** = model just starts learning (usually too weak).
* **Too many epochs** = risk of *overfitting* (memorizing the training set instead of generalizing).

---

### 🔹 Training loop structure

When you write:

```python
epochs = 10
for epoch in range(epochs):
    for inputs, targets in training_data:
        # forward, loss, backward, update
```

It means:

* Outer loop = **epochs** (how many times we cycle through the full dataset).
* Inner loop = **batches/samples** inside that epoch.

---

### 🔹 Analogy (to make it simple 🎒📘)

Imagine you’re studying for an exam:

* **Dataset** = the textbook.
* **1 epoch** = you read the entire book once.
* **Multiple epochs** = you reread the book multiple times to understand better.

✅ Epoch = one full pass through the entire training dataset.

✅ Iteration = one update step (one batch of data is passed forward and backward).

🔑 So the big picture is the same:

Inputs (text) → Neural Network (RNN/LSTM/Transformer) → Loss (next-word prediction error) → Optimizer (updates weights) → Training Loop (repeat)

---
