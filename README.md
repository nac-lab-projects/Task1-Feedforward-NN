
# *Penn Treebank Feedforward Neural Network*

### *Task 1*
***Description:***
*Date: September 25, 2025*

***Assigned tasks included:***

*1. Developing a conceptual understanding of neural networks and the backpropagation algorithm.*
*2. Implementing a simple prototype feedforward neural network with:*

* * *1 input layer*

* * *2 hidden layers*

* * *1 output layer*

***Dataset:***
*Using the Penn Treebank dataset from Kaggle, with the data already pre-split for training, validation, and testing. The dataset is stored in the `ptbdataset` folder:*
*- `ptb.train.txt` → Training set*
*- `ptb.valid.txt` → Validation set*
*- `ptb.test.txt` → Test set*

***Repository Structure:***
plaintext
```

penn_treebank_project/
│
├── ptbdataset/
│   ├── ptb.test.txt
│   ├── ptb.train.txt
│   └── ptb.valid.txt
│
├── data_utils.py        # Preprocessing utilities (tokenization, vocab creation, padding)
├── train_utils.py       # Training functions for the neural network
├── test_utils.py        # Testing/evaluation functions
├── predict.py           # Functions for next-word prediction
├── model.py             # Definition of the feedforward neural network
├── Main.ipynb           # Jupyter notebook demonstrating the full pipeline
├── model.pth            # Saved trained model
├── requirements.txt     # Project dependencies
└── README.md            # Project description and instructions


```
***Usage:***

*1. Preprocess the data using `data_utils.py`.*

*2. Train the model via `train_utils.py` or `Main.ipynb`.*

*3. Evaluate the model using `test_utils.py`.*

*4. Perform next-word prediction using `predict.py`.*

---
