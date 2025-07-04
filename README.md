# 🌸 Iris Flower Classification (PyTorch)

This project implements a simple feedforward neural network using **PyTorch** to classify Iris flowers into 3 species based on 4 features.

---

## 🚀 How to Run

1. **Install required libraries:**

```bash
pip install torch scikit-learn matplotlib
```

2. **Run the script** (or execute the notebook cells):

Make sure to load and preprocess the Iris dataset before training the model.

---

## 🧠 Model Architecture

- **Input Layer**: 4 features (sepal & petal length/width)
- **Hidden Layers**: Two fully connected layers (128 & 64 neurons)
- **Activation**: ReLU
- **Output Layer**: 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

---

## 🧾 Data Preprocessing

- Data is loaded using `sklearn.datasets.load_iris()`.
- Features (`X`) are converted to `torch.float32`.
- Labels (`y`) are converted to `torch.long` for classification:

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

- Data is split into **training** and **testing** sets using `train_test_split`.

---

## 📊 Output Example

Each epoch prints:

```
Epoch [1/100], Loss: 1.094, Train Accuracy: 0.733, Test Accuracy: 0.700
```

A plot shows **Training vs. Test Accuracy** after training.

---

## 💾 Saving & Loading the Model

To save:
```python
torch.save(model.state_dict(), 'iris_model.pth')
```

To load:
```python
model.load_state_dict(torch.load('iris_model.pth'))
model.eval()
```

---

## ✅ Requirements

- Python 3.x
- torch
- scikit-learn
- matplotlib

---

## 📂 Files

- `main.py` or Jupyter notebook – Training & evaluation
- `iris_model.pth` – Saved trained model (optional)
