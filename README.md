# 📚 Student Performance Classification (PyTorch)

This project uses a simple feedforward neural network built with **PyTorch** to classify student performance into 4 categories based on 33 features.

---

## 🚀 How to Run

1. **Install dependencies:**

```bash
pip install torch pandas scikit-learn matplotlib
```

2. **Run the script:**

```bash
python main.py
```

Make sure the CSV file (`student_data.csv`) is in the same folder.

---

## 🧠 Model Overview

- Input: 33 features per student
- Output: 4 performance classes
- Model: 2 hidden layers (128 and 64 neurons), ReLU activations
- Loss: CrossEntropyLoss
- Optimizer: Adam

---

## 📂 Files

- `main.py` – Main training script
- `student_data.csv` – CSV data file with features and `Performance` label
- `model.pth` – Trained model saved after training

---

## 💡 Notes

- Labels must be type `torch.long` for classification:
  ```python
  y = torch.tensor(y, dtype=torch.long)
  ```
- Accuracy is printed every epoch
- A plot of training vs. test accuracy is shown at the end

---

## 💾 Save & Load Model

To save:
```python
torch.save(model.state_dict(), 'model.pth')
```

To load:
```python
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## 📈 Output Example

```
Epoch [1/100], Loss: 1.386, Train Accuracy: 0.270, Test Accuracy: 0.250
...
```

A plot will show training and test accuracy over time.

---

## ✅ Requirements

- Python 3.x
- torch
- pandas
- scikit-learn
- matplotlib
