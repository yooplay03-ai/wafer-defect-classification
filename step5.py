import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

KEY = "X_max"
d = np.load("data/wafer32.npz", allow_pickle=True)
X, y, lot = d[KEY], d["y"], d["lot"]
classes = list(d["classes"])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = next(sgkf.split(X, y, groups=lot))

def to_tensor(a):
    onehot = np.eye(3, dtype=np.float32)[a]
    return torch.from_numpy(onehot).permute(0, 3, 1, 2)

X_tr, X_te = to_tensor(X[tr]), to_tensor(X[te])
y_tr = torch.tensor(y[tr])

torch.manual_seed(0)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
    nn.Linear(64, len(classes)),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(10):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    prob = torch.softmax(model(X_te), dim=1).numpy()
pred = prob.argmax(axis=1)
true = y[te]
X_test = X[te]

S, L = classes.index("Scratch"), classes.index("Loc")
groups = {
    "Scratch correct": np.where((true == S) & (pred == S))[0],
    "Scratch -> Loc": np.where((true == S) & (pred == L))[0],
    "Loc -> Scratch": np.where((true == L) & (pred == S))[0],
    "Loc correct": np.where((true == L) & (pred == L))[0],
}
for name, idx in groups.items():
    print(f"{name}: {len(idx)}장")

fig, axes = plt.subplots(4, 6, figsize=(12, 9))
for row, (name, idx) in enumerate(groups.items()):
    for col in range(6):
        ax = axes[row, col]
        ax.axis("off")
        if col < len(idx):
            i = idx[col]
            ax.imshow(X_test[i])
            title = f"{name}\n{prob[i, pred[i]]:.2f}" if col == 0 else f"{prob[i, pred[i]]:.2f}"
            ax.set_title(title, fontsize=8)
plt.tight_layout()
plt.savefig("errors_scratch_loc.png")