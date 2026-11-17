import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

KEY = "X_max"   # 원래는 "X_nearest"

d = np.load("data/wafer32.npz", allow_pickle=True)
X, y, lot, split = d[KEY], d["y"], d["lot"], d["split"]
classes = list(d["classes"])

# 1. 원본 구분에서 같은 로트가 섞였나?
train_lots = set(lot[split == "Training"])
test_lots = set(lot[split == "Test"])
print("원본 Training과 Test에 모두 있는 로트:", len(train_lots & test_lots), "/", len(set(lot)))

# 2. 로트 단위로 직접 나누기
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = next(sgkf.split(X, y, groups=lot))
print("공부용:", len(tr), " 시험용:", len(te), " 겹치는 로트:", len(set(lot[tr]) & set(lot[te])))

# 3. 0/1/2 지도를 3채널로 바꾸기
def to_tensor(a):
    onehot = np.eye(3, dtype=np.float32)[a]
    return torch.from_numpy(onehot).permute(0, 3, 1, 2)

X_tr, X_te = to_tensor(X[tr]), to_tensor(X[te])
y_tr = torch.tensor(y[tr])
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)

# 4. CNN 만들기
torch.manual_seed(42)
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

# 5. 학습
for epoch in range(10):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    print(f"epoch {epoch + 1}: 오차 {total / len(X_tr):.4f}")

# 6. 평가
model.eval()
with torch.no_grad():
    pred = model(X_te).argmax(dim=1).numpy()

acc = accuracy_score(y[te], pred)
macro = f1_score(y[te], pred, average="macro")
print(f"\n정확도 {acc:.3f} | macro F1 {macro:.3f}")
print(classification_report(y[te], pred, target_names=classes, digits=3))
print(confusion_matrix(y[te], pred))

mlflow.set_experiment("wafer-defect")
with mlflow.start_run(run_name=f"cnn_baseline_{KEY}"):
    mlflow.log_param("resize", KEY)
    mlflow.log_param("split", "StratifiedGroupKFold by lot")
    mlflow.log_param("class_weight", "none")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_f1", macro)