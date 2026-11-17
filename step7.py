import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

KEY = "X_max"

# 1. 데이터 불러오기
d = np.load("data/wafer32.npz", allow_pickle=True)
X, y, lot = d[KEY], d["y"], d["lot"]
classes = list(d["classes"])

# 2. 로트 단위로 나누기 (같은 로트는 한쪽에만)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = next(sgkf.split(X, y, groups=lot))

# 3. 0/1/2 지도를 3채널로 바꾸기
def to_tensor(a):
    onehot = np.eye(3, dtype=np.float32)[a]
    return torch.from_numpy(onehot).permute(0, 3, 1, 2)

X_tr, X_te = to_tensor(X[tr]), to_tensor(X[te])
y_tr = torch.tensor(y[tr])

# 4. 증강: 무작위로 90도씩 돌리고, 절반 확률로 좌우 뒤집기
def augment(xb):
    k = int(torch.randint(0, 4, (1,)))
    xb = torch.rot90(xb, k, dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=(3,))
    return xb

# 5. CNN 만들기 (실험마다 새로)
def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
        nn.Linear(64, len(classes)),
    )

# 6. 학습하고 평가하기
def train_eval(use_aug, seed, epochs=25):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            if use_aug:
                xb = augment(xb)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(dim=1).numpy()
    macro = f1_score(y[te], pred, average="macro")
    per_class = f1_score(y[te], pred, average=None)
    return macro, per_class

# 7. plain과 aug를 둘 다 25 epoch로, 시드 3개씩
mlflow.set_experiment("wafer-defect")
results = {}
for use_aug in [False, True]:
    name = "aug" if use_aug else "plain"
    macros, per = [], []
    for seed in [0, 1, 2]:
        macro, per_class = train_eval(use_aug, seed, epochs=25)
        macros.append(macro)
        per.append(per_class)
        print(f"{name} 25ep seed {seed}: macro F1 {macro:.3f}")
        with mlflow.start_run(run_name=f"cnn_{name}_25ep_seed{seed}"):
            mlflow.log_param("resize", KEY)
            mlflow.log_param("augment", "rot90 + flip" if use_aug else "none")
            mlflow.log_param("epochs", 25)
            mlflow.log_param("seed", seed)
            mlflow.log_metric("macro_f1", macro)
            for c, f in zip(classes, per_class):
                mlflow.log_metric(f"f1_{c}", f)
    results[name] = (np.mean(macros), np.std(macros), np.mean(per, axis=0))

# 8. 결과 비교
print("\n[25 epoch]")
for name, (mean, std, _) in results.items():
    print(f"{name}: macro F1 {mean:.3f} ± {std:.3f}")

print("\n패턴별 평균 F1 (plain → aug)")
for c, a, b in zip(classes, results["plain"][2], results["aug"][2]):
    print(f"{c:10s} {a:.3f} → {b:.3f}")