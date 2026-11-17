import sys
import numpy as np
import pandas as pd
import pandas.core.indexes
import matplotlib.pyplot as plt

sys.modules["pandas.indexes"] = pandas.core.indexes

df = pd.read_pickle("data/LSWMD.pkl")

def first(x):
    return x[0][0] if len(x) > 0 else ""

df["label"] = df["failureType"].apply(first)
df["split"] = df["trianTestLabel"].apply(first)

defects = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
df = df[df["label"].isin(defects)].reset_index(drop=True)
print("불량 패턴 웨이퍼:", len(df))

# 방법 1: 가장 가까운 칸 값 가져오기
def resize_nearest(m, size=32):
    h, w = m.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return m[np.ix_(rows, cols)]

# 방법 2: 구역 안에 불량이 있으면 불량으로 (가는 선이 안 끊기게)
def resize_max(m, size=32):
    h, w = m.shape
    r = np.linspace(0, h, size, endpoint=False).astype(int)
    c = np.linspace(0, w, size, endpoint=False).astype(int)
    m = np.maximum.reduceat(m, r, axis=0)
    return np.maximum.reduceat(m, c, axis=1)

X_nearest = np.stack([resize_nearest(m) for m in df["waferMap"]]).astype(np.uint8)
X_max = np.stack([resize_max(m) for m in df["waferMap"]]).astype(np.uint8)
y = df["label"].map({c: i for i, c in enumerate(defects)}).values
lot = df["lotName"].astype(str).values
split = df["split"].astype(str).values

print("X:", X_max.shape, " y:", y.shape)
print("로트 수:", len(set(lot)))
print(pd.Series(split).value_counts())

np.savez_compressed("data/wafer32.npz", X_nearest=X_nearest, X_max=X_max,
                    y=y, lot=lot, split=split, classes=np.array(defects))
print("저장 완료")

# 줄여도 긁힘이 살아 있는지 확인
i = int(np.where(y == defects.index("Scratch"))[0][0])
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(df["waferMap"].iloc[i])
ax[0].set_title(f"original {df['waferMap'].iloc[i].shape}")
ax[1].imshow(X_nearest[i])
ax[1].set_title("nearest 32x32")
ax[2].imshow(X_max[i])
ax[2].set_title("max 32x32")
plt.tight_layout()
plt.savefig("resize_check.png")