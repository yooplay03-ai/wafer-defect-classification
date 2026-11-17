import sys
import numpy as np
import pandas as pd
import pandas.core.indexes

sys.modules["pandas.indexes"] = pandas.core.indexes

df = pd.read_pickle("data/LSWMD.pkl")

def first(x):
    return x[0][0] if len(x) > 0 else ""

df["label"] = df["failureType"].apply(first)
defects = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
df = df[df["label"].isin(defects)].reset_index(drop=True)

def resize_max(m, size=64):
    h, w = m.shape
    r = np.linspace(0, h, size, endpoint=False).astype(int)
    c = np.linspace(0, w, size, endpoint=False).astype(int)
    m = np.maximum.reduceat(m, r, axis=0)
    return np.maximum.reduceat(m, c, axis=1)

X64 = np.stack([resize_max(m) for m in df["waferMap"]]).astype(np.uint8)
y = df["label"].map({c: i for i, c in enumerate(defects)}).values
lot = df["lotName"].to_numpy(dtype=str)

print("X64:", X64.shape)
np.savez_compressed("data/wafer64.npz", X_max=X64, y=y, lot=lot, classes=np.array(defects))
print("저장 완료")