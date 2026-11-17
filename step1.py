import sys
import pandas as pd
import pandas.core.indexes
import matplotlib.pyplot as plt

#옛날 pandas로 저장된 파일이라. 옛 이름을 지금 이름으로 연결
sys.modules["pandas.indexes"] = pandas.core.indexes

df=pd.read_pickle("data/LSWMD.pkl")
print("웨이퍼 수:", len(df))
print("칸:", df.columns.tolist())

def get_label(x):
    return x[0][0] if len(x) > 0 else "unlabeled"

df["label"]=df["failureType"].apply(get_label)
print(df["label"].value_counts())

df["size"] = df["waferMap"].apply(lambda m:m.shape)
print("가장 흔한 크기 : ")
print(df["size"].value_counts().head())

classes = sorted(c for c in df["label"].unique() if c != "unlabeled")
fig, axes = plt.subplots(1, len(classes), figsize=(18,3))
for ax, c in zip(axes, classes):
    ax.imshow(df[df["label"]==c]["waferMap"].iloc[0])
    ax.set_title(c)
    ax.axis("off")
plt.tight_layout()
plt.savefig("wafer_examples.png")
