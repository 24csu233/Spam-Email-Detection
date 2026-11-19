
from datasets import load_dataset
import pandas as pd
from pathlib import Path

print("Downloading SpamAssassin dataset...")

url = "https://huggingface.co/datasets/talby/spamassassin/resolve/refs%2Fconvert%2Fparquet/text/train/0000.parquet"

dataset = load_dataset(
    "parquet",
    data_files=url,
    split="train"
)

df = dataset.to_pandas()

# Convert numeric labels into spam/ham if necessary
feature = dataset.features["label"]

if hasattr(feature, "int2str"):
    df["label"] = df["label"].map(feature.int2str)

df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(str).str.lower().str.strip()

# Keep only valid labels
df = df[df["label"].isin(["spam", "ham"])]

Path("data").mkdir(exist_ok=True)
df.to_csv("data/spam.csv", index=False)

print("\nDataset downloaded successfully!")
print("Total emails:", len(df))
print("\nEmail distribution:")
print(df["label"].value_counts())
