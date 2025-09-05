import pandas as pd
from sklearn.model_selection import train_test_split
import os

IN_PATH  = "data/processed/news.csv"
OUT_DIR  = "data/processed"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_PATH)

    assert {"text", "label"}.issubset(df.columns), "news.csv must have text, label columnns"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("Saved:", os.path.join(OUT_DIR,"train.csv"), "and", os.path.join(OUT_DIR,"test.csv"))

if __name__ == "__main__":
    main()
