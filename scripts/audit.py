#!/usr/bin/env python3
from datasets import load_dataset
import pandas as pd
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="flat-multievent-50m")
    p.add_argument("--split", default="train[:100000]")
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"loading yandex/yambda ({args.config}, {args.split})")
    ds = load_dataset("yandex/yambda", args.config, split=args.split)
    df = ds.to_pandas()

    print(f"\nrows: {len(df)}")
    print("\ncolumns:", df.columns.tolist())
    print("\nfeatures:", ds.features)

    if "event_type" in df.columns:
        print("\nevent_type counts:")
        print(df["event_type"].value_counts().head(10))

    cat = [c for c in df.columns if df[c].dtype == "object" and c != "event_type"]
    for c in cat[:3]:
        print(f"\n{c} top:")
        print(df[c].value_counts().head(5))

    num = df.select_dtypes("number").columns.tolist()
    if num:
        print("\nnumeric describe:")
        print(df[num].describe().T.head(5))

    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"\nsaved parquet → {args.out}")

if __name__ == "__main__":
    main()