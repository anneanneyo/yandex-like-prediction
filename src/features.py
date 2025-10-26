#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def prior_rate(g, ycol, tcol):
    g = g.sort_values(tcol)
    n = np.arange(len(g), dtype=np.int64)
    s = g[ycol].to_numpy().astype(np.int64)
    prev_sum = np.concatenate(([0], np.cumsum(s)[:-1]))
    prev_cnt = np.maximum(n, 1)
    return pd.Series(prev_sum / prev_cnt, index=g.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)[["uid","item_id","listen_ts","label"]].copy()

    df["hour"] = (df["listen_ts"] // 3600) % 24
    df["day"] = df["listen_ts"] // 86400
    df["dow"] = df["day"] % 7
    df["is_weekend"] = df["dow"].isin([5,6]).astype("int8")

    df["user_prior_like_rate"] = df.groupby("uid", group_keys=False).apply(
        prior_rate, ycol="label", tcol="listen_ts"
    ).astype("float32")

    df["item_prior_like_rate"] = df.groupby("item_id", group_keys=False).apply(
        prior_rate, ycol="label", tcol="listen_ts"
    ).astype("float32")

    out = df.drop(columns=["day"])
    out.to_parquet(args.out, index=False)
    print(f"saved → {args.out}; rows={len(out)}")

if __name__ == "__main__":
    main()
