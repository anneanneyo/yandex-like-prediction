import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-days", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)[["uid","item_id","timestamp","event_type"]]
    listens = df[df.event_type=="listen"][["uid","item_id","timestamp"]].copy()
    likes   = df[df.event_type=="like"][["uid","item_id","timestamp"]].copy()

    listens["listen_id"] = np.arange(len(listens))
    m = listens.merge(likes, on=["uid","item_id"], how="left", suffixes=("_listen","_like"))

    dt  = m["timestamp_like"] - m["timestamp_listen"]
    win = args.window_days * 24 * 3600
    m["label"] = (dt>=0) & (dt<=win)

    out = m.groupby("listen_id", as_index=False).agg(
        uid=("uid","first"),
        item_id=("item_id","first"),
        listen_ts=("timestamp_listen","first"),
        like_ts=("timestamp_like","min"),
        label=("label","any"),
    )

    out.to_parquet(args.out, index=False)
    pos = int(out["label"].sum())
    print(f"✅ saved → {args.out} | rows={len(out):,} | positives={pos:,}")

if __name__ == "__main__":
    main()
