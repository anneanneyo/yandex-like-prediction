import argparse
import pandas as pd
import lightgbm as lgb
from datetime import datetime

def load_model():
    return lgb.Booster(model_file="data/model_lgb.txt")

def make_features(df):
    ts = df.get("ts", datetime.utcnow().timestamp())
    df["hour"] = pd.to_datetime(df["ts"], unit="s").dt.hour if "ts" in df else datetime.utcnow().hour
    df["dow"] = pd.to_datetime(df["ts"], unit="s").dt.weekday if "ts" in df else datetime.utcnow().weekday()
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["user_prior_like_rate"] = 0.15
    df["item_prior_like_rate"] = 0.12
    return df[["uid", "item_id", "hour", "dow", "is_weekend", "user_prior_like_rate", "item_prior_like_rate"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", type=int, help="user id")
    ap.add_argument("--item", type=int, help="item id")
    ap.add_argument("--ts", type=float, help="timestamp (optional)")
    ap.add_argument("--input", help="path to csv with uid,item_id")
    ap.add_argument("--output", help="path to save csv with preds")
    args = ap.parse_args()

    model = load_model()

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = pd.DataFrame([dict(uid=args.uid, item_id=args.item, ts=args.ts)])

    feats = make_features(df)
    df["pred"] = model.predict(feats)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"✅ Predictions saved to {args.output}")
    else:
        for _, r in df.iterrows():
            print(f"uid={int(r.uid)}, item={int(r.item_id)} → like_prob={r.pred:.3f}")

if __name__ == "__main__":
    main()
