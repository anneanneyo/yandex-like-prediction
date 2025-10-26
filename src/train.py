#!/usr/bin/env python3
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train_features_2m.parquet")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_parquet(args.src).sort_values("listen_ts")
    cut = int(len(df) * (1 - args.val_frac))
    tr, va = df.iloc[:cut], df.iloc[cut:]

    y_tr = tr["label"].astype(int)
    y_va = va["label"].astype(int)
    drop = [c for c in ["label", "listen_ts", "like_ts"] if c in df.columns]
    X_tr = tr.drop(columns=drop)
    X_va = va.drop(columns=drop)

    pos = y_tr.mean()
    spw = (1 - pos) / max(pos, 1e-6)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.9,
        bagging_freq=1,
        metric="auc",
        verbose=-1,
        scale_pos_weight=spw,
    )

    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtr,
        num_boost_round=2000,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    p = model.predict(X_va, num_iteration=model.best_iteration)
    roc = roc_auc_score(y_va, p)
    pr = average_precision_score(y_va, p)

    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    model.save_model("data/model_lgb.txt", num_iteration=model.best_iteration)
    imp = pd.Series(model.feature_importance(), index=X_tr.columns).sort_values(ascending=False)
    imp.to_csv("reports/feature_importance.csv")
    print("saved model → data/model_lgb.txt")
    print("saved importances → reports/feature_importance.csv")

if __name__ == "__main__":
    main()
