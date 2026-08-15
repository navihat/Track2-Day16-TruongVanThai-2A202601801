import json
import os
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_result.json")


def main():
    results = {}

    print(f"[1/5] Loading dataset from {DATA_PATH} ...")
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    load_time = time.perf_counter() - t0
    X = df.drop(columns=["Class"])
    y = df["Class"]
    results["num_rows"] = len(df)
    results["num_features"] = X.shape[1]
    results["load_data_time_sec"] = round(load_time, 4)
    print(f"    Loaded {len(df)} rows, {X.shape[1]} features in {load_time:.2f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[2/5] Training LGBMClassifier ...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    train_time = time.perf_counter() - t0
    results["train_time_sec"] = round(train_time, 4)
    results["best_iteration"] = int(model.best_iteration_)
    print(f"    Training done in {train_time:.2f}s, best_iteration={model.best_iteration_}")

    print("[3/5] Evaluating on test set ...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    results["auc_roc"] = round(roc_auc_score(y_test, y_pred_proba), 6)
    results["accuracy"] = round(accuracy_score(y_test, y_pred), 6)
    results["f1_score"] = round(f1_score(y_test, y_pred), 6)
    results["precision"] = round(precision_score(y_test, y_pred), 6)
    results["recall"] = round(recall_score(y_test, y_pred), 6)
    print(
        f"    AUC-ROC={results['auc_roc']:.4f} Accuracy={results['accuracy']:.4f} "
        f"F1={results['f1_score']:.4f} Precision={results['precision']:.4f} "
        f"Recall={results['recall']:.4f}"
    )

    print("[4/5] Measuring inference latency (single row, avg over 100 runs) ...")
    single_row = X_test.iloc[[0]]
    for _ in range(5):  # warm-up
        model.predict(single_row)
    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(single_row)
    avg_latency_ms = (time.perf_counter() - t0) / n_runs * 1000
    results["inference_latency_1row_ms"] = round(avg_latency_ms, 4)
    print(f"    Avg latency: {avg_latency_ms:.4f} ms/row")

    print("[5/5] Measuring inference throughput (batch of 1000 rows) ...")
    batch = X_test.iloc[:1000]
    t0 = time.perf_counter()
    model.predict(batch)
    batch_time = time.perf_counter() - t0
    throughput = len(batch) / batch_time
    results["inference_batch_size"] = len(batch)
    results["inference_batch_time_sec"] = round(batch_time, 4)
    results["inference_throughput_rows_per_sec"] = round(throughput, 2)
    print(f"    Throughput: {throughput:.2f} rows/sec ({len(batch)} rows in {batch_time:.4f}s)")

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
