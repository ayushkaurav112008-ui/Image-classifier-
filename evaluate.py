"""Evaluate the saved model, run parameter experiments and baseline comparison.
   python evaluate.py                # confusion matrix + report for saved model
   python evaluate.py --sweep        # BLS configuration experiments + graphs
   python evaluate.py --baselines    # BLS vs baseline ML models
   python evaluate.py --ablation     # seed variance + sparse-mapping ablation"""
import argparse
import json
import numpy as np
import pandas as pd
from src.bls import BroadLearningSystem
from src.config import BLSConfig, DataConfig, EXPERIMENT_DIR, MODEL_DIR, VIS_DIR
from src.evaluation import (get_classification_report, get_confusion_matrix, plot_confusion_matrix,
                            plot_config_comparison, plot_metric_vs_nodes, plot_model_comparison)
from src.database import Database
from src.training import PreparedData, run_baselines, run_experiments, store_experiment_rows, train_bls


def data_from_saved(n_test):
    meta = json.loads(str(np.load(MODEL_DIR / "preprocessor.npz")["meta"]))
    d = meta["data_cfg"]; d["image_size"] = tuple(d["image_size"]); d["n_test"] = n_test
    return PreparedData(DataConfig(**d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bls_model"); p.add_argument("--sweep", action="store_true")
    p.add_argument("--baselines", action="store_true"); p.add_argument("--ablation", action="store_true"); p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=5000); p.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = p.parse_args()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True); VIS_DIR.mkdir(parents=True, exist_ok=True)

    if a.sweep or a.baselines or a.ablation:
        data = PreparedData(DataConfig(n_train=a.n_train, n_test=a.n_test))
    else:
        model = BroadLearningSystem.load_model(MODEL_DIR / f"{a.model}.npz")
        data = data_from_saved(a.n_test)
        m = model.evaluate(data.x_test, data.y_test); pred = model.predict(data.x_test)
        print({k: round(v, 4) for k, v in m.items()})
        cm = get_confusion_matrix(data.y_test, pred, len(data.encoder.classes_))
        report = get_classification_report(data.y_test, pred, data.encoder.classes_); print(report)
        (EXPERIMENT_DIR / "classification_report.txt").write_text(report)
        plot_confusion_matrix(cm, data.encoder.classes_, VIS_DIR / "confusion_matrix.png", "BLS confusion matrix")
        print("Saved confusion_matrix.png and classification_report.txt")

    if a.ablation:
        rows = []
        for sparse in (True, False):                       # same architecture, 5 random seeds each
            for seed in range(1, 6):
                _, m = train_bls(data, BLSConfig(seed=seed), use_sparse_mapping=sparse)
                rows.append(dict(sparse_mapping=sparse, seed=seed, accuracy=m["accuracy"], f1_score=m["f1_score"],
                                 training_time=m["training_time"]))
        ab = pd.DataFrame(rows); ab.to_csv(EXPERIMENT_DIR / "ablation_seeds.csv", index=False)
        print(ab.groupby("sparse_mapping")[["accuracy", "f1_score", "training_time"]].agg(["mean", "std"]).round(4).to_string())

    if a.sweep:
        df = run_experiments(data, feature_groups=(5, 10, 20), nodes_per_group=(10,),
                             enhancement_nodes=(100, 300, 500, 1000, 2000), regs=(2 ** -30,), seeds=tuple(a.seeds))
        df.to_csv(EXPERIMENT_DIR / "node_sweep.csv", index=False)
        plot_metric_vs_nodes(df, "accuracy", VIS_DIR / "accuracy_vs_nodes.png", "Test accuracy", "Accuracy vs number of nodes")
        plot_metric_vs_nodes(df, "training_time", VIS_DIR / "training_time_vs_nodes.png", "Training time (s)", "Training time vs number of nodes")
        # regularisation sweep at a fixed architecture
        dfr = run_experiments(data, feature_groups=(10,), enhancement_nodes=(1000,),
                              regs=(1e-9, 1e-6, 1e-3, 1e-1, 1.0, 10.0), seeds=tuple(a.seeds))
        dfr.to_csv(EXPERIMENT_DIR / "reg_sweep.csv", index=False)
        store_experiment_rows(Database(), df, data); store_experiment_rows(Database(), dfr, data)
        allr = pd.concat([df, dfr], ignore_index=True); allr.to_csv(EXPERIMENT_DIR / "all_experiments.csv", index=False)
        plot_config_comparison(allr, VIS_DIR / "config_comparison.png")
        print(allr.sort_values("accuracy", ascending=False).head(8).to_string(index=False))

    if a.baselines:
        best, bm = train_bls(data, BLSConfig())
        rows = run_baselines(data)
        bls_row = pd.DataFrame([dict(model="BLS (10x10 F, 1000 E)", **{k: bm[k] for k in
                                ["accuracy", "precision", "recall", "f1_score", "training_time", "prediction_time"]})])
        cmp = pd.concat([bls_row, rows], ignore_index=True)
        store_experiment_rows(Database(), rows, data, model_col="model")   # baselines (BLS row is logged by train.py)
        cmp.to_csv(EXPERIMENT_DIR / "baseline_comparison.csv", index=False)
        plot_model_comparison(cmp, VIS_DIR / "model_comparison.png"); print(cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
