"""Train a BLS model and save it.   Example:  python train.py --enh-nodes 1000 --groups 10"""
import argparse
from src.config import BLSConfig, DataConfig, MODEL_DIR
from src.config import PROCESSED_DIR
from src.database import Database
from src.preprocessing import export_sample_images
from src.training import PreparedData, train_bls, save_pipeline, store_run_in_db


def main():
    p = argparse.ArgumentParser(description="Train the Broad Learning System image classifier")
    p.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "image_folder"])
    p.add_argument("--image-folder", default=None)
    p.add_argument("--n-train", type=int, default=20000); p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--features", default="pixels", choices=["pixels", "hog"])
    p.add_argument("--groups", type=int, default=10, help="feature-mapping windows")
    p.add_argument("--nodes-per-group", type=int, default=10)
    p.add_argument("--enh-nodes", type=int, default=1000)
    p.add_argument("--reg", type=float, default=2 ** -30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sparse", action="store_true", help="skip sparse-autoencoder refinement of mapping weights")
    p.add_argument("--name", default="bls_model")
    p.add_argument("--no-db", action="store_true", help="do not log this run to the SQL database")
    a = p.parse_args()

    dcfg = DataConfig(dataset=a.dataset, image_folder=a.image_folder, n_train=a.n_train, n_test=a.n_test,
                      feature_method=a.features, seed=a.seed)
    print("Loading and preprocessing data ...")
    data = PreparedData(dcfg)
    print(f"  train {data.x_train.shape}, test {data.x_test.shape}, classes {len(data.encoder.classes_)}")
    bcfg = BLSConfig(n_feature_groups=a.groups, nodes_per_group=a.nodes_per_group,
                     n_enhancement_nodes=a.enh_nodes, reg=a.reg, seed=a.seed)
    print(f"Training BLS: {bcfg.n_feature_nodes} feature nodes + {bcfg.n_enhancement_nodes} enhancement nodes ...")
    model, m = train_bls(data, bcfg, use_sparse_mapping=not a.no_sparse)
    # Temperature-scale the pseudo-probabilities on a validation split that was NOT used for fitting.
    # This changes only the confidence values, never the predicted classes.
    if data.x_val is not None:
        t = model.calibrate_temperature(data.x_val, data.y_val)
        print(f"  confidence temperature T={t:.3f} (fitted on {len(data.y_val)} held-out validation images)")
    else:
        print("  no validation split available -> confidence left uncalibrated (T=1)")
    model_path = model.save_model(MODEL_DIR / f"{a.name}.npz"); pipe_path = save_pipeline(data)
    print(f"  accuracy={m['accuracy']:.4f}  f1={m['f1_score']:.4f}  train_time={m['training_time']:.2f}s "
          f"pred_time={m['prediction_time']:.3f}s")
    print(f"Saved model -> {model_path}\nSaved preprocessing -> {pipe_path}")
    export_sample_images(data.dataset, PROCESSED_DIR / "samples")
    if not a.no_db:
        exp_id = store_run_in_db(Database(), data, model, m, bcfg)
        print(f"Logged to SQL database: experiment_id={exp_id} (+ image metadata and test predictions)")


if __name__ == "__main__":
    main()
