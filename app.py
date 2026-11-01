"""Streamlit interface:  streamlit run app.py"""
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.config import EXPERIMENT_DIR, PROCESSED_DIR, VIS_DIR
from src.database import Database
from src.prediction import Predictor
from src.preprocessing import load_image, preprocess_image

st.set_page_config(page_title="BLS Image Classifier", layout="wide")


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_resource
def get_db() -> Database:
    return Database()


st.title("Image Classifier Using Broad Learning System (BLS)")
st.caption("Fashion-MNIST · BLS implemented from scratch in NumPy · results stored in SQLite")

try:
    predictor, db = get_predictor(), get_db()
except FileNotFoundError as err:
    st.error(f"{err}"); st.stop()

cfg = predictor.model.config
with st.sidebar:
    st.header("Loaded model")
    st.write(f"Feature nodes: **{cfg.n_feature_nodes}** ({cfg.n_feature_groups} groups)")
    st.write(f"Enhancement nodes: **{cfg.n_enhancement_nodes}**")
    st.write(f"Regularisation λ: **{cfg.reg:.2e}**")
    st.write(f"Input size: **{predictor.data_cfg.image_size[0]}×{predictor.data_cfg.image_size[1]}**, "
             f"{'grayscale' if predictor.data_cfg.grayscale else 'RGB'}")

tab_pred, tab_exp, tab_res, tab_hist = st.tabs(["Predict", "Experiments (SQL)", "Result plots", "Prediction history"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    left, right = st.columns(2)
    with left:
        mode = st.radio("Image source", ["Sample test image", "Upload image"], horizontal=True)
        source, name, actual = None, None, None
        if mode == "Sample test image":
            samples = sorted((PROCESSED_DIR / "samples").glob("*.png"))
            if samples:
                choice = st.selectbox("Choose a sample", [p.name for p in samples])
                source, name = PROCESSED_DIR / "samples" / choice, f"sample_{choice}"
                stem = choice.rsplit("_", 1)[0].replace("-", "/").replace("_", " ")
                actual = predictor.class_names.index(stem) if stem in predictor.class_names else None
            else:
                st.info("No samples found. Run `python train.py` to export them.")
        else:
            up = st.file_uploader("PNG / JPG image", type=["png", "jpg", "jpeg", "bmp", "webp"])
            invert = st.checkbox("Invert colours", help="The model was trained on light objects on a dark background. "
                                 "Tick this for photos with a light background.")
            if up is not None:
                data = up.getvalue()
                source, name = data, f"upload_{up.name}"
        if source is not None:
            try:
                img = load_image(source, predictor.data_cfg.grayscale)
                if mode == "Upload image" and invert:
                    img = 255 - img
                st.image(img, caption="Original", width=220, clamp=True)
                small = preprocess_image(img, predictor.data_cfg)
                st.image(cv2.resize((small * 255).astype("uint8"), (224, 224), interpolation=cv2.INTER_NEAREST),
                         caption=f"Preprocessed ({predictor.data_cfg.image_size[0]}×{predictor.data_cfg.image_size[1]})", width=220)
            except ValueError as err:
                st.error(str(err)); source = None
    with right:
        st.subheader("Prediction")
        if source is not None and st.button("Classify and save to database", type="primary"):
            res = predictor.predict_image(img)
            exp_id = db.latest_bls_experiment_id()
            pid = db.store_single_prediction(name, res["class_index"], res["confidence"], exp_id, actual)
            st.session_state["last"] = (res, pid)
        if "last" in st.session_state:
            res, pid = st.session_state["last"]
            st.metric("Predicted class", res["class_name"], f"confidence {res['confidence']:.1%}", delta_color="off")
            st.bar_chart(pd.DataFrame(res["top_k"], columns=["class", "probability"]).set_index("class"))
            st.caption(f"Saved as prediction_id={pid}. Confidence is a softmax of the BLS output scores with a "
                       "temperature fitted on held-out validation images — an estimate, not a guaranteed probability.")
        st.info("The model was trained on 28×28 Fashion-MNIST images (single centred clothing item). "
                "Photos that look very different from that data will often be misclassified.")

# ------------------------------------------------------------------ Experiments
with tab_exp:
    exps = db.experiments()
    if exps.empty:
        st.info("No experiments logged yet.")
    else:
        st.subheader("All logged experiments (view v_experiment_ranking)")
        st.dataframe(exps, width="stretch", hide_index=True)
        last = db.latest_bls_experiment_id()
        st.subheader(f"Per-class accuracy of latest BLS run (experiment {last}, computed in SQL)")
        ca = db.class_accuracy(last)
        if not ca.empty:
            st.bar_chart(ca.set_index("class_name")["accuracy"])

# ------------------------------------------------------------------ Result plots
with tab_res:
    for title, fn in [("Confusion matrix", "confusion_matrix.png"), ("Accuracy vs nodes", "accuracy_vs_nodes.png"),
                      ("Training time vs nodes", "training_time_vs_nodes.png"),
                      ("Top BLS configurations", "config_comparison.png"), ("BLS vs baselines", "model_comparison.png")]:
        if (VIS_DIR / fn).exists():
            st.subheader(title); st.image(str(VIS_DIR / fn))
    bc = EXPERIMENT_DIR / "baseline_comparison.csv"
    if bc.exists():
        st.subheader("Baseline comparison table"); st.dataframe(pd.read_csv(bc).round(4), hide_index=True)
    if not any((VIS_DIR / f).exists() for f in ["confusion_matrix.png"]):
        st.info("Run `python evaluate.py --sweep --baselines` to generate result plots.")

# ------------------------------------------------------------------ History
with tab_hist:
    hist = db.recent_predictions(50)
    st.dataframe(hist, width="stretch", hide_index=True) if not hist.empty else st.info("Nothing stored yet.")
