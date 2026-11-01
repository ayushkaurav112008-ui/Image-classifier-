# Image Classifier Using Broad Learning System (BLS)

A complete Python + SQL image-classification project. The **Broad Learning System is implemented from scratch in NumPy** (`src/bls.py`) — no deep-learning framework and no pretrained network is used. Trained and evaluated on **Fashion-MNIST**; results, experiments and predictions are stored in **SQLite**; a **Streamlit** app classifies new images.

> All numbers in this README were produced by actually running the code in this repository (tables below are generated from the CSV files in `experiments/experiment_results/`). Training times are from one CPU machine and will differ on yours; accuracies are deterministic for a given seed.

## 1. Objectives
1. Build a modular preprocessing → feature → BLS → evaluation → SQL pipeline.
2. Implement feature-mapping nodes, enhancement nodes, the broad matrix `A = [Z | H]` and the ridge output weights explicitly.
3. Study how the number of nodes, regularisation and seed change accuracy and training time.
4. Compare BLS honestly with baseline classifiers on identical features.
5. Provide a reproducible SQL-backed workflow and a simple prediction UI.

## 2. Technologies
Python 3.10+, NumPy (BLS maths), Pandas, scikit-learn (metrics + baselines only), OpenCV (image I/O, resizing, optional HOG), Matplotlib, SQLite (`sqlite3`), Streamlit, pytest.

## 3. Dataset
**Fashion-MNIST** (Zalando Research): 28×28 grayscale images, 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). The four official `.gz` files are downloaded automatically from the dataset's GitHub repository into `data/raw/` on first run.

Default subset (configurable in `src/config.py` / CLI flags): **20,000 training** images and **5,000 test** images (random, seeded), plus **2,000 validation** images drawn from the *unused* part of the official training set, used only to calibrate confidence values. To use another dataset set `dataset="image_folder"` with a folder laid out as `root/<class_name>/*.png` (or `root/train/<class>/…` and `root/test/<class>/…`); nothing in `bls.py` changes (verified by `tests/test_project.py::test_image_folder_dataset_swap_end_to_end`).

## 4. BLS theory (viva-friendly)
A normal deep network learns *all* weights with back-propagation over many epochs. BLS instead builds a **wide** random feature layer and learns **only the last layer**, in one closed-form step.

Let `X` be the N×d matrix of flattened, standardised images and `Y` the N×C one-hot label matrix.

1. **Feature-mapping nodes.** For each of *g* windows (groups) make `Z_i = φ(X·W_i + b_i)`, giving `nodes_per_group` nodes per window. Each `W_i` starts random and is then refined by a **sparse autoencoder** (solved with ADMM/lasso, `sparse_autoencoder_weights`) so the weights are sparse and tuned to keep the information needed to reconstruct the input (this refinement improved accuracy by about 2 points in our ablation, section 13.4). Each node is rescaled to [0,1]. Stacking all windows gives `Z = [Z_1 | … | Z_g]`.
2. **Enhancement nodes.** Apply a random *orthogonal* matrix and a non-linearity to the mapped features: `H = tanh(s · [Z | 1] · W_h)`. This adds non-linear combinations of the features. `s` keeps `tanh` out of saturation.
3. **Broad feature matrix.** Concatenate side by side: **`A = [Z | H]`**. The network is "broad" because features are placed side by side instead of stacked in deep layers.
4. **Output weights.** We want `A·W ≈ Y`. The textbook answer is the pseudo-inverse `W = A⁺Y`. Because `A` can be ill-conditioned we use the regularised (ridge) equivalent
   `W = (AᵀA + λI)⁻¹ AᵀY` (primal form, used when N ≥ nodes) or `W = Aᵀ(AAᵀ + λI)⁻¹ Y` (dual form, when N < nodes). `numpy.linalg.solve` is used instead of an explicit inverse for numerical stability. Both forms are unit-tested against scikit-learn's `Ridge`.
5. **Prediction.** `Ŷ = A_new · W`; the predicted class is `argmax(Ŷ)`.
6. **Confidence.** BLS regresses to one-hot targets, so its scores are not probabilities. `predict_proba` returns `softmax(Ŷ / T)` where the temperature `T` is fitted on held-out validation images (temperature scaling). On the test set this gives mean confidence and accuracy that are close (see Results) — but treat it as an estimate, not a guarantee.

**Why it trains fast:** the only "learning" is one linear solve (plus a small ADMM step per window). There are no epochs and no gradients.

## 5. System architecture
```
 raw images ──► preprocessing.py ──► feature_extraction.py ──► bls.py (BroadLearningSystem)
 (IDX / folder)  load,resize,gray,     standardise (fit on       Z (mapping) → H (enhancement)
                 normalise,flatten     train only) [+HOG]        A=[Z|H] → W=(AᵀA+λI)⁻¹AᵀY
                                                                      │
        training.py  (train, sweeps, baselines) ◄─────────────────────┤
        evaluation.py (metrics, confusion matrix, plots)              │
        prediction.py (single-image inference) ◄──────────────────────┘
        database.py ⇄ SQLite (classes, images, experiments, predictions, views)
        train.py · evaluate.py · app.py (Streamlit)
```
Preprocessing, feature code, model, evaluation and database code are separate modules with no circular dependencies, so each can be tested alone (`tests/`).

## 6. Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Database setup
Nothing to do manually: `Database()` creates `database/bls_project.db` from `database/schema.sql` on first use (SQLite ships with Python). To inspect it: `sqlite3 database/bls_project.db` and e.g. `SELECT * FROM v_experiment_ranking;`.

## 8. Training
```bash
python train.py                                   # defaults: 10 groups x 10 nodes, 1000 enhancement nodes
python train.py --groups 20 --enh-nodes 2000 --n-train 60000 --n-test 10000
python train.py --features hog                    # HOG features instead of raw pixels
python train.py --no-sparse                       # skip sparse-autoencoder refinement
python train.py --dataset image_folder --image-folder path/to/data
```
Saves `models/saved_models/bls_model.npz` and `preprocessor.npz`, exports sample images to `data/processed/samples/`, and logs the run + 5,000 test predictions to SQL.

## 9. Evaluation
```bash
python evaluate.py                      # confusion matrix + classification report of the saved model
python evaluate.py --sweep              # node-count and regularisation experiments + graphs
python evaluate.py --baselines          # BLS vs Logistic Regression / Random Forest / SVM
python evaluate.py --ablation           # seed variance and sparse-mapping ablation
pytest -q tests                         # unit tests
```
Graphs go to `visualizations/`, tables to `experiments/experiment_results/`.

## 10. Prediction interface
```bash
streamlit run app.py
```
Pick a sample test image or upload one → see the preprocessed 28×28 input → **Classify** → predicted class, confidence and top-3 probabilities → the prediction is stored in the `predictions` table. Other tabs show the SQL experiment table, per-class accuracy computed in SQL, result plots and prediction history. *Uploaded photos that look unlike Fashion-MNIST (busy backgrounds, several objects, different framing) will often be misclassified; tick "Invert colours" for light-background images.*

## 11. SQL schema
| Table | Purpose | Key constraints |
|---|---|---|
| `classes(class_id, class_name)` | class lookup | PK `class_id`, `class_name` UNIQUE |
| `images(image_id, file_name, actual_class, dataset_type)` | every train/test/uploaded image | PK; FK `actual_class→classes`; `dataset_type` CHECK in (train,test,uploaded); UNIQUE(file_name, dataset_type) |
| `experiments(experiment_id, model_name, feature_nodes, enhancement_nodes, regularization, …, accuracy, precision, recall, f1_score, training_time, prediction_time, created_at)` | one row per training run (BLS or baseline) | metrics CHECK in [0,1], times ≥ 0 |
| `predictions(prediction_id, image_id, experiment_id, predicted_class, confidence, prediction_time)` | one row per prediction | FK→images (CASCADE), FK→experiments (SET NULL), FK→classes; confidence CHECK in [0,1] |

Indexes on `images(actual_class)`, `images(dataset_type)`, `predictions(image_id | experiment_id | predicted_class)` and `experiments(model_name, accuracy DESC)`. Views: `v_prediction_results` (joined names + `is_correct`), `v_class_accuracy` (per-class accuracy in SQL), `v_experiment_ranking` (window-function `RANK()`). All statements are parameterised; foreign keys are enforced with `PRAGMA foreign_keys=ON`; each operation runs in a transaction that rolls back on error. As a cross-check, the accuracy recomputed inside SQL from stored predictions equals the accuracy computed in Python.

## 12. Milestones
| # | Milestone | Where |
|---|---|---|
| 1 | Dataset selection & preprocessing | `src/preprocessing.py`, `src/config.py` |
| 2 | Feature extraction & BLS design | `src/feature_extraction.py`, `BroadLearningSystem._map_features/_enhance/broad_matrix` |
| 3 | BLS implementation & training | `src/bls.py`, `src/training.py`, `train.py` |
| 4 | Evaluation & parameter analysis | `src/evaluation.py`, `evaluate.py --sweep`, `--ablation` |
| 5 | SQL integration, baseline comparison, final results | `database/schema.sql`, `src/database.py`, `evaluate.py --baselines`, this README |

## 13. Results
Setup: 20,000 training / 5,000 test images, features = standardised raw pixels (784), macro-averaged precision/recall/F1. Default BLS = 10 windows × 10 = **100 feature nodes + 1000 enhancement nodes**, λ = 2⁻³⁰, seed 42.

### 13.1 Main model
| Accuracy | Precision | Recall | F1 | Train time (s) | Predict 5,000 imgs (s) |
|---|---|---|---|---|---|
| 0.8532 | 0.8511 | 0.8516 | 0.8508 | 2.53 | 0.207 |

Per-class report (`experiments/experiment_results/classification_report.txt`):
```
              precision    recall  f1-score   support

 T-shirt/top     0.8112    0.8178    0.8145       494
     Trouser     0.9919    0.9606    0.9760       508
    Pullover     0.7667    0.7412    0.7537       510
       Dress     0.8178    0.8818    0.8486       499
        Coat     0.7454    0.7909    0.7675       507
      Sandal     0.9556    0.9098    0.9322       521
       Shirt     0.6408    0.5772    0.6073       473
     Sneaker     0.8863    0.9426    0.9136       488
         Bag     0.9600    0.9674    0.9637       521
  Ankle boot     0.9347    0.9269    0.9308       479

    accuracy                         0.8532      5000
   macro avg     0.8511    0.8516    0.8508      5000
weighted avg     0.8526    0.8532    0.8524      5000
```
See `visualizations/confusion_matrix.png`. The weakest class is **Shirt**, which is mostly confused with T-shirt/top, Pullover and Coat (visually similar garments); Trouser, Bag and Ankle boot are the easiest.

### 13.2 Effect of the number of nodes (test accuracy, one seed)
| enhancement nodes | 50 feature nodes | 100 feature nodes | 200 feature nodes |
|---|---|---|---|
| 100 | 0.8024 | 0.8180 | 0.8270 |
| 300 | 0.8236 | 0.8354 | 0.8400 |
| 500 | 0.8362 | 0.8436 | 0.8446 |
| 1000 | 0.8488 | 0.8532 | 0.8578 |
| 2000 | 0.8534 | 0.8614 | 0.8624 |

Graphs: `visualizations/accuracy_vs_nodes.png`, `training_time_vs_nodes.png`, `config_comparison.png`. Accuracy rises with more nodes but with diminishing returns: the smallest model (150 nodes) reaches 0.8024 in 0.72 s, the largest (2200 nodes) 0.8624 in 6.31 s. Going from 1000 to 2000 enhancement nodes adds only about 0.6 points on average while training time roughly doubles.

### 13.3 Effect of regularisation (10×10 feature nodes, 1000 enhancement nodes)
| regularization | accuracy | f1_score | training_time |
|---|---|---|---|
| 1e-09 | 0.8532 | 0.8508 | 2.5023 |
| 1e-06 | 0.8506 | 0.8482 | 2.4845 |
| 1e-03 | 0.8234 | 0.8187 | 2.5289 |
| 1e-01 | 0.7930 | 0.7846 | 2.5220 |
| 1e+00 | 0.7738 | 0.7607 | 2.5190 |
| 1e+01 | 0.7444 | 0.7218 | 2.6042 |

In this setting (20,000 samples, 1,100 nodes, float64) the least-regularised solutions are the most accurate and larger λ progressively under-fits. The system is over-determined (N ≫ nodes), so little regularisation is needed; with fewer samples or many more nodes a larger λ would be expected to matter more (not tested here).

### 13.4 Ablation and seed variance (5 seeds each, default architecture)
| Mapping weights | Accuracy (mean) | Accuracy (std) | F1 (mean) | Train time (s) |
|---|---|---|---|---|
| plain random mapping weights | 0.8333 | 0.0030 | 0.8293 | 1.6343 |
| sparse-autoencoder refinement (default) | 0.8536 | 0.0013 | 0.8512 | 2.4676 |

Seed-to-seed standard deviation is about 0.13–0.30 accuracy points, so the ~2.0-point gain from the sparse-autoencoder refinement is well outside random variation, at the cost of a slightly longer training time. Node-sweep differences smaller than ~0.3 points should not be over-interpreted.

### 13.5 Baseline comparison (same features, same train/test split)
| Model | Accuracy | Precision | Recall | F1 | Train time (s) | Predict time (s) |
|---|---|---|---|---|---|---|
| BLS (10x10 F, 1000 E) | 0.8532 | 0.8511 | 0.8516 | 0.8508 | 2.53 | 0.207 |
| Logistic Regression | 0.8136 | 0.8115 | 0.8122 | 0.8116 | 47.94 | 0.030 |
| Random Forest | 0.8652 | 0.8628 | 0.8633 | 0.8619 | 23.29 | 0.169 |
| SVM (RBF) | 0.8814 | 0.8797 | 0.8800 | 0.8796 | 31.59 | 36.253 |

![model comparison](visualizations/model_comparison.png)

BLS (0.8532) is **more accurate than** Logistic Regression and **less accurate than** Random Forest, SVM (RBF) (best baseline: SVM (RBF), 0.8814). It is therefore *not* the most accurate model in this experiment. Its clear advantage is training cost: about 9× faster than Random Forest, 12× faster than the RBF-SVM and 19× faster than Logistic Regression (lbfgs, max_iter=1500) on this machine. BLS also predicts much faster than the SVM (0.21 s vs 36.3 s), but Logistic Regression predicts faster than BLS.

### 13.6 Confidence quality
Temperature T = 0.139 (fitted on 2,000 validation images). On the 5,000 test images the mean confidence is 0.872 against an accuracy of 0.853; the expected calibration error (10 bins) is 0.023, i.e. slightly over-confident. Predictions with confidence above 0.95 (2842 images) were correct 97.3% of the time, while predictions below 0.5 confidence (362 images) were correct only 43.6% of the time, so low confidence is a useful warning sign.

## 14. Limitations
* One dataset split; the node/regularisation sweeps use a single seed (differences below roughly 0.3 percentage points are within seed noise, see 13.4).
* Only a 20,000-image training subset is used by default (full 60,000 is supported via `--n-train 60000`; not benchmarked here).
* Baselines use fixed, reasonable hyper-parameters and were **not** tuned; a tuned SVM/RF or a CNN would likely score higher. The comparison shows how BLS behaves under these settings, not that any method is optimal.
* Confidence values are calibrated on validation data but are still estimates.
* Real-world photographs differ strongly from Fashion-MNIST, so uploaded images are outside the training distribution.

## 15. Future scope
Incremental BLS (adding nodes/samples without retraining from scratch), full 60k training and tuned baselines, cascaded/recurrent BLS variants, colour datasets (RGB is already supported by the preprocessing), a REST API, and cross-validation with confidence intervals.

## 16. Likely viva questions
* **What is learned in BLS?** Only the output weight matrix `W`; mapping and enhancement weights are random (mapping weights are lightly refined by a sparse autoencoder).
* **Why no back-propagation?** `A·W ≈ Y` is linear in `W`, so the optimum is a closed-form ridge solution.
* **What does λ do?** It shrinks weights and stabilises the solve when `AᵀA` is ill-conditioned; too large a λ under-fits (see 13.3).
* **Why enhancement nodes?** Mapping nodes are near-linear in the input; enhancement nodes add non-linearity.
* **Is BLS better than an SVM here?** Not in accuracy in our runs; it trains much faster (see 13.5).
* **How do you avoid data leakage?** The scaler is fitted on training data only; the test set is never used for fitting or calibration.

## 17. Project layout
```
BLS_Image_Classifier/
├── app.py  train.py  evaluate.py  requirements.txt  README.md
├── src/  config.py preprocessing.py feature_extraction.py bls.py training.py evaluation.py prediction.py database.py
├── database/schema.sql   (+ bls_project.db after first run)
├── data/raw  data/processed      models/saved_models      experiments/experiment_results      visualizations
└── tests/test_project.py
```
