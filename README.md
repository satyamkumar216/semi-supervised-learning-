# SSL-MedBench: Benchmarking Semi-Supervised Learning in Medicine

**SSL-MedBench** is a benchmarking platform designed to rigorously evaluate and compare Semi-Supervised Learning (SSL) algorithms on clinical and medical datasets. It tests models across varying labeled fractions, tunes them via randomized cross-validation on labeled points, computes multi-dimensional evaluation metrics (F1-macro, AUC-ROC, Robustness, Efficiency), and exports production-ready artifacts.

Strict constraints observed:
- **No Neural Networks**: Pure machine learning algorithms (Random Forests, Support Vector Machines, Graph Transduction, Gaussian Mixtures).
- **Zero Data Leakage**: Standard scaling and tuning are performed strictly on training partitions per fold.
- **Fair Tuning**: All models are automatically tuned on identical partitions with an identical search budget.

---

## Project Structure

```
ssl_medbench/
├── backend/
│   ├── main.py            # FastAPI app: REST routes and background worker
│   ├── config.py          # Central constants: seeds, fractions, weights, budgets
│   ├── schemas.py         # Pydantic request and response models
│   ├── data_utils.py      # CSV loading, health report, preprocessing, label masking
│   ├── tuner.py           # Stratified hyperparameter search with SSL wrapper
│   ├── scorer.py          # Metrics computation and composite final score
│   ├── pipeline.py        # 5-fold outer cross-validation and evaluation protocol
│   ├── exporter.py        # Retraining winner model (.pkl) & enriched CSV generation
│   ├── models/
│   │   ├── __init__.py    # MODEL_REGISTRY mapping
│   │   ├── base.py        # BaseSSLModel (sklearn BaseEstimator + ClassifierMixin)
│   │   ├── team1_self_training.py       # Self-Training with Random Forest
│   │   ├── team2_label_propagation.py   # RBF Graph Label Propagation from scratch
│   │   ├── team3_co_training.py         # Two-view Co-Training SVM
│   │   ├── team4_s3vm.py                # Transductive SVM (TSVM / S3VM)
│   │   └── team5_em_gmm.py              # Semi-Supervised EM + GMM
│   └── outputs/           # Plots, reports, models, and exported datasets
├── frontend/
│   ├── index.html         # 4-section responsive benchmark interface
│   ├── style.css          # Dark-friendly theme with CSS variables
│   └── app.js             # Vanilla JS state controller and polling engine
├── tests/
│   └── test_smoke.py      # Smoke tests covering models and pipeline components
├── requirements.txt
└── README.md
```

---

## Setup & Quickstart

### 1. Install Dependencies

Ensure Python 3.9+ is installed, then install requirements:

```bash
cd ssl_medbench
pip install -r requirements.txt
```

### 2. Run the Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn backend.main:app --reload --port 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## Benchmarking Protocol

1. **Upload Dataset**: Upload any medical CSV dataset (e.g. Breast Cancer, Heart Disease, Diabetes). The system displays a data health summary and automatically suggests the target column.
2. **Stratified Label Masking**: Simulates realistic low-resource clinical settings by masking training labels to **5%, 10%, 20%, and 30%** (unlabeled = `-1`).
3. **Hyperparameter Auto-Tuning**: Performs `RandomizedSearchCV` (5 iterations per fold, model, and label fraction) using labeled points while supplying the unlabeled pool to the SSL wrapper.
4. **5-Fold Cross-Validation**: Evaluates both default and tuned variants on held-out test folds with zero data leakage.
5. **Multi-Criterion Scoring**:
   $$\text{Score} = 0.4 \times \text{F1} + 0.3 \times \text{AUC} + 0.2 \times \text{Robustness} + 0.1 \times \text{Efficiency}$$
   - **Robustness**: $1 - (\text{F1}_{30\%} - \text{F1}_{5\%})$, penalizing models that degrade steeply with few labels.
   - **Efficiency**: $\text{F1} / \text{training\_time}$, min-max normalized across candidates.
6. **Artifact Export**:
   - **Mode A**: Save the winning model (`winner_model.pkl`) with its fitted scaler.
   - **Mode B**: Generate an enriched dataset (`enriched_dataset.csv`) pseudo-labeling confident instances ($> 80\%$).

---

## How to Add a New Model (3 Steps)

1. **Create the Model File** (`backend/models/teamX_your_model.py`):
   Subclass `BaseSSLModel` and implement:
   - `__init__(self, **hyperparams)`: Store each hyperparameter as an attribute.
   - `fit(self, X, y)`: Train your algorithm where `y == -1` represents unlabeled data.
   - `predict(self, X)` & `predict_proba(self, X)`: Classification outputs.
   - `get_name() -> str`, `get_param_grid() -> dict`, `get_convergence_trace() -> list`.

2. **Register the Model** (`backend/models/__init__.py`):
   Import your model and add it to `MODEL_REGISTRY`:
   ```python
   from .teamX_your_model import YourModel

   MODEL_REGISTRY["Your Model Name"] = YourModel
   ```

3. **Verify Compatibility**:
   Add a smoke test in `tests/test_smoke.py` and verify using:
   ```bash
   pytest tests/test_smoke.py
   ```

---

## Running Tests

Run the unit and smoke test suite:
```bash
pytest tests/
```
