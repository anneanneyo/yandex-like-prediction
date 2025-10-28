# Yandex Music Like Prediction

## Description

This project predicts whether a user will "like" a track on Yandex.Music based on their listening history.  
It implements a complete machine learning pipeline with data processing, feature generation, model training, evaluation, and inference.

**Tech stack:** Python, Pandas, LightGBM, Scikit-learn, Matplotlib, HuggingFace Datasets

Main components:
- **scripts/audit.py** – dataset sanity checks and validation
- **scripts/build_dataset.py** – dataset preparation and feature table creation
- **src/features.py** – feature generation
- **src/train.py** – model training
- **src/predict.py** – prediction on unseen data
- **src/plot_feature_importance.py** – visualization of feature importance

---

## Flow

```mermaid
flowchart LR
    A[Load and prepare data] --> B[Feature Engineering]
    B --> C[Train LightGBM model]
    C --> D[Evaluate metrics]
    D --> E[Predict likes]
```

---

## Results

| Metric | Score |
|--------|--------|
| ROC-AUC | 0.8217 |
| PR-AUC | 0.1140 |

Model achieves stable performance and correctly ranks positive “like” interactions.  
Top features: user activity, track popularity, listening time.

---

## Instructions

### Setup

```bash
git clone https://github.com/yourusername/yandex-like-prediction.git
cd yandex-like-prediction
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Build dataset

```bash
python scripts/build_dataset.py
```

### Train model

```bash
python src/train.py
```

The trained model is saved to:
```
yandex_like_project/data/model_lgb.txt
```

### Predict

```bash
python src/predict.py
```

Predictions are stored in:
```
yandex_like_project/reports/preds.csv
```

---

## Repository structure

```
Yandex_Pet_Project/
└── yandex_like_project/
    ├── configs/
    ├── data/
    │   ├── audit_sample_2m.parquet
    │   ├── train_listen_like_2m.parquet
    │   ├── train_features_2m.parquet
    │   ├── sample_requests.csv
    │   └── model_lgb.txt
    ├── reports/
    │   ├── figures/
    │   │   └── feature_importance.png
    │   ├── feature_importance.csv
    │   └── preds.csv
    ├── scripts/
    │   ├── audit.py
    │   └── build_dataset.py
    ├── src/
    │   ├── features.py
    │   ├── train.py
    │   ├── predict.py
    │   └── plot_feature_importance.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

---

## Future improvements

- Add CatBoost for categorical features  
- Add user embeddings (Word2Vec or MF)  
- Use temporal cross-validation  
- Explore neural models for sequence-based recommendations

---

## Author

Anna Gubareva
