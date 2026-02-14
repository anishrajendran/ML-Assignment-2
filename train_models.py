
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

# ───────────────────────────────────────────────
# Output directory
# ───────────────────────────────────────────────
OUTPUT_DIR = 'model'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ───────────────────────────────────────────────
# Load & preprocess dataset
# ───────────────────────────────────────────────
try:
    wine_df = pd.read_csv('winequality.csv')
except FileNotFoundError:
    print("Error: winequality.csv not found.")
    exit()

# Encode wine_type (red/white → 0/1)
le_wine = LabelEncoder()
wine_df['wine_type'] = le_wine.fit_transform(wine_df['wine_type'])

# Bin quality into low / medium / high
bins = [2, 5, 6, 9]
labels = ['low', 'medium', 'high']
wine_df['quality_category'] = pd.cut(
    wine_df['quality'], bins=bins, labels=labels, include_lowest=True
)

# Encode target
le_target = LabelEncoder()
wine_df['quality_encoded'] = le_target.fit_transform(wine_df['quality_category'])

# Features & target
X = wine_df.drop(['quality', 'quality_category', 'quality_encoded'], axis=1)
y = wine_df['quality_encoded']

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ───────────────────────────────────────────────
# Models to train
# ───────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'KNN':                 KNeighborsClassifier(),
    'Naive Bayes':         GaussianNB(),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
}

# ───────────────────────────────────────────────
# Train, bundle, and save
# ───────────────────────────────────────────────
print("Training models and saving bundled artifacts …")

for name, estimator in models.items():
    print(f"  Training {name} …")
    estimator.fit(X_train_scaled, y_train)

    # ── Bundle everything into ONE dict ─────────
    bundle = {
        'model':     estimator,
        'scaler':    scaler,
        'le_wine':   le_wine,
        'le_target': le_target,
    }

    filename = name.lower().replace(' ', '_') + '.joblib'
    joblib.dump(bundle, os.path.join(OUTPUT_DIR, filename))

    # Quick evaluation
    y_pred = estimator.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, estimator.predict_proba(X_test_scaled), multi_class='ovr')
    except Exception:
        auc = 0.0
    print(f"    {name} — Accuracy: {acc:.4f}, AUC: {auc:.4f}")

print(f"\nDone — {len(models)} bundled model files saved to '{OUTPUT_DIR}/'.")
