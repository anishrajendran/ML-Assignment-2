
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import shutil
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef,
)

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────
st.set_page_config(page_title="Wine Quality Prediction", layout="wide")

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# ───────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────
def get_model_files():
    """Return sorted list of .joblib files in MODEL_DIR."""
    if not os.path.exists(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib'))


def friendly_name(filename: str) -> str:
    """'logistic_regression.joblib' → 'Logistic Regression'"""
    return filename.replace('.joblib', '').replace('_', ' ').title()


def load_bundle(filename: str):
    """Load a bundled model dict from MODEL_DIR.

    Each .joblib is expected to be a dict with keys:
        model, scaler, le_wine, le_target
    """
    path = os.path.join(MODEL_DIR, filename)
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and 'model' in bundle:
        return bundle
    # Fallback: if someone uploaded a raw estimator (not a dict)
    return None


def handle_zip_upload(uploaded_zip):
    """Extract .joblib files from a ZIP into MODEL_DIR."""
    extracted = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, uploaded_zip.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith('/') or not member.lower().endswith('.joblib'):
                    continue
                basename = os.path.basename(member)
                if not basename:
                    continue
                extracted_path = zf.extract(member, tmpdir)
                shutil.move(extracted_path, os.path.join(MODEL_DIR, basename))
                extracted.append(basename)
    return extracted


# ───────────────────────────────────────────────
# Navigation
# ───────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction Page", "Admin Page"])


# ═══════════════════════════════════════════════
# ADMIN PAGE
# ═══════════════════════════════════════════════
if page == "Admin Page":
    st.title("Admin Page – Model Management")

    st.write("### Upload Model Artifacts")
    st.info(
        "Upload a **ZIP** file containing `.joblib` model files.  \n"
        "Each `.joblib` should be a bundled dict with keys: "
        "`model`, `scaler`, `le_wine`, `le_target`.  \n"
        "The file name (without extension) becomes the model name in the dropdown."
    )

    uploaded_zip = st.file_uploader("Upload ZIP of .joblib files", type=['zip'])

    if uploaded_zip is not None:
        with st.spinner("Extracting models …"):
            extracted = handle_zip_upload(uploaded_zip)
        if extracted:
            st.success(f"Extracted **{len(extracted)}** model file(s):")
            for name in extracted:
                st.write(f"- `{name}` → **{friendly_name(name)}**")
        else:
            st.warning("No `.joblib` files found inside the uploaded ZIP.")

    # Individual upload fallback
    with st.expander("Or upload individual .joblib files"):
        uploaded_files = st.file_uploader(
            "Select .joblib files", accept_multiple_files=True,
            type=['joblib'], key="individual_upload",
        )
        if uploaded_files:
            for uf in uploaded_files:
                with open(os.path.join(MODEL_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
                st.success(f"Saved `{uf.name}` → **{friendly_name(uf.name)}**")

    # List existing models
    st.write("---")
    st.write("### Existing Models")
    model_files = get_model_files()
    if model_files:
        for mf in model_files:
            st.text(f"• {friendly_name(mf)}  ({mf})")
    else:
        st.warning("No models found. Upload a ZIP or individual `.joblib` files above.")


# ═══════════════════════════════════════════════
# PREDICTION PAGE
# ═══════════════════════════════════════════════
elif page == "Prediction Page":
    st.title("Wine Quality Prediction")

    model_files = get_model_files()

    if not model_files:
        st.error("No models available. Go to the **Admin Page** to upload models.")
    else:
        # Build display-name → filename mapping
        name_map = {friendly_name(mf): mf for mf in model_files}
        selected_display = st.sidebar.selectbox("Select Model", list(name_map.keys()))
        selected_file = name_map[selected_display]

        st.write(f"### Selected Model: {selected_display}")
        st.caption(f"File: `{selected_file}`")

        # Load the bundled model
        bundle = load_bundle(selected_file)

        if bundle is None:
            st.error(
                f"`{selected_file}` is not a valid bundled model file.  \n"
                "Expected a dict with keys: `model`, `scaler`, `le_wine`, `le_target`.  \n"
                "Please re-export using `train_models.py`."
            )
        else:
            model     = bundle['model']
            scaler    = bundle['scaler']
            le_wine   = bundle['le_wine']
            le_target = bundle['le_target']

            # Dataset upload
            uploaded_dataset = st.file_uploader("Upload Dataset (CSV)", type=['csv'])

            if uploaded_dataset:
                df = pd.read_csv(uploaded_dataset)
                st.write("### Uploaded Dataset Preview")
                st.dataframe(df.head())

                # ── Preprocessing ───────────────
                if 'wine_type' in df.columns:
                    try:
                        unknown = set(df['wine_type']) - set(le_wine.classes_)
                        if unknown:
                            st.error(f"Unknown wine_type labels: {unknown}")
                        else:
                            df['wine_type'] = le_wine.transform(df['wine_type'])
                    except Exception as e:
                        st.error(f"Error encoding wine_type: {e}")

                potential_targets = ['quality', 'quality_category', 'quality_encoded']
                X_cols = [c for c in df.columns if c not in potential_targets]
                X = df[X_cols]

                # ── Predict & evaluate ──────────
                try:
                    X_scaled = scaler.transform(X)
                    y_pred = model.predict(X_scaled)
                    y_pred_labels = le_target.inverse_transform(y_pred)

                    df_result = df.copy()
                    df_result['Predicted_Quality_Encoded'] = y_pred
                    df_result['Predicted_Quality_Label'] = y_pred_labels

                    st.write("### Predictions")
                    st.dataframe(df_result.head())

                    # Ground truth (if available)
                    y_true = None
                    if 'quality_encoded' in df.columns:
                        y_true = df['quality_encoded']
                    elif 'quality_category' in df.columns:
                        y_true = le_target.transform(df['quality_category'])
                    elif 'quality' in df.columns:
                        bins = [2, 5, 6, 9]
                        labels = ['low', 'medium', 'high']
                        temp_cat = pd.cut(df['quality'], bins=bins,
                                          labels=labels, include_lowest=True)
                        y_true = le_target.transform(temp_cat)

                    if y_true is not None:
                        st.write("### Evaluation Metrics")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                        col2.metric("Precision (Weighted)",
                                    f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
                        col3.metric("Recall (Weighted)",
                                    f"{recall_score(y_true, y_pred, average='weighted'):.4f}")

                        col4, col5 = st.columns(2)
                        col4.metric("F1 Score (Weighted)",
                                    f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
                        col5.metric("MCC Score",
                                    f"{matthews_corrcoef(y_true, y_pred):.4f}")

                        # AUC
                        try:
                            y_proba = model.predict_proba(X_scaled)
                            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
                            st.metric("AUC Score (OVR)", f"{auc:.4f}")
                        except Exception:
                            st.warning("AUC Score skipped (model may not support predict_proba).")

                        st.write("### Classification Report")
                        st.text(classification_report(
                            y_true, y_pred, target_names=le_target.classes_
                        ))

                        st.write("### Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=le_target.classes_,
                                    yticklabels=le_target.classes_)
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during prediction/evaluation: {e}")
