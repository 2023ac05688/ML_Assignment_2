import json
from pathlib import Path
import runpy

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent


def _run_training_script_if_available():
    train_script = ROOT / "model" / "train_models.py"
    if not train_script.exists():
        return False
    runpy.run_path(str(train_script), run_name="__main__")
    return True


def _ensure_artifacts_present():
    metadata_file = ROOT / "model_details.json"

    # First recovery attempt: metadata missing, try generating all artifacts.
    if not metadata_file.exists():
        _run_training_script_if_available()

    if not metadata_file.exists():
        raise FileNotFoundError(
            "model_details.json is missing and model/train_models.py could not recreate it."
        )

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    required_files = [ROOT / "test_data.csv", ROOT / "model_metrics.csv"]
    for rel_path in metadata.get("model_files", {}).values():
        required_files.append(ROOT / rel_path)

    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        # Second recovery attempt: some artifacts missing, retrain/regenerate.
        _run_training_script_if_available()
        missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]

    if missing:
        missing_display = ", ".join(missing)
        raise FileNotFoundError(
            "Required artifact files are missing: " + missing_display
        )

    return metadata


@st.cache_resource
def load_metadata_and_models():
    metadata = _ensure_artifacts_present()
    models = {}
    for model_name, rel_path in metadata["model_files"].items():
        models[model_name] = joblib.load(ROOT / rel_path)
    return metadata, models


def probability_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return None


def compute_metrics(model, x_eval, y_eval):
    y_pred = model.predict(x_eval)
    y_score = probability_scores(model, x_eval)

    metric_values = {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_score) if y_score is not None else float("nan"),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, y_pred),
    }

    cm = confusion_matrix(y_eval, y_pred)
    report = classification_report(y_eval, y_pred, output_dict=False)
    return metric_values, cm, report


def draw_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="BITS ML Assignment 2", page_icon="📊", layout="wide")
    st.title("Classification Model Evaluator")

    try:
        metadata, models = load_metadata_and_models()
    except Exception as exc:
        st.error("App initialization failed.")
        st.exception(exc)
        st.info(
            "Ensure these files are committed to GitHub: app.py, requirements.txt, "
            "model_details.json, model_metrics.csv, test_data.csv, and the model/ folder."
        )
        st.stop()

    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    st.markdown(
        f"Dataset used for training: **{metadata['dataset_name']}** (Source: {metadata['dataset_source']})."
    )

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Select model", list(models.keys()))
        uploaded = st.file_uploader("Upload test CSV", type=["csv"])

    if uploaded is not None:
        eval_df = pd.read_csv(uploaded)
        st.success("Uploaded test file loaded.")
    else:
        eval_df = pd.read_csv(ROOT / "test_data.csv")
        st.info("Using default test_data.csv.")

    missing_features = [col for col in feature_columns if col not in eval_df.columns]
    if missing_features:
        st.error(
            "Uploaded CSV is missing required feature columns. "
            f"Missing count: {len(missing_features)}"
        )
        st.stop()

    x_eval = eval_df[feature_columns]
    model = models[model_name]

    col1, col2 = st.columns([1, 1])

    if target_column in eval_df.columns:
        y_eval = eval_df[target_column]
        metric_values, cm, report = compute_metrics(model, x_eval, y_eval)

        with col1:
            st.subheader(f"Metrics for {model_name}")
            metrics_df = pd.DataFrame([metric_values]).T
            metrics_df.columns = ["Value"]
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

        with col2:
            st.subheader("Confusion Matrix")
            draw_confusion_matrix(cm)

        st.subheader("Classification Report")
        st.code(report)
    else:
        st.warning(
            "No target column found in uploaded CSV. "
            "Predictions are shown, but metrics require the target column."
        )
        predictions = model.predict(x_eval)
        out_df = eval_df.copy()
        out_df["prediction"] = predictions
        st.dataframe(out_df.head(30), use_container_width=True)

    st.subheader("Model Comparison on Default Test Set")
    summary_df = pd.read_csv(ROOT / "model_metrics.csv")
    st.dataframe(summary_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}",
    }), use_container_width=True)


if __name__ == "__main__":
    main()
