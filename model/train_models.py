import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def model_probability_scores(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_score = model_probability_scores(model, x_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score) if y_score is not None else float("nan"),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    target_column = "target"
    feature_columns = [col for col in df.columns if col != target_column]

    x = df[feature_columns]
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    metrics_rows = []
    detailed = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        joblib.dump(model, model_dir / f"{model_name.lower().replace(' ', '_')}.joblib")

        scores = evaluate_model(model, x_test, y_test)
        detailed[model_name] = scores

        metrics_rows.append(
            {
                "ML Model Name": model_name,
                "Accuracy": scores["Accuracy"],
                "AUC": scores["AUC"],
                "Precision": scores["Precision"],
                "Recall": scores["Recall"],
                "F1": scores["F1"],
                "MCC": scores["MCC"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="F1", ascending=False)
    metrics_df.to_csv(root / "model_metrics.csv", index=False)

    test_df = x_test.copy()
    test_df[target_column] = y_test.values
    test_df.to_csv(root / "test_data.csv", index=False)

    with open(root / "model_details.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
                "dataset_source": "UCI Machine Learning Repository",
                "target_column": target_column,
                "feature_columns": feature_columns,
                "model_files": {
                    name: f"model/{name.lower().replace(' ', '_')}.joblib"
                    for name in models.keys()
                },
                "metrics": detailed,
            },
            f,
            indent=2,
        )

    print("Training complete. Created: model files, model_metrics.csv, test_data.csv, model_details.json")


if __name__ == "__main__":
    main()
