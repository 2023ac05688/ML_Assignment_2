# BITS ML Assignment 2 - Classification Models and Streamlit Deployment

## a) Problem Statement
Build and evaluate multiple machine learning classification models on a single dataset, compare their performance using standard metrics, and provide an interactive Streamlit app where users can upload test data, select models, and view evaluation results.

## b) Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (also available through `sklearn.datasets`)
- **Task Type:** Binary classification
- **Target Variable:** `target` (0 = malignant, 1 = benign)
- **Total Instances:** 569
- **Total Features:** 30
- **Why selected:** Meets assignment constraints (minimum 500 instances and 12 features) and is suitable for demonstrating multiple classifiers.

## c) Github Repository Link
- Replace this with your actual link before submission:
- **GitHub Repo:** https://github.com/2023ac05688/ML_Assignment_2

Repository contains:
- `app.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `model/` (trained model files + training script)

## d) Models Used and Metrics Comparison
Implemented models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9954 | 0.9595 | 0.9861 | 0.9726 | 0.9245 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9298 | 0.9524 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Observations About Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performance on this dataset with highest Accuracy, F1, and MCC; very strong class separation. |
| Decision Tree | Good interpretability but lower generalization; comparatively weaker AUC and MCC. |
| kNN | Balanced performance, but slightly below top models; may be sensitive to feature scaling and neighbor choice. |
| Naive Bayes | Strong AUC and Recall; performs well despite distribution assumptions, but not the top F1/MCC. |
| Random Forest (Ensemble) | High and stable performance with strong AUC and F1; close second to Logistic Regression. |

**Overall Winner for this dataset:** Logistic Regression

## Streamlit App Features Implemented
- CSV test data upload option (`test_data.csv` supported)
- Model selection dropdown
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Comparison table showing results of all models on test data

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Streamlit Deployment Link
- Replace this with your deployed app URL:
- **Streamlit App:** https://mlassignment2-ckfvjeg9okucrsqc3xf4kb.streamlit.app/

