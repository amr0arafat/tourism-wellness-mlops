"""
Model training step.

- Loads the train/test splits produced by the data-prep job (workflow artifact)
- Builds a preprocessing + XGBoost pipeline and tunes it with GridSearchCV
- Logs every tuned parameter combination, the best parameters and the evaluation
  metrics to MLflow
- Saves the best model to tourism_project/deployment/ so the workflow can commit it
"""
import json
import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPLIT_DIR = os.getenv("SPLIT_DIR", ".")
MODEL_DIR = os.getenv("MODEL_DIR", "tourism_project/deployment")
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
THRESHOLD = 0.45      # decision threshold for "will purchase"; slightly below 0.5 to favour recall

# MLflow: the workflow starts a local server on port 5000; in Colab we point it at the ngrok URL.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package")

# ---------------------------------------------------------------- data
Xtrain = pd.read_csv(os.path.join(SPLIT_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(SPLIT_DIR, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(SPLIT_DIR, "ytrain.csv")).squeeze("columns")
ytest = pd.read_csv(os.path.join(SPLIT_DIR, "ytest.csv")).squeeze("columns")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]

# ---------------------------------------------------------------- model
# Preprocessing: impute + scale numeric features, impute + one-hot encode categoricals
preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), categorical_features),
)

# Handle the ~4:1 class imbalance by up-weighting the positive (purchase) class
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid (step name of the classifier inside the pipeline is "xgbclassifier")
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
    "xgbclassifier__reg_lambda": [1, 5],
}


def compute_metrics(y_true, y_prob, prefix):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
    }


with mlflow.start_run(run_name="xgboost-gridsearch"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every tuned parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"candidate-{i:02d}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])
            mlflow.log_metric("cv_std_f1", results["std_test_score"][i])
    print(f"Evaluated {len(results['params'])} parameter combinations with 5-fold CV")

    # Best parameters and model
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("decision_threshold", THRESHOLD)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    print(f"Best CV F1     : {grid_search.best_score_:.4f}")

    # Evaluate on train and test sets
    train_prob = best_model.predict_proba(Xtrain)[:, 1]
    test_prob = best_model.predict_proba(Xtest)[:, 1]
    metrics = {**compute_metrics(ytrain, train_prob, "train"),
               **compute_metrics(ytest, test_prob, "test")}
    mlflow.log_metrics(metrics)

    test_pred = (test_prob >= THRESHOLD).astype(int)
    print("\nTest-set classification report (threshold = %.2f):" % THRESHOLD)
    print(classification_report(ytest, test_pred, digits=3))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(ytest, test_pred))
    print("\nMetrics:", json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    # Save the best model (committed to the repo by the pipeline) + its metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_params": grid_search.best_params_, "threshold": THRESHOLD,
                   **{k: round(v, 4) for k, v in metrics.items()}}, f, indent=2)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    mlflow.log_artifact(METRICS_PATH, artifact_path="model")
    print(f"\nBest model saved to {MODEL_PATH}")
