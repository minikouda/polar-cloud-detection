'''
This script runs the best logistic regression model for reproducibility
'''

import os
import pandas as pd
from prepare_data import prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import json

# use the selected features and model parameters best identified from the tuning process in the notebook to run the final model and save the results
with open(os.path.join("../results/logistic_main_run", "run_metadata.json")) as f:
    metadata = json.load(f)
    DATA_PARAMS_SUBSET = metadata["data_params"]
    MODEL_PARAMS = metadata["best_model_params"]
    MODEL_THRESHOLD = metadata["best_model_threshold"]

if isinstance(MODEL_PARAMS.get("class_weight"), dict):
    # convert class_weight keys from string back to int if needed
    MODEL_PARAMS["class_weight"] = {int(k): v for k, v in MODEL_PARAMS["class_weight"].items()}

DATA_PARAMS = {
    "filepaths": ["../data/O012791.npz", "../data/O013257.npz", "../data/O013490.npz"],
    "coord_cols": ["y_coord", "x_coord"],
    "image_col": "image_id",
    "label_col": "label",
    "train_images": ["O013490"],
    "val_images": ["O012791"],
    "test_images": ["O013257"],
    "labeled_only": True,
    "map_label_to_binary": True,
    **DATA_PARAMS_SUBSET,  # use the same data processing settings as the tuning run
}

selected_features_json = os.path.join("../results/logistic_main_run", "selected_features.json")
selected_features_csv = os.path.join("../results/logistic_main_run", "selected_features.csv")

if os.path.exists(selected_features_json):
    with open(selected_features_json) as f:
        SELECTED_FEATURES = json.load(f)
else:
    # Backward compatibility with older outputs.
    SELECTED_FEATURES = pd.read_csv(selected_features_csv)["feature"].tolist()


# =======================================================
# Helper function for evaluating and saving 
# =======================================================

def evaluate_model(y_true, y_prob, threshold=0.5, y_pred=None, dataset_name=None):
    """
    Evaluate a binary classifier using probabilities and a decision threshold.
    """
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    if dataset_name is not None:
        print(f"\n{dataset_name} Performance:")

    print("Evaluation:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }

def run_best_lr(
        data_params = DATA_PARAMS,
        selected_features = SELECTED_FEATURES,
        model_params = MODEL_PARAMS,
        threshold = MODEL_THRESHOLD,
        output_dir = '../results/logistic_best_run'
):
    # prepare data
    train_df, val_df, test_df,_ = prepare_data(**data_params)
    X_train, y_train = train_df[selected_features], train_df[data_params["label_col"]]
    X_val, y_val = val_df[selected_features], val_df[data_params["label_col"]]
    X_test, y_test = test_df[selected_features], test_df[data_params["label_col"]]

    # fit model
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    # predict probabilities and labels based on the best threshold identified from tuning
    train_prob = model.predict_proba(X_train)[:, 1]
    train_pred = (train_prob >= threshold).astype(int)
    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= threshold).astype(int)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    # evaluate and save results
    train_results = evaluate_model(y_train, y_prob=train_prob, threshold=threshold, y_pred=train_pred, dataset_name="Training Set")
    val_results = evaluate_model(y_val, y_prob=val_prob, threshold=threshold, y_pred=val_pred, dataset_name="Validation Set")
    test_results = evaluate_model(y_test, y_prob=test_prob, threshold=threshold, y_pred=test_pred, dataset_name="Test Set")
    # combine all results into a dictionary
    results = []
    results.append({
        "dataset": "train",
        **train_results
    })
    results.append({
        "dataset": "val",
        **val_results
    })
    results.append({
        "dataset": "test",
        **test_results
    })
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "final_results.csv"))
    return results


if __name__ == "__main__":
    results = run_best_lr()
    print("\nFinished.")
    print("Best threshold:", MODEL_THRESHOLD)
    print("Validation F1:", results[1]["f1_score"])
    print("Test F1:", results[2]["f1_score"])
    









