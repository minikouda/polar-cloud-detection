'''
This script runs the best svm model for reproducibility
'''
import os
import pandas as pd
from prepare_data import prepare_data
from sklearn.svm import SVC
import json

from run_lr import evaluate_model  # reuse the same evaluation function for consistency


# use the selected features and model parameters best identified from the tuning process in the notebook to run the final model and save the results
with open(os.path.join("../results/svm_main_run", "run_metadata.json")) as f:
    metadata = json.load(f)
    DATA_PARAMS_SUBSET = metadata["data_params"]
    MODEL_PARAMS = metadata["best_model_params"]

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

selected_features_json = os.path.join("../results/svm_main_run", "selected_features.json")
selected_features_csv = os.path.join("../results/svm_main_run", "selected_features.csv")

if os.path.exists(selected_features_json):
    with open(selected_features_json) as f:
        SELECTED_FEATURES = json.load(f)
else:
    # Backward compatibility with older outputs.
    SELECTED_FEATURES = pd.read_csv(selected_features_csv)["feature"].tolist()

#=======================================================
# Run Best SVM
#=======================================================


def run_best_svm(data_params = DATA_PARAMS,
                 model_params = MODEL_PARAMS, 
                 selected_features = SELECTED_FEATURES, 
                output_dir="../results/svm_best_run"):

    # prepare data
    train_df, val_df, test_df,_ = prepare_data(**data_params)
    X_train, y_train = train_df[selected_features], train_df[data_params["label_col"]]
    X_val, y_val = val_df[selected_features], val_df[data_params["label_col"]]
    X_test, y_test = test_df[selected_features], test_df[data_params["label_col"]]

    # train model
    best_svm_model = SVC(**model_params)
    best_svm_model.fit(X_train, y_train)

    # predict probabilities for evaluation
    train_prob = best_svm_model.predict_proba(X_train)[:, 1]
    val_prob = best_svm_model.predict_proba(X_val)[:, 1]
    test_prob = best_svm_model.predict_proba(X_test)[:, 1]

    # fro svm we can directly get the predicted label
    train_pred = best_svm_model.predict(X_train)
    val_pred = best_svm_model.predict(X_val)
    test_pred = best_svm_model.predict(X_test)

    # evaluate and save results
    train_results = evaluate_model(y_train, y_prob=train_prob, y_pred=train_pred, dataset_name="Training Set")
    val_results = evaluate_model(y_val, y_prob=val_prob, y_pred=val_pred, dataset_name="Validation Set")
    test_results = evaluate_model(y_test, y_prob=test_prob, y_pred=test_pred, dataset_name="Test Set")

    # save results to a dictionary
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
    results = run_best_svm()
    print("\nFinished.")
    print("Validation F1:", results[1]["f1_score"])
    print("Test F1:", results[2]["f1_score"])
