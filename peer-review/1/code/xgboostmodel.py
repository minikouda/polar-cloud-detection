import os
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt


def xgboost_train(
    X_train,
    y_train,
    model_path='./xgboost_model.joblib',
    scale_pos_weight=1, 
    n_estimators: int = 100,
    max_depth: int = 6,                  # XGBoost default is 6
    learning_rate: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """
    Trains an XGBClassifier and saves the model to file.
    """
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",           # suppresses XGBoost's default warning
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    print(f"Model trained and saved to {model_path}")

    return clf


def xgboost_predict(X_test, model_path='./xgboost_model.joblib'):
    """
    Load a trained XGBoost model from file and make predictions.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    loaded_model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    predictions = loaded_model.predict(X_test)
    return predictions


def _evaluate(clf, X, y, split=""):
    """
    Helper function to calculate metrics and print a classification report.
    """
    preds = clf.predict(X)

    metrics = {
        "accuracy":    accuracy_score(y, preds),
        "f1_cloud":    f1_score(y, preds, pos_label=1,  zero_division=0),
        "f1_no_cloud": f1_score(y, preds, pos_label=0,  zero_division=0),
        "f1_macro":    f1_score(y, preds, average="macro", zero_division=0),
    }

    print(f"\n── {split} Metrics ──")
    print(classification_report(y, preds, target_names=["not cloud", "cloud"], zero_division=0))

    return metrics


def xgboost_evaluate(clf, X_val, y_val, X_test, y_test, 
                     train_df, val_df, test_df, FEATURE_COLS, 
                     top_ft=25, bar=False):
    """
    Main evaluation function that runs validation/test passes and extracts feature importance.
    """
    val_metrics  = _evaluate(clf, X_val,  y_val,  split="Validation")
    test_metrics = _evaluate(clf, X_test, y_test, split="Test")

    # Extract and sort feature importance (gain-based, same API as sklearn)
    print("\nFeature Importances:")
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(importance_df.to_string(index=False))

    top_features = importance_df.head(top_ft)["feature"].tolist()
    if bar: 
        clf.get_booster().feature_names = importance_df['feature'].tolist()
        plot_importance(clf)
        plt.show()

    return {
        "model":              clf,
        "train_df":           train_df,
        "val_df":             val_df,
        "test_df":            test_df,
        "val_metrics":        val_metrics,
        "test_metrics":       test_metrics,
        "feature_importance": importance_df,
        "top_features":       top_features,
    }