import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def assign_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign quadrants within each image using image-specific median x and y.
    """
    df = df.copy()

    medians = (
        df.groupby("image_id")[["x", "y"]]
        .median()
        .rename(columns={"x": "median_x", "y": "median_y"})
    )

    df = df.merge(medians, left_on="image_id", right_index=True, how="left")

    def get_q(row):
        if row["x"] <= row["median_x"] and row["y"] <= row["median_y"]:
            return "Q1"
        elif row["x"] > row["median_x"] and row["y"] <= row["median_y"]:
            return "Q2"
        elif row["x"] <= row["median_x"] and row["y"] > row["median_y"]:
            return "Q3"
        else:
            return "Q4"

    df["quadrant"] = df.apply(get_q, axis=1)
    df.drop(columns=["median_x", "median_y"], inplace=True)

    return df


def prepare_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    drop_zero_labels: bool = True,
    drop_columns=("image_id", "x", "y")
):
    """
    Prepare train and test data:
    - assign quadrants to train
    - filter out expert_label == 0 if needed
    - create X/y
    - create groups for GroupKFold
    """

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train = assign_quadrants(df_train)

    if drop_zero_labels:
        df_train_filtered = df_train[df_train["expert_label"] != 0].copy()
        df_test_filtered = df_test[df_test["expert_label"] != 0].copy()
    else:
        df_train_filtered = df_train.copy()
        df_test_filtered = df_test.copy()

    df_train_filtered["group"] = (
        df_train_filtered["image_id"].astype(str)
        + "_"
        + df_train_filtered["quadrant"].astype(str)
    )

    groups = df_train_filtered["group"]

    # X/y before dropping quadrant from train
    X_train = df_train_filtered.drop(
        columns=["expert_label", "quadrant", "group"],
        errors="ignore"
    )
    y_train = df_train_filtered["expert_label"]

    X_test = df_test_filtered.drop(columns=["expert_label"], errors="ignore")
    y_test = df_test_filtered["expert_label"]

    # Drop unnecessary columns
    X_train = X_train.drop(columns=list(drop_columns), errors="ignore")
    X_test = X_test.drop(columns=list(drop_columns), errors="ignore")

    return X_train, y_train, X_test, y_test, groups, df_train_filtered, df_test_filtered


def convert_labels_for_lgbm(y_train: pd.Series, y_test: pd.Series):
    """
    Convert labels from {-1, 1} to {0, 1} for LightGBM.
    """
    y_train_lgb = y_train.replace(-1, 0)
    y_test_lgb = y_test.replace(-1, 0)
    return y_train_lgb, y_test_lgb


def tune_lightgbm(X_train, y_train, groups, param_grid=None, n_jobs=-1, verbose=1):
    """
    Tune LightGBM using GroupKFold.
    """
    if param_grid is None:
        param_grid = {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1, 0.15]
        }

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="accuracy",
        boosting_type="gbrt",
        verbose=-1
    )

    n_splits = len(pd.Series(groups).unique())
    gkf = GroupKFold(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=gkf,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid_search.fit(X_train, y_train, groups=groups)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_params_, grid_search


def train_full_model(X_train, y_train_lgb, params_lgb, n_estimators=1000):
    """
    Train LightGBM on full training data.
    """
    model = lgb.LGBMClassifier(**params_lgb, n_estimators=n_estimators)
    model.fit(X_train, y_train_lgb)
    return model


def get_feature_importance(model, X_train):
    """
    Return sorted feature importance DataFrame.
    """
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return feature_importance


def clean_top_features(top_features, excluded_prefixes=None):
    """
    Remove features containing unwanted prefixes.
    """
    if excluded_prefixes is None:
        excluded_prefixes = ["AF_", "BF_", "CF_", "DF_", "AN_"]

    cleaned = [
        f for f in top_features
        if not any(prefix in f for prefix in excluded_prefixes)
    ]
    return cleaned


def plot_top_features(feature_importance, top_n=25):
    """
    Plot top N important features.
    """
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feature_importance.head(top_n),
        x="importance",
        y="feature"
    )
    plt.title(f"Top {top_n} Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test_lgb, plot_cm=True):
    """
    Predict and evaluate model on test set.
    """
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test_lgb, y_pred))

    accuracy = accuracy_score(y_test_lgb, y_pred)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_test_lgb, y_pred)

    if plot_cm:
        plt.figure(figsize=(8, 6))
        plt.title("Confusion Matrix - LightGBM")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }


def run_lightgbm_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    param_grid=None,
    top_n_features=105,
    excluded_prefixes=None,
    n_estimators=1000,
    plot_features=True,
    plot_cm=True
):
    """
    Central function:
    - prepares data
    - tunes LightGBM
    - trains full model
    - selects top features
    - retrains on selected features
    - evaluates on test set
    """

    # Prepare data
    X_train, y_train, X_test, y_test, groups, _, _ = prepare_data(df_train, df_test)

    # Convert labels
    y_train_lgb, y_test_lgb = convert_labels_for_lgbm(y_train, y_test)

    # Tune model
    best_params, grid_search = tune_lightgbm(
        X_train=X_train,
        y_train=y_train_lgb,
        groups=groups,
        param_grid=param_grid
    )

    # Base params
    params_lgb = {
        "objective": "binary",
        "metric": "accuracy",
        "boosting_type": "gbrt",
        "verbose": -1,
    }
    params_lgb.update(best_params)

    # Train full model on all features
    lgb_model_full = train_full_model(X_train, y_train_lgb, params_lgb, n_estimators=n_estimators)

    # Feature importance
    feature_importance = get_feature_importance(lgb_model_full, X_train)

    if plot_features:
        plot_top_features(feature_importance, top_n=25)

    # Select top features
    top_features = feature_importance.head(top_n_features)["feature"].tolist()
    top_features_cleaned = clean_top_features(top_features, excluded_prefixes=excluded_prefixes)

    # Restrict data
    X_train_top = X_train[top_features_cleaned]
    X_test_top = X_test[top_features_cleaned]

    # Train final model
    lgb_model_final = train_full_model(X_train_top, y_train_lgb, params_lgb, n_estimators=n_estimators)

    # Evaluate
    results = evaluate_model(lgb_model_final, X_test_top, y_test_lgb, plot_cm=plot_cm)

    return {
        "best_params": best_params,
        "params_lgb": params_lgb,
        "grid_search": grid_search,
        "feature_importance": feature_importance,
        "top_features_cleaned": top_features_cleaned,
        "model_full": lgb_model_full,
        "model_final": lgb_model_final,
        "X_train_top": X_train_top,
        "X_test_top": X_test_top,
        "y_train_lgb": y_train_lgb,
        "y_test_lgb": y_test_lgb,
        "results": results
    }



def main():
    print("\n===== lightGBM =====")
    # Load data
    df_train_1 = pd.read_csv("../data/O012791.csv")
    df_train_2 = pd.read_csv("../data/O013257.csv")
    df_test = pd.read_csv("../data/O013490.csv")

    # Combine train data
    df_train = pd.concat([df_train_1, df_train_2], ignore_index=True)

    # Run pipeline
    output = run_lightgbm_pipeline(
        df_train=df_train,
        df_test=df_test,
        top_n_features=105,
        excluded_prefixes=["AF_", "BF_", "CF_", "DF_", "AN_"],
        plot_features=False,
        plot_cm=False
    )

    # Print key results
    print("\n===== FINAL RESULTS =====")
    print("Best params:", output["best_params"])
    print("Accuracy:", output["results"]["accuracy"])

    print("\nTop 10 selected features:")
    print(output["top_features_cleaned"][:10])


if __name__ == "__main__":
    main()