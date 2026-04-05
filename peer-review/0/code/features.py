import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter


def keep_labeled(arr: np.ndarray) -> np.ndarray:
    """Keep only expert-labeled pixels (+1 cloud, -1 no cloud)."""
    return arr[arr[:, 10] != 0]


def engineer_features(arr_img: np.ndarray, window_size: int = 3, eps: float = 1e-6):
    """Engineer patch-statistics and nonlinear interaction features.

    Parameters
    ----------
    arr_img : np.ndarray
        Labeled pixel rows with columns:
        [y, x, NDAI, SD, CORR, DF, CF, BF, AF, AN, Label].
    window_size : int
        Square neighborhood size used for patch statistics.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    X_base, X_new, y, base_names, new_names
    """
    cols = ["Y", "X", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "Label"]
    df = pd.DataFrame(arr_img, columns=cols)

    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)
    df["Label"] = df["Label"].astype(int)

    predictors = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
    rad_cols = ["DF", "CF", "BF", "AF", "AN"]

    X_base = df[predictors].to_numpy(dtype=float)
    y = df["Label"].to_numpy()
    base_names = predictors

    x = df["X"].to_numpy()
    y_coord = df["Y"].to_numpy()

    h = int(y_coord.max()) + 1
    w = int(x.max()) + 1
    c = len(predictors)

    feat = df[predictors].to_numpy(dtype=float)
    grid = np.full((h, w, c), np.nan, dtype=float)
    grid[y_coord, x, :] = feat

    chan_means = np.nanmean(grid, axis=(0, 1))
    nan_mask = np.isnan(grid)
    if np.any(nan_mask):
        grid[nan_mask] = np.take(chan_means, np.where(nan_mask)[2])

    grid_mean = uniform_filter(grid, size=(window_size, window_size, 1), mode="reflect")

    grid_sq_mean = uniform_filter(grid**2, size=(window_size, window_size, 1), mode="reflect")
    grid_var = np.maximum(grid_sq_mean - grid_mean**2, 0.0)
    grid_std = np.sqrt(grid_var)

    grid_max = maximum_filter(grid, size=(window_size, window_size, 1), mode="reflect")
    grid_min = minimum_filter(grid, size=(window_size, window_size, 1), mode="reflect")
    grid_contrast = grid_max - grid_min

    patch_mean = grid_mean[y_coord, x, :]
    patch_std = grid_std[y_coord, x, :]
    patch_contrast = grid_contrast[y_coord, x, :]

    patch_names = []
    for j, var in enumerate(predictors):
        df[f"{var}_mean"] = patch_mean[:, j]
        df[f"{var}_std"] = patch_std[:, j]
        df[f"{var}_contrast"] = patch_contrast[:, j]
        patch_names += [f"{var}_mean", f"{var}_std", f"{var}_contrast"]

    df["SD_log"] = np.log(np.abs(df["SD"]) + eps)
    df["NDAI_sq"] = df["NDAI"] ** 2
    df["CORR_sq"] = df["CORR"] ** 2

    df["Rad_mean"] = df[rad_cols].mean(axis=1)
    df["Rad_std"] = df[rad_cols].std(axis=1, ddof=0)

    nonlinear_names = ["SD_log", "NDAI_sq", "CORR_sq", "Rad_mean", "Rad_std"]

    df["NDAI_x_SD"] = df["NDAI"] * df["SD"]
    df["NDAI_div_SD"] = df["NDAI"] / (df["SD"] + eps)
    df["NDAI_div_CORR"] = df["NDAI"] / (df["CORR"] + eps)
    df["CORR_div_SD"] = df["CORR"] / (df["SD"] + eps)

    df["AF_minus_AN"] = df["AF"] - df["AN"]
    df["DF_minus_AN"] = df["DF"] - df["AN"]
    df["CF_minus_BF"] = df["CF"] - df["BF"]
    df["AF_minus_BF"] = df["AF"] - df["BF"]

    interaction_names = [
        "NDAI_x_SD",
        "NDAI_div_SD",
        "NDAI_div_CORR",
        "CORR_div_SD",
        "AF_minus_AN",
        "DF_minus_AN",
        "CF_minus_BF",
        "AF_minus_BF",
    ]

    new_names = patch_names + nonlinear_names + interaction_names
    X_new = df[new_names].to_numpy(dtype=float)

    return X_base, X_new, y, base_names, new_names


def engineer_features_unlabeled(arr_img: np.ndarray, window_size: int = 3, eps: float = 1e-6):
    """Engineer base and new features for unlabeled pixel arrays.

    Parameters
    ----------
    arr_img : np.ndarray
        Pixel rows with columns [y, x, NDAI, SD, CORR, DF, CF, BF, AF, AN].

    Returns
    -------
    X_base, X_new, base_names, new_names
    """
    cols = ["Y", "X", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
    df = pd.DataFrame(arr_img, columns=cols)

    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)

    predictors = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
    rad_cols = ["DF", "CF", "BF", "AF", "AN"]

    x_base = df[predictors].to_numpy(dtype=float)
    base_names = predictors

    x = df["X"].to_numpy()
    y_coord = df["Y"].to_numpy()

    h = int(y_coord.max()) + 1
    w = int(x.max()) + 1
    c = len(predictors)

    feat = df[predictors].to_numpy(dtype=float)
    grid = np.full((h, w, c), np.nan, dtype=float)
    grid[y_coord, x, :] = feat

    chan_means = np.nanmean(grid, axis=(0, 1))
    nan_mask = np.isnan(grid)
    if np.any(nan_mask):
        grid[nan_mask] = np.take(chan_means, np.where(nan_mask)[2])

    grid_mean = uniform_filter(grid, size=(window_size, window_size, 1), mode="reflect")
    grid_sq_mean = uniform_filter(grid**2, size=(window_size, window_size, 1), mode="reflect")
    grid_var = np.maximum(grid_sq_mean - grid_mean**2, 0.0)
    grid_std = np.sqrt(grid_var)
    grid_max = maximum_filter(grid, size=(window_size, window_size, 1), mode="reflect")
    grid_min = minimum_filter(grid, size=(window_size, window_size, 1), mode="reflect")
    grid_contrast = grid_max - grid_min

    patch_mean = grid_mean[y_coord, x, :]
    patch_std = grid_std[y_coord, x, :]
    patch_contrast = grid_contrast[y_coord, x, :]

    patch_names = []
    for j, var in enumerate(predictors):
        df[f"{var}_mean"] = patch_mean[:, j]
        df[f"{var}_std"] = patch_std[:, j]
        df[f"{var}_contrast"] = patch_contrast[:, j]
        patch_names += [f"{var}_mean", f"{var}_std", f"{var}_contrast"]

    df["SD_log"] = np.log(np.abs(df["SD"]) + eps)
    df["NDAI_sq"] = df["NDAI"] ** 2
    df["CORR_sq"] = df["CORR"] ** 2
    df["Rad_mean"] = df[rad_cols].mean(axis=1)
    df["Rad_std"] = df[rad_cols].std(axis=1, ddof=0)

    nonlinear_names = ["SD_log", "NDAI_sq", "CORR_sq", "Rad_mean", "Rad_std"]

    df["NDAI_x_SD"] = df["NDAI"] * df["SD"]
    df["NDAI_div_SD"] = df["NDAI"] / (df["SD"] + eps)
    df["NDAI_div_CORR"] = df["NDAI"] / (df["CORR"] + eps)
    df["CORR_div_SD"] = df["CORR"] / (df["SD"] + eps)
    df["AF_minus_AN"] = df["AF"] - df["AN"]
    df["DF_minus_AN"] = df["DF"] - df["AN"]
    df["CF_minus_BF"] = df["CF"] - df["BF"]
    df["AF_minus_BF"] = df["AF"] - df["BF"]

    interaction_names = [
        "NDAI_x_SD",
        "NDAI_div_SD",
        "NDAI_div_CORR",
        "CORR_div_SD",
        "AF_minus_AN",
        "DF_minus_AN",
        "CF_minus_BF",
        "AF_minus_BF",
    ]

    new_names = patch_names + nonlinear_names + interaction_names
    x_new = df[new_names].to_numpy(dtype=float)

    return x_base, x_new, base_names, new_names
