
'''
This contains functions to prepare data for main classification model.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

def clean( filepath,
          column_names= [
                        "y_coord",
                        "x_coord",
                        "NDAI",
                        "SD",
                        "CORR",
                        "radiance_DF",
                        "radiance_CF",
                        "radiance_BF",
                        "radiance_AF",
                        "radiance_AN",
                        "label"]
        ):
    '''
    This function takes in a filepath to a .npz file, loads the labeled data, and performs the following cleaning steps:
    1. Converts the numpy array to a pandas dataframe with the specified column names.
    2. Drops duplicate rows based on the 'y_coord' and 'x_coord' columns, keeping only the first occurrence.
    3. Drops any rows with missing values and prints a warning if any missing values are found.
    4. Performs sanity checks on the data:
        - Checks if the 'label' column contains only -1, 0, or 1. If invalid values are found, it prints a warning and keeps only the valid labels.
        - Checks if the radiance columns ('radiance_DF', 'radiance_CF', 'radiance_BF', 'radiance_AF', 'radiance_AN') contain only non-negative values. If negative values are found, it prints a warning and keeps only the non-negative values.
        - Checks if the 'NDAI' column contains values within the range [-1, 1]. If values outside this range are found, it prints a warning and keeps only the values within the expected range (allowing for some outliers).
        - Checks if the 'SD' column contains only non-negative values. If negative values are found, it prints a warning and keeps only the non-negative values.
        - Checks if the 'CORR' column contains values within the range [-1, 1]. If values outside this range are found, it prints a warning and keeps only the values within the expected range (allowing for some outliers).
    5. Returns the cleaned dataframe.
    '''
    with np.load(filepath) as data: 
        df = dict(data)['arr_0']

    if df.shape[1] == len(column_names)-1:
        # if the label column is missing, add a label column with all values set null
        print(f"Warning:  {filepath} is unlabeled. Adding a label column with all values set to 0.")
        df = np.hstack((df, np.full((df.shape[0], 1), 0)))  # add a column of 0 values for the label
        print(f"Loaded data from {filepath} with shape {df.shape}")

    df = pd.DataFrame(df, columns=column_names)

    # check the shape of the dataframe
    if df.shape[1] != len(column_names):
        print(f"Warning: Expected {len(column_names)} columns but found {df.shape[1]} columns. Check the column names and data format.")
        return None

    # drop duplicates based on y_coord and x_coord
    df = df.drop_duplicates(subset=['y_coord', 'x_coord'])

    # drop missing values
    if df.isnull().sum().sum() > 0:
        print(f"Warning: {df.isnull().sum().sum()} missing values found. Dropping missing values.")
    df = df.dropna()

    # sanity check
    
    # Check if labels are only -1, 0, or 1
    if not df['label'].isin([-1, 0, 1]).all():
        print("Sanity Check Failed: label contains invalid values")
        df = df[df['label'].isin([-1, 0, 1])] # Keep only valid labels
    
    # Check if radiance values are non-negative
    radiance_columns = ['radiance_DF', 'radiance_CF', 'radiance_BF', 'radiance_AF', 'radiance_AN']
    for col in radiance_columns:
        if not (df[col] >= 0).all():
            print(f"Sanity Check Failed: {col} contains negative values")
            df = df[df[col] >= 0] # Keep only non-negative values

    # Check if NADI values are within a reasonable range (e.g., -1 to 1)(allowing for some outliers)
    if not ((df['NDAI'] >= -1) & (df['NDAI'] <= 1)).all():
        print("Sanity Check Failed: NDAI contains values outside the expected range [-1, 1]")
        df = df[(df['NDAI'] >= -1.2) & (df['NDAI'] <= 1.2)] # Keep only values within the expected range ( alllowing for some outliers)
    # Check if SD values are non-negative
    if not (df['SD'] >= 0).all():
        print("Sanity Check Failed: SD contains negative values")
        df = df[df['SD'] >= 0] # Keep only non-negative values
    # Check if CORR values are between -1 and 1
    if not ((df['CORR'] >= -1) & (df['CORR'] <= 1)).all():
        print("Sanity Check Failed: CORR contains values outside the expected range [-1, 1]")
        df = df[(df['CORR'] >= -1.2) & (df['CORR'] <= 1.2)] # Keep only values within the expected range ( allowing for some outliers)
    
    return df



# prepare funtion with the splitting judgment calls

def prepare_data(
    filepaths: list,
    coord_cols: list = ["y_coord", "x_coord"],
    image_col: str = "image_id",
    label_col: str = "label",
    train_images: list | None = None,
    val_images: list | None = None,
    test_images: list | None = None,
    unlabeled_images: list[str] | None = None,  # if specified, these images will be prepared for sanity checks and visualization but not included in train/val/test splits
    # ---- common
    labeled_only: bool = True,
    map_label_to_binary: bool = False,
    # ---- scaling 
    scaler_type: str | None = None,  # "standard", "robust", or None,
    scaler_feature_cols: list | None = None,  # columns to scale (if None, use feature_cols
    # ---- remove outliers 
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0,  # Z-score threshold for outlier removal
    outlier_cols: list | None = None,  # columns to check for outliers (if None, use feature_cols
):
    """
    This function takes in a list of filepaths to .npz files, loads and concatenates the data, and performs the following preparation steps:
    1. Loads and concatenates the data from the specified filepaths, adding an 'image_id' column extracted from the filename.
    2. Subsets the data into train/val/test splits based on the specified image IDs for each split.
    3. Merges engineered features from separate CSV files for each image (if they exist) based on the 'image_id', 'y_coord', and 'x_coord' columns. It drops redundant columns after merging and checks that the merged dataframe has the expected number of columns.
    4. Optionally removes outliers based on Z-score thresholds for specified columns, fitting the outlier detection on the training set and applying it to the validation and test sets.
    5. Optionally scales specified feature columns using either standard scaling or robust scaling, fitting the scaler on the training set and applying it to the validation and test sets.
    6. Optionally filters the data to only include labeled examples (where the label is not 0).
    7. Optionally maps the labels to binary values (1 for cloud, 0 for not cloud).
    8. Returns the prepared train, validation, and test dataframes, as well as an optional dataframe for the unlabeled data if specified.

    Attention: 
    At this stage, we are only merging the engineered features for the labeled data, since that's what we have available. For the unlabeled data, we will keep the original features for sanity checks and visualization, but we won't be able to merge the engineered features since they are not available for the unlabeled images. In the future, if we generate engineered features for the unlabeled data as well, we can easily modify the code to merge those features in as well.

    """
    # Load and concatenate data
    print(f"Loading data from {len(filepaths)} files...")
    dfs = []
    for filepath in filepaths:
        df = clean(filepath)
        df[image_col] = os.path.basename(filepath).split('.')[0]  # Extract image ID from filename
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Subset to specified images for train/val/test
    if train_images is None:
        train_images = []
    if val_images is None:
        val_images = []
    if test_images is None:
        test_images = []

    train_df = df_all[df_all[image_col].isin(train_images)].copy()
    val_df = df_all[df_all[image_col].isin(val_images)].copy()
    test_df = df_all[df_all[image_col].isin(test_images)].copy()
    if unlabeled_images:
        unlabeled_df = df_all[df_all[image_col].isin(unlabeled_images)].copy()
    else:
        unlabeled_df = None


    # temporary: we only have engineered features for the training set for labeled data
    print("Merging engineered features...")
    for df_name, df_ref in zip(['train_df', 'val_df', 'test_df'], [train_df, val_df, test_df]):
        temp_data = []
        for image_id in df_ref[image_col].unique():
            image_df = df_ref[df_ref[image_col] == image_id]
            engineered_filepath = f"../data/{image_id}.csv"
            if os.path.exists(engineered_filepath):
                df_engineered = pd.read_csv(engineered_filepath)
                df_engineered.rename(columns={"y": "y_coord", "x": "x_coord"}, inplace=True)
                merged_df = image_df.merge(df_engineered, on=[image_col, 'y_coord', 'x_coord'], how="left")
                redundant_cols = ['NDAI_y', 'SD_y', 'CORR_y', 'expert_label', 'DF', 'CF', 'BF', 'AF', 'AN']
                cols_to_drop = [c for c in redundant_cols if c in merged_df.columns]
                merged_df.drop(columns=cols_to_drop, inplace=True)
                merged_df.rename(columns={col: col[:-2] for col in merged_df.columns if col.endswith('_x')}, inplace=True)
                assert merged_df.shape[1] == 116, f"Expected 116 columns, but got {merged_df.shape[1]}"
                temp_data.append(merged_df)
                print(f"Successfully merged engineered features from {engineered_filepath} for image {image_id}.")
                print(f"After merging, {image_id} has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
            else:
                print(f"Warning: Engineered features file {engineered_filepath} not found. Skipping feature merging for {image_id}.")
                # Keep original rows for this image when engineered features are missing
                temp_data.append(image_df)
        # Update the original dataframe reference with the merged dataframe
        if temp_data:
            merged_all = pd.concat(temp_data, ignore_index=True)
        else:
            # No engineered features found for any image in this split; keep original dataframe unchanged
            print(f"No engineered features merged for {df_name}. Leaving {df_name} unchanged.")
            merged_all = df_ref
        if df_name == 'train_df':
            train_df = merged_all
        elif df_name == 'val_df':
            val_df = merged_all
        elif df_name == 'test_df':
            test_df = merged_all

    print(f"Finished merging engineered features. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
    # remove outliers based on Z-score within the training set (fit on train, apply to val/test)

    print("Preparing data...")
    if remove_outliers:
        if outlier_cols is None:
            outlier_cols = [col for col in train_df.columns if col not in coord_cols + [image_col, label_col]]
        # Compute Z-scores based on training set
        train_stats = train_df[outlier_cols].agg(['mean', 'std'])
        z_scores_train = (train_df[outlier_cols] - train_stats.loc['mean']) / train_stats.loc['std']
        z_scores_val = (val_df[outlier_cols] - train_stats.loc['mean']) / train_stats.loc['std']
        z_scores_test = (test_df[outlier_cols] - train_stats.loc['mean']) / train_stats.loc['std']

        # Keep only rows where all specified columns have Z-score within the threshold
        train_df = train_df[(z_scores_train.abs() <= outlier_threshold).all(axis=1)].copy()
        val_df = val_df[(z_scores_val.abs() <= outlier_threshold).all(axis=1)].copy()
        test_df = test_df[(z_scores_test.abs() <= outlier_threshold).all(axis=1)].copy()

        if unlabeled_df is not None:
            z_scores_unlabeled = (unlabeled_df[outlier_cols] - train_stats.loc['mean']) / train_stats.loc['std']
            # Optionally, we could also filter the unlabeled data based on outliers for sanity checks and visualization
            unlabeled_df = unlabeled_df[(z_scores_unlabeled.abs() <= outlier_threshold).all(axis=1)].copy()


    # Scaling (fit on train, apply to val/test)
    if scaler_type is not None:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'robust', or None")
        if scaler_feature_cols is None:
            scaler_feature_cols = [col for col in train_df.columns if col not in coord_cols + [image_col, label_col]]

        train_df[scaler_feature_cols] = scaler.fit_transform(train_df[scaler_feature_cols])
        val_df[scaler_feature_cols] = scaler.transform(val_df[scaler_feature_cols])
        test_df[scaler_feature_cols] = scaler.transform(test_df[scaler_feature_cols])
        if unlabeled_df is not None:
            unlabeled_df[scaler_feature_cols] = scaler.transform(unlabeled_df[scaler_feature_cols])
        
    # Drop unlabeled AFTER split
    if labeled_only:
        train_df = train_df[train_df[label_col] != 0].copy()
        val_df = val_df[val_df[label_col] != 0].copy()
        test_df = test_df[test_df[label_col] != 0].copy()


    # Optional: map to binary labels
    if map_label_to_binary:
        # 1 -> 1 (cloud), -1 -> 0 (not cloud)
        for df in (train_df, val_df, test_df):
            df[label_col] = (df[label_col] == 1).astype(int)

    return (
        train_df, val_df, test_df, unlabeled_df
    )
