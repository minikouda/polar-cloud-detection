import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm

def load_npz_files(path, remove_label=False, filepaths=None):
    """Load NPZ files from the specified path and optionally remove the label column. 
    Returns image IDs, full data arrays, patch data arrays, and file paths."""
    image_ids, data_full, data_patch = [], [], []
    if filepaths is None:
        filepaths = glob.glob(path)
    for fp in filepaths:
        image_ids.append(os.path.basename(fp))
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        data_full.append(data)
        data_patch.append(data[:, :-1] if remove_label and data.shape[1] == 11 else data)
    return image_ids, data_full, data_patch, filepaths

def compute_global_grid(data_list):
    """Compute the global grid parameters (min, max, height, width) from a list of data arrays. 
    Returns global minimum y, maximum y, minimum x, maximum x, grid height, and grid width."""
    all_y = np.concatenate([img[:, 0] for img in data_list]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in data_list]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height, width = int(global_maxy - global_miny + 1), int(global_maxx - global_minx + 1)
    return global_miny, global_maxy, global_minx, global_maxx, height, width

def reshape_images(data_list, global_miny, global_minx, height, width):
    """Reshape each image array onto a common grid based on global coordinates. 
    Returns a numpy array of reshaped images."""
    nchannels = data_list[0].shape[1] - 2
    images = []
    for img in data_list:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    return np.array(images)

def normalize_images(images, keepdims=True):
    """Normalize images per channel using global means and standard deviations. 
    Returns the normalized images."""
    if keepdims:
        means = np.mean(images, axis=(0, 2, 3), keepdims=True)
        stds = np.std(images, axis=(0, 2, 3), keepdims=True)
    else:
        means = np.mean(images, axis=(0, 2, 3))[:, None, None]
        stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    return (images - means) / stds

def compute_surrounding_features(patch, exclude_center=False):
    """Compute the minimum, mean, and maximum for each channel of a patch; 
    optionally excluding the center pixel. Returns a numpy array of computed features."""
    nchannels, patch_size, _ = patch.shape
    features = []
    center = patch_size // 2  # Works for odd patch sizes.
    for c in range(nchannels):
        channel_data = patch[c]
        if exclude_center:
            flat = channel_data.flatten()
            center_index = center * patch_size + center
            values = np.delete(flat, center_index)
        else:
            values = channel_data.flatten()
        features.append(values.min())
        features.append(values.mean())
        features.append(values.max())
    return np.array(features)

def make_dataframe_surrounding(path, patch_sizes=[3, 5, 9], filepaths=None):
    """Generate a DataFrame with surrounding features for each patch in labeled data using different patch sizes. 
    Returns a concatenated DataFrame containing min, mean, and max features per channel."""
    data_columns = ['y', 'x', 'NDAI', 'SD', 'CORR', 'DF', 'CF', 'BF', 'AF', 'AN', 'expert_label']
    channel_names = data_columns[2:-1]
    image_ids, images_long_full, images_long_patch, _ = load_npz_files(
        f"{path}/*.npz", remove_label=True, filepaths=filepaths
    )
    if len(images_long_full) == 0:
        raise ValueError("No input npz files found for surrounding feature calculation.")
    global_miny, _, global_minx, _, height, width = compute_global_grid(images_long_full)
    images = reshape_images(images_long_patch, global_miny, global_minx, height, width)
    images = normalize_images(images, keepdims=True)
    df_chunks = []
    for i, orig_data in enumerate(tqdm(images_long_full, desc="surrounding features")):
        padded_images = {p: np.pad(images[i], ((0, 0), (p // 2, p // 2), (p // 2, p // 2)), mode="reflect")
                         for p in patch_sizes}
        chunk_rows = []
        for row in orig_data:
            row_dict = {"image_id": image_ids[i]}
            row_dict.update(dict(zip(data_columns, row)))
            for p in patch_sizes:
                pad_len = p // 2
                y_idx = int(row[0]) - global_miny + pad_len
                x_idx = int(row[1]) - global_minx + pad_len
                patch = padded_images[p][:, y_idx - pad_len:y_idx + pad_len + 1,
                        x_idx - pad_len:x_idx + pad_len + 1].astype(np.float32)
                features = compute_surrounding_features(patch)
                for ch in range(len(channel_names)):
                    row_dict[f"{channel_names[ch]}_{p}_min"] = features[ch * 3]
                    row_dict[f"{channel_names[ch]}_{p}_mean"] = features[ch * 3 + 1]
                    row_dict[f"{channel_names[ch]}_{p}_max"] = features[ch * 3 + 2]
            chunk_rows.append(row_dict)
        df_chunks.append(pd.DataFrame(chunk_rows))
    return pd.concat(df_chunks, ignore_index=True)

def feature_engineering(
    data_dir,
    path_output,
    embedding_dir="../data",
    labeled_only=False,
    filepaths=None,
):
    labeled_ids = {"O012791", "O013257", "O013490"}
    os.makedirs(path_output, exist_ok=True)
    all_filepaths = sorted(glob.glob(f"{data_dir}/*.npz"))

    if labeled_only:
        npz_files = []
        for fp in all_filepaths:
            stem = os.path.splitext(os.path.basename(fp))[0]
            if stem in labeled_ids:
                npz_files.append(fp)
    elif filepaths is None:
        npz_files = all_filepaths
    else:
        npz_files = []
        for fp in filepaths:
            if os.path.isabs(fp):
                npz_files.append(fp)
            else:
                npz_files.append(os.path.join(data_dir, fp))

    if len(npz_files) == 0:
        raise ValueError("No target npz files found after filtering.")

    df_pca = make_dataframe_surrounding(
        data_dir, patch_sizes=[3, 5, 9, 13], filepaths=npz_files
    )

    missing = []
    pairs = []
    for npz_file in tqdm(npz_files, desc="checking embeddings"):
        stem = os.path.splitext(os.path.basename(npz_file))[0]
        file = os.path.join(embedding_dir, f"{stem}_ae.csv")
        if not os.path.exists(file):
            missing.append(stem)
            continue
        pairs.append((stem, file))

    if missing:
        raise FileNotFoundError("Missing embedding files for: " + ", ".join(missing))

    for stem, file in tqdm(pairs, desc="merging and saving"):
        temp_df = pd.read_csv(file)
        if "image_id" in temp_df.columns:
            temp_df = temp_df.drop(columns=["image_id"])

        df_one = df_pca[df_pca["image_id"] == f"{stem}.npz"].copy()
        df_one = pd.merge(df_one, temp_df, on=["x", "y"], how="inner")
        df_one["image_id"] = stem

        cols = df_one.columns.tolist()
        cols = ["image_id"] + [c for c in cols if c != "image_id"]
        df_one = df_one[cols]

        output_file = os.path.join(path_output, f"{stem}.csv")
        df_one.to_csv(output_file, index=False)
