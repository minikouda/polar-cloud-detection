import glob
import numpy as np


def make_data(patch_size=9, filepaths=None):
    """
    Load the image data and create patches from it.
    Args:
        patch_size: The size of the patches to create.
    Returns:
        images_long: A list of numpy arrays of the original images.
        patches: A list of lists of patches for each image.
    """

    # load images
    if filepaths is None:
        filepaths = sorted(glob.glob("../data/*.npz"))
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]  # remove labels
        images_long.append(data)

    # Compute global min and max for x and y over all images
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    # Reshape each image onto the common grid.
    nchannels = images_long[0].shape[1] - 2
    images = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        # Use global minimums to get relative coordinates.
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width), dtype=np.float32)
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    # Now that all images have the same shape, convert to a 4D array.
    images = np.array(images, dtype=np.float32)
    pad_len = patch_size // 2

    # Global normalization across images.
    means = np.mean(images, axis=(0, 2, 3), dtype=np.float64)[:, None, None].astype(np.float32)
    stds = np.std(images, axis=(0, 2, 3), dtype=np.float64)[:, None, None].astype(np.float32)
    images = (images - means) / stds

    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        # Pad the image by reflecting across the border.
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        # Use global min values to compute relative indices.
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float32))
        patches.append(patches_img)

    return images_long, patches


def make_data_on_the_fly(patch_size=9, mode="all"):
    """
    Load the image data and prepare metadata for on-the-fly patch extraction.
    Args:
        patch_size: The size of the patches to create.
        mode: One of "all", "unlabeled_only", or "labeled_only".
    Returns:
        images_long: A list of numpy arrays of the original images.
        images_padded: A list of the normalized and padded images.
        coords: array of (image_idx, y_rel, x_rel) for each pixel.
        image_codes: A list of image file IDs (e.g., O012345) aligned with image_idx.
    """

    # load images
    valid_modes = {"all", "unlabeled_only", "labeled_only"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

    filepaths = sorted(glob.glob("../data/*.npz"))
    images_long = []
    image_codes = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        has_label = data.shape[1] == 11

        if mode == "unlabeled_only" and has_label:
            continue
        if mode == "labeled_only" and not has_label:
            continue

        if has_label:
            data = data[:, :-1]  # remove labels
        images_long.append(data)
        image_codes.append(fp.split("/")[-1].replace(".npz", ""))

    if len(images_long) == 0:
        raise ValueError(f"No images left after filtering with mode='{mode}'")

    # Compute global min and max for x and y over all images
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    # Reshape each image onto the common grid.
    nchannels = images_long[0].shape[1] - 2
    images = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width), dtype=np.float32)
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    # Now that all images have the same shape, convert to a 4D array.
    images = np.array(images, dtype=np.float32)
    pad_len = patch_size // 2

    # Global normalization across images.
    means = np.mean(images, axis=(0, 2, 3), dtype=np.float64)[:, None, None].astype(np.float32)
    stds = np.std(images, axis=(0, 2, 3), dtype=np.float64)[:, None, None].astype(np.float32)
    images = (images - means) / stds

    # Pad once and reuse in Dataset.
    images_padded = np.pad(
        images,
        ((0, 0), (0, 0), (pad_len, pad_len), (pad_len, pad_len)),
        mode="reflect",
    ).astype(np.float32, copy=False)

    # Build a global index table: each row maps to one patch center.
    coords = []
    for i in range(len(images_long)):
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        y_rel = ys - global_miny
        x_rel = xs - global_minx
        for y, x in zip(y_rel, x_rel):
            coords.append((i, int(y), int(x)))

    coords = np.array(coords)
    return images_long, images_padded, coords, image_codes
