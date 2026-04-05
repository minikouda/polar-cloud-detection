import glob
import os
import numpy as np

# The 3 images with expert labels, and exclude from pretraining.
LABELED_IMAGE_BASENAMES = {"O012791.npz", "O013257.npz", "O013490.npz"}


def make_data(patch_size=9, max_images=None, max_patches_per_image=None, exclude_labeled_images=False, only_labeled_images=False):
    """
    Load the image data and create patches from it.
    Args:
        patch_size: The size of the patches to create.
        max_images: If set, use only the first max_images files (saves memory).
        max_patches_per_image: If set, randomly sample this many patches per image
            so that total patch count stays bounded when using many images.
        exclude_labeled_images: If True, exclude the 3 expert-labeled images (O012791, O013257, O013490)
            so that pretraining uses only the 161 unlabeled images per lab instructions.
        only_labeled_images: If True, load only the 3 expert-labeled images (for finetuning).
            Ignored if exclude_labeled_images is True.
    Returns:
        images_long: A list of numpy arrays of the original images.
        patches: A list of lists of patches for each image.
    """

    # load images (support both lab2/data/*.npz and lab2/data/image_data/*.npz)
    filepaths = sorted(glob.glob("../data/image_data/*.npz") or glob.glob("../data/*.npz"))

    if only_labeled_images:
        filepaths = sorted([fp for fp in filepaths if os.path.basename(fp) in LABELED_IMAGE_BASENAMES])
        print(f"Finetuning mode: using only the 3 labeled images ({len(filepaths)}), order: {[os.path.basename(f) for f in filepaths]}.")
    elif exclude_labeled_images:
        filepaths = [fp for fp in filepaths if os.path.basename(fp) not in LABELED_IMAGE_BASENAMES]
        print(f"Excluding 3 labeled images; using {len(filepaths)} unlabeled images for pretraining.")

    # Optionally limit the number of images to control memory usage
    if max_images is not None:
        filepaths = filepaths[:max_images]
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]
        images_long.append(data)

    # Global min and max for x and y over all images
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    # Reshape images - convert each image from long format into array (channels, height, width)
    nchannels = images_long[0].shape[1] - 2
    images = []
    for img in images_long:
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
    print('done reshaping images')

    images = np.array(images)
    pad_len = patch_size // 2

    # Standardize each channel using global mean and standard deviation across all images
    means = np.mean(images, axis=(0, 2, 3))[:, None, None]
    stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    images = (images - means) / stds

    # Reflection padding
    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
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
        # Per-image cap to avoid OOM when using many images
        if max_patches_per_image is not None and len(patches_img) > max_patches_per_image:
            idx = np.random.choice(len(patches_img), size=max_patches_per_image, replace=False)
            patches_img = [patches_img[j] for j in idx]
        patches.append(patches_img)

    return images_long, patches

