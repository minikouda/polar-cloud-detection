# EXAMPLE USAGE:
# python get_embedding.py configs/default.yaml ../results/default/default.pt

import sys
import os
import glob
import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from autoencoder import Autoencoder
from data import make_data


def get_embedding(
    config_path,
    weights_path,
    labeled_only=False,
    filepaths=None,
    data_dir="../data",
    output_dir="../data",
):
    config = yaml.safe_load(open(config_path, "r"))

    all_filepaths = sorted(glob.glob(f"{data_dir}/*.npz"))
    if labeled_only:
        labeled_ids = {"O012791", "O013257", "O013490"}
        filepaths = []
        for fp in all_filepaths:
            stem = os.path.splitext(os.path.basename(fp))[0]
            if stem in labeled_ids:
                filepaths.append(fp)
    elif filepaths is None:
        filepaths = all_filepaths
    else:
        converted = []
        for fp in filepaths:
            if os.path.isabs(fp):
                converted.append(fp)
            else:
                converted.append(os.path.join(data_dir, fp))
        filepaths = converted

    if len(filepaths) == 0:
        raise ValueError("No target npz files found after filtering. ") 

    print("Loading the saved model")
    # initialize the autoencoder class
    model = Autoencoder(patch_size=config["data"]["patch_size"], **config["autoencoder"])
    # tell PyTorch to load the model onto the CPU if no GPU is available
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"
    # load model weights
    weights = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(weights)
    # put the model in evaluation mode
    model.eval()

    print("Making the patch data")
    images_long, patches = make_data(
        patch_size=config["data"]["patch_size"],
        filepaths=filepaths
        )
    image_codes = []
    for fp in filepaths:
        image_codes.append(os.path.splitext(os.path.basename(fp))[0])

    print("Obtaining embeddings")
    # get the embedding for each patch
    embeddings = [] # what we will save
    for i in tqdm(range(len(images_long))):
        # to make this faster, we use torch.no_grad() to disable gradient tracking
        with torch.no_grad():
            # get the embedding of array of patches
            emb = model.embed(torch.tensor(np.array(patches[i])))
            # NOTE: if your model is quite big, you may not be able to fit
            # all of the data into the GPU memory at once for inference.
            # In that case, you can loop over smaller bathches of data.

            # in the following line we:
            # - detach the tensor from the computation graph
            # - move it to the cpu
            # - turn it into a numpy array
            emb = emb.detach().cpu().numpy()

        embeddings.append(emb)

    print("Saving the embeddings")
    # save the embeddings as csv
    for i in tqdm(range(len(images_long))):
        ae_cols = []
        for j in range(8):
            ae_cols.append(f"ae{j}")
        embedding_df = pd.DataFrame(embeddings[i], columns=ae_cols)
        embedding_df["image_id"] = f"{image_codes[i]}"
        embedding_df["y"] = images_long[i][:, 0]
        embedding_df["x"] = images_long[i][:, 1]
        # move image_id, y and x to front
        cols = embedding_df.columns.tolist()
        cols = [cols[-3], cols[-2], cols[-1]] + cols[:-3]
        embedding_df = embedding_df[cols]
        # save to csv
        embedding_df.to_csv(f"{output_dir}/{image_codes[i]}_ae.csv", index=False)


if __name__ == "__main__":
    config_path = sys.argv[1]
    weights_path = sys.argv[2]
    get_embedding(config_path, weights_path)
