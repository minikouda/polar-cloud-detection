# EXAMPLE USAGE:
# python get_embedding.py configs/default.yaml checkpoints/default-epoch=009.ckpt

import sys
import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from autoencoder import Autoencoder
from data import make_data


config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

config = yaml.safe_load(open(config_path, "r"))

print("Loading model")
model = Autoencoder(
    patch_size=config["data"]["patch_size"],
    **config["autoencoder"]
)
map_location = None if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(checkpoint_path, map_location=map_location)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

print("Making patch data")
images_long, patches = make_data(
    patch_size=config["data"]["patch_size"],
    max_images=config["data"].get("max_images"),
    max_patches_per_image=None,
)

print("Obtaining embeddings")
embeddings = []

for i in tqdm(range(len(images_long))):
    with torch.no_grad():
        emb = model.embed(torch.tensor(np.array(patches[i])))
        emb = emb.detach().cpu().numpy()

    embeddings.append(emb)

print("Saving embeddings")
save_indices = config.get("save_image_indices")
if save_indices is not None:
    print(f"Only saving CSVs for image indices: {save_indices}")

for i in tqdm(range(len(images_long))):
    if save_indices is not None and (i + 1) not in save_indices:
        continue

    n_emb = config["autoencoder"]["embedding_size"]
    embedding_df = pd.DataFrame(
        embeddings[i],
        columns=[f"ae{j}" for j in range(n_emb)]
    )
    embedding_df["y"] = images_long[i][:, 0]
    embedding_df["x"] = images_long[i][:, 1]

    cols = embedding_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    embedding_df = embedding_df[cols]

    embedding_df.to_csv(f"../data/image{i+1}_ae.csv", index=False)