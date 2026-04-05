# EXAMPLE USAGE:
# python run_autoencoder.py configs/default.yaml

import numpy as np
import sys
import os
import yaml  # pip install pyyaml
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
# pip install torchinfo
# from torchinfo import summary

from autoencoder import Autoencoder
from patchdataset import PatchDataset
from data import make_data_on_the_fly

def main(config_path):
    print("loading config file")
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    config = yaml.safe_load(open(config_path, "r"))

    # set random seeds for reproducibility
    seed = config.get("seed", 0)
    L.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    print("making the patch data")
    # prepare data for on-the-fly patch extraction
    data_mode = config.get("data", {}).get("mode", "all")
    _, images_padded, coords, image_codes = make_data_on_the_fly(
        patch_size=config["data"]["patch_size"],
        mode=data_mode,
    )

    # do train/val/test split by image
    image_ids = np.unique(coords[:, 0])
    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.8)
    test_ratio = split_cfg.get("test_ratio", 0.0)
    if "val_ratio" in split_cfg:
        val_ratio = split_cfg["val_ratio"]
    else:
        val_ratio = 1.0 - train_ratio - test_ratio

    shuffled_image_ids = np.random.permutation(image_ids)
    n_train_images = int(len(shuffled_image_ids) * train_ratio)
    n_val_images = int(len(shuffled_image_ids) * val_ratio)

    train_image_ids = shuffled_image_ids[:n_train_images]
    val_image_ids = shuffled_image_ids[n_train_images : n_train_images + n_val_images]
    test_image_ids = shuffled_image_ids[n_train_images + n_val_images :]

    train_idx = np.where(np.isin(coords[:, 0], train_image_ids))[0]
    val_idx = np.where(np.isin(coords[:, 0], val_image_ids))[0]
    test_idx = np.where(np.isin(coords[:, 0], test_image_ids))[0]

    # create train and val datasets
    train_dataset = PatchDataset(
        coords=coords,
        indices=train_idx,
        patch_size=config["data"]["patch_size"],
    )
    has_val = len(val_idx) > 0
    if has_val:
        val_dataset = PatchDataset(
            coords=coords,
            indices=val_idx,
            patch_size=config["data"]["patch_size"],
        )

    # create train and val dataloaders
    train_loader_kwargs = dict(config["dataloader_train"])
    val_loader_kwargs = dict(config["dataloader_val"])

    if train_loader_kwargs.get("num_workers", 0) > 0:
        train_loader_kwargs["persistent_workers"] = True
    if has_val and val_loader_kwargs.get("num_workers", 0) > 0:
        val_loader_kwargs["persistent_workers"] = True
    if torch.cuda.is_available():
        train_loader_kwargs["pin_memory"] = True
        if has_val:
            val_loader_kwargs["pin_memory"] = True

    dataloader_train = DataLoader(
        train_dataset, **train_loader_kwargs
    )
    dataloader_val = None
    if has_val:
        dataloader_val = DataLoader(
            val_dataset, **val_loader_kwargs
        )

    print("initializing model")
    # Initialize an autoencoder object
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        images_padded=images_padded,
        **config["autoencoder"],
    )

    init_checkpoint = config.get("init_checkpoint", None)
    if init_checkpoint:
        print(f"loading init weights: {init_checkpoint}")
        assert os.path.exists(init_checkpoint), f"Model weights file {init_checkpoint} not found"
        map_location = None if torch.cuda.is_available() else "cpu"
        init_obj = torch.load(init_checkpoint, map_location=map_location)
        model.load_state_dict(init_obj)

    print(model)

    # if running in slurm, add slurm job id info to the config file
    if "SLURM_JOB_ID" in os.environ:
        config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

    # initialize the wandb logger, giving it our config file
    # to save, and also configuring the logger itself.
    wandb_logger = WandbLogger(config=config, **config["wandb"])

    # initialize the trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        **config["trainer"],
    )

    # save test image IDs for final evaluation
    exp_name = config["wandb"]["name"]
    save_dir = config.get("output", {}).get("dirpath")
    if save_dir is None:
        save_dir = config.get("checkpoint", {}).get("dirpath", "../results")
    os.makedirs(save_dir, exist_ok=True)
    test_id_path = os.path.join(save_dir, f"{exp_name}_test_ids.txt")
    with open(test_id_path, "w") as f:
        for image_idx in test_image_ids:
            f.write(f"{image_codes[int(image_idx)]}\n")
    print(f"saved test ids: {test_id_path}")

    print("training")
    if has_val:
        trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    else:
        trainer.fit(model, train_dataloaders=dataloader_train)

    # export one .pt file (final model state)
    state_dict = model.state_dict()
    pt_filename = f"{exp_name}.pt"
    pt_path = os.path.join(save_dir, pt_filename)
    torch.save(state_dict, pt_path)
    print(f"saved model: {pt_path}")

    # clean up memory
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python run_autoencoder.py <config_path>")
    main(sys.argv[1])
