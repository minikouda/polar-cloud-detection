# EXAMPLE USAGE
# python run_autoencoder.py configs/default.yaml

import gc
import os
import sys

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from data import make_data
from patchdataset import PatchDataset


def main():
    print("loading config file")
    if len(sys.argv) < 2:
        raise ValueError("Please provide a config file path.")

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    L.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)

    gc.collect()
    torch.cuda.empty_cache()

    print("making the patch data")
    # Use only labeled images for finetuning
    # Otherwise exclude them for pretraining
    only_labeled = config["data"].get("only_labeled_images", False)

    _, patches = make_data(
        patch_size=config["data"]["patch_size"],
        max_images=config["data"].get("max_images"),
        max_patches_per_image=config["data"].get("max_patches_per_image"),
        exclude_labeled_images=not only_labeled,
        only_labeled_images=only_labeled,
    )

    n_images = len(patches)
    if n_images < 2:
        raise ValueError("Need at least two images to create train and val splits.")

    # Split by image so validation uses unseen scenes
    img_idx = rng.permutation(n_images)
    n_train_img = int(round(0.8 * n_images))
    n_train_img = min(n_images - 1, max(1, n_train_img))

    train_img_idx = img_idx[:n_train_img]
    val_img_idx = img_idx[n_train_img:]

    train_patches = [p for i in train_img_idx for p in patches[i]]
    val_patches = [p for i in val_img_idx for p in patches[i]]

    print(
        f"train val split by images: {len(train_img_idx)} train images, "
        f"{len(val_img_idx)} val images, "
        f"{len(train_patches)} train patches, "
        f"{len(val_patches)} val patches"
    )

    # Optionally cap the total number of patches
    max_patches = config["data"].get("max_patches")
    if max_patches is not None:
        max_train = int(round(0.8 * max_patches))
        max_val = int(round(0.2 * max_patches))

        if len(train_patches) > max_train:
            print(f"subsampling train patches to {max_train}")
            chosen = rng.choice(len(train_patches), size=max_train, replace=False)
            train_patches = [train_patches[i] for i in chosen]

        if len(val_patches) > max_val:
            print(f"subsampling val patches to {max_val}")
            chosen = rng.choice(len(val_patches), size=max_val, replace=False)
            val_patches = [val_patches[i] for i in chosen]

    train_dataset = PatchDataset(train_patches)
    val_dataset = PatchDataset(val_patches)

    dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
    dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

    print("initializing model")
    # Build the autoencoder
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    # Load pretrained weights if a checkpoint is provided
    resume_path = config.get("checkpoint_resume")
    if resume_path:
        if not os.path.isabs(resume_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            resume_path = os.path.join(config_dir, "..", resume_path)

        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"checkpoint_resume not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"loaded weights from {resume_path}")

    print(model)

    print("preparing for training")
    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    if "SLURM_JOB_ID" in os.environ:
        config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

    log_dir = config["trainer"].get("log_dir", "lightning_logs")
    csv_logger = CSVLogger(save_dir=log_dir, name="ae")

    loggers = [csv_logger]
    try:
        wandb_logger = WandbLogger(config=config, **config["wandb"])
        loggers.append(wandb_logger)
    except ModuleNotFoundError:
        print("wandb is not installed so training will continue with CSV logging only")

    trainer = L.Trainer(
        logger=loggers,
        callbacks=[checkpoint_callback],
        **config["trainer"],
    )

    print("training")
    trainer.fit(
        model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()