from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Wrapper dataset class for our datasets of patches
    """

    def __init__(
        self,
        patches=None,
        transform=None,
        images_padded=None,
        coords=None,
        indices=None,
        patch_size=9,
    ):
        self.patches = patches
        self.transform = transform
        self.images_padded = images_padded
        self.coords = coords
        self.indices = indices
        self.pad_len = patch_size // 2
        # self.transform is for any torchvision transforms
        # you want to apply to the patches (for data augmentation)

    def __len__(self):
        if self.patches is not None:
            return len(self.patches)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.patches is not None:
            sample = self.patches[idx]
            if self.transform:
                sample = self.transform(sample)
            return sample

        real_idx = self.indices[idx]
        image_idx, y_rel, x_rel = self.coords[real_idx]
        return int(image_idx), int(y_rel), int(x_rel)
