from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import random
import cv2
import torch


def window_image(img, center=50, width=150):
    low = center - width // 2
    high = center + width // 2
    return np.clip(img, low, high)

# random patches generation for traning process
class VLR_LR_Dataset(Dataset):
    def __init__(self, vlr_path, lr_path, patch_size=64, transforms=None):
        """
        Training dataset: returns random patches for each __getitem__,
        so that the model sees different samples at every epoch.

        vlr_path: very low resolution slice
        lr_path: low resolution slice
        """
        self.vlr = nib.load(vlr_path).get_fdata().astype(np.float32)
        self.lr = nib.load(lr_path).get_fdata().astype(np.float32)

        if self.vlr.ndim == 3 and self.vlr.shape[2] == 1:
            self.vlr = self.vlr[:,:,0]
        if self.lr.ndim == 3 and self.lr.shape[2] == 1:
            self.lr = self.lr[:,:,0]
        
        assert self.vlr.ndim == 2
        assert self.lr.ndim  == 2

        # IVLR: upscale VLR to LR size with bicubic iterpolation
        h, w = self.lr.shape
        self.ivlr = cv2.resize(self.vlr, (w, h), interpolation=cv2.INTER_CUBIC)

        self.patch_size = patch_size

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        ph   = self.patch_size
        h, w = self.lr.shape

        #random extraction
        x = random.randint(0, h - ph) # vertical axis
        y = random.randint(0, w - ph) # horizontal axis

        ivlr_patch = self.ivlr[x:x+ph, y:y+ph]
        lr_patch = self.lr[x:x+ph, y:y+ph]

        #windowing
        ivlr_patch = window_image(ivlr_patch)
        lr_patch   = window_image(lr_patch)

        #normalization [0,1]
        amin = lr_patch.min()
        amax = lr_patch.max()
        if amax > amin:
            ivlr_patch = (ivlr_patch - amin) / (amax - amin)
            lr_patch   = (lr_patch   - amin) / (amax - amin)

        #tensors
        ivlr_t = torch.from_numpy(ivlr_patch).unsqueeze(0).float() # (h, w) --> (1, h, w) 1 == n_channels
        lr_t   = torch.from_numpy(lr_patch).unsqueeze(0).float()

        return ivlr_t, lr_t

# fixed patches generation for validation process
class FixedVLR_LR_Dataset(Dataset):
    """
    Validation dataset: generates a fixed set of patches ONCE
    during initialization and keeps them in RAM.
    """

    def __init__(self, vlr_path, lr_path, patch_size=64, num_patches=500):
        self.vlr = nib.load(vlr_path).get_fdata().astype(np.float32)
        self.lr  = nib.load(lr_path).get_fdata().astype(np.float32)

        if self.vlr.ndim == 3 and self.vlr.shape[2] == 1:
            self.vlr = self.vlr[:, :, 0]
        if self.lr.ndim == 3 and self.lr.shape[2] == 1:
            self.lr = self.lr[:, :, 0]

        assert self.vlr.ndim == 2
        assert self.lr.ndim == 2

        h, w = self.lr.shape
        self.ivlr = cv2.resize(self.vlr, (w, h), interpolation=cv2.INTER_CUBIC)

        self.patch_size = patch_size
        self.num_patches = num_patches

        # preload fixed patches into RAM
        self.data = []
        for _ in range(num_patches):
            self.data.append(self._extract_patch())

    def _extract_patch(self):
        ph = self.patch_size
        h, w = self.lr.shape

        x = random.randint(0, h - ph)
        y = random.randint(0, w - ph)

        ivlr_patch = self.ivlr[x:x+ph, y:y+ph]
        lr_patch   = self.lr[x:x+ph, y:y+ph]

        ivlr_patch = window_image(ivlr_patch)
        lr_patch   = window_image(lr_patch)

        amin, amax = lr_patch.min(), lr_patch.max()
        if amax > amin:
            ivlr_patch = (ivlr_patch - amin) / (amax - amin)
            lr_patch   = (lr_patch   - amin) / (amax - amin)

        ivlr_t = torch.from_numpy(ivlr_patch).unsqueeze(0).float()
        lr_t   = torch.from_numpy(lr_patch).unsqueeze(0).float()

        return ivlr_t, lr_t

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        return self.data[idx]
