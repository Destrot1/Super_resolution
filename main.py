import os
import logging
from torch.utils.data import DataLoader, random_split
import torch
import nibabel as nib
import numpy as np
from src.logger import configure_logging
from src.nifti import save_slice, view_slice, degrade_resolution, view_inference_slices
from src.vlr_lr_dataset import VLR_LR_Dataset, FixedVLR_LR_Dataset
from src.build_model import ResidualSRNet
from src.train import train_model
from src import inference

SLICE_IDX    = 8 # select which slice to extract from the volume
DATA_DIR     = "data"
HR_PATH      = os.path.join(DATA_DIR, "HR_slice.nii.gz")
LR_PATH      = os.path.join(DATA_DIR, "LR_slice.nii.gz")
VLR_PATH     = os.path.join(DATA_DIR, "VLR_slice.nii.gz")
WEIGHTS_PATH = os.path.join("model_weights", "weights.pth")

PATCH_SIZE  = 64
BATCH_SIZE  = 8
NUM_WORKERS = 0 # multiprocessing: disabled

NUM_CHANNELS = 64 # n_features_maps
NUM_LAYERS   = 5 # total --> 3 hidden layers

NUM_EPOCHS = 20
LR = 1e-4 # initial learning rate

def main():

    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
        
    # create model
    model = ResidualSRNet(num_channels=NUM_CHANNELS, num_layers=NUM_LAYERS).to(device)
    print(model)


    # extract slice
    save_slice(file_path = os.path.join(DATA_DIR, "HR_volume.nii.gz"), slice_idx=SLICE_IDX, save_dir=DATA_DIR)

    # create low and very-low-resolution slices
    degrade_resolution(hr_slice_path = HR_PATH, save_path = LR_PATH)  #low-resolution
    degrade_resolution(hr_slice_path = LR_PATH, save_path = VLR_PATH) #very-low-resolution
    # view_slice(file_path = os.path.join(DATA_DIR, "HR_slice.nii.gz"), resolution="High Resolution")
    # view_slice(file_path = os.path.join(DATA_DIR, "LR_slice.nii.gz"), resolution="Low Resolution")
    # view_slice(file_path = os.path.join(DATA_DIR, "VLR_slice.nii.gz"), resolution="Very Low Resolution")

    # create training dataset (random patches every epoch)
    train_dataset = VLR_LR_Dataset(
        vlr_path=VLR_PATH,
        lr_path = LR_PATH,
        patch_size = PATCH_SIZE
    )

    # create validation dataset (fixed patches precomputed only once)
    val_dataset = FixedVLR_LR_Dataset(
        vlr_path = VLR_PATH,
        lr_path = LR_PATH,
        patch_size = PATCH_SIZE
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # training and weights saving
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=NUM_EPOCHS, lr=LR, save_path=WEIGHTS_PATH)

    # inference
    # first step (classic inference)
    vlr_slice = nib.load(VLR_PATH).get_fdata().astype(np.float32) # input
    lr_slice  = nib.load(LR_PATH).get_fdata().astype(np.float32)  # target

    output_slice, metrics = inference.infer_and_compare(model, vlr_slice, lr_slice, device)
    logger.info(f"Metrics: MSE={metrics['MSE']:.6f}, PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}")

    view_inference_slices(vlr_slice, output_slice, lr_slice, title="Classic Inference", save_path="results/classic_inference.png")


    # second step (2x super resolution)
    """
    Super-resolution evaluation using a prior target as input.
    Here, we assess the model's ability to perform 2x super-resolution on an image
    that was not seen during training. The previous LR target is used as input, 
    and the model's output is compared against a high-resolution reference (HR slice)
    to measure performance metrics (MSE, PSNR, SSIM)
    """
    prev_target = lr_slice
    hr_slice    = nib.load(HR_PATH).get_fdata().astype(np.float32)

    output_slice2, metrics2 = inference.infer_target_as_input(model, prev_target, hr_slice, device)
    logger.info(f"Metrics2: MSE={metrics2['MSE']:.6f}, PSNR={metrics2['PSNR']:.2f}, SSIM={metrics2['SSIM']:.4f}")

    view_inference_slices(prev_target, output_slice2, hr_slice, title="2x Super Resolution", save_path="results/2x_super_resolut")


if __name__ == "__main__":

    logger = configure_logging(log_file="train.log")
    logger = logging.getLogger(__name__)
    logger.info("Program started")

    main()