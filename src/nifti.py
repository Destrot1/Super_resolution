import os
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.logger import configure_logging
import logging


logger = logging.getLogger(__name__)


def save_slice(file_path, slice_idx, save_dir=None):
    """
    save only one HR slice from the entire volume
    """
    img = nib.load(file_path)
    vol_data = img.get_fdata()
    logger.info(f"Nifti shape = {vol_data.shape}")

    if slice_idx<0 or slice_idx > vol_data.shape[2]:
        logger.error("Error in slice idx")
        raise ValueError("error in slice idx")
    
    # extract slice
    slice_data = vol_data[:,:, slice_idx]
    
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    slice_img = nib.Nifti1Image(slice_data, affine=img.affine) # maintain affine matrix
    save_path = os.path.join(save_dir, "HR_slice.nii.gz")
    nib.save(slice_img, save_path)
    logger.info(f"Slice saved in RAS orientation at: {save_path}")

    
def window_image(slice_data, window_center, window_width):
    """
    apply windowing and normalization for better GUI visualization
    """
    img_min = window_center - window_width/2
    img_max = window_center + window_width/2
    slice_windowed = np.clip(slice_data, img_min, img_max)
    slice_norm = (slice_windowed - slice_windowed.min()) / (slice_windowed.max() - slice_windowed.min())

    return slice_windowed, slice_norm


def view_slice(file_path,resolution):
    """
    visualize saved slice.nii
    """
    img = nib.load(file_path)
    slice_data = img.get_fdata()

    window_center = 50
    window_width = 150

    slice_windowed, slice_norm = window_image(slice_data, window_center, window_width)

    plt.figure(figsize=(6,6))
    plt.imshow(slice_norm.T, cmap="gray", origin='lower')
    plt.title(f"Slice {resolution}", fontweight='bold')
    plt.axis('off')
    plt.show()


def degrade_resolution(hr_slice_path, save_path, scale=2, blur_kernel=(5,5), sigma=1.5):
    """
    generate low-resolution slice:
    gaussian-blur (simulates the natural blur of acquisition systems)
    +
    downsampling bicubic (simulates low scanner resolution: less pixels)
    """
    img = nib.load(hr_slice_path)
    hr_slice = img.get_fdata()
    hr_slice = hr_slice.astype(np.float32)
    blurred = cv2.GaussianBlur(hr_slice, ksize=blur_kernel, sigmaX=sigma)

    hr_h, hr_w = hr_slice.shape
    lr_h, lr_w = hr_h // scale, hr_w // scale

    # downsampling
    lr_slice = cv2.resize(blurred, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    lr_image = nib.Nifti1Image(lr_slice, affine=img.affine) # maintain affine matrix
    nib.save(lr_image, save_path)
    logger.info(f"Degraded slice saved in: {save_path}")


def view_inference_slices(input_slice, output_slice, target_slice, title="Inference", window_center=50, window_width=150, save_path = None):
    # windowing and normalization
    _, input_norm  = window_image(input_slice,  window_center, window_width)
    _, output_norm = window_image(output_slice, window_center, window_width)
    _, target_norm = window_image(target_slice, window_center, window_width)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow(input_norm.T,  cmap="gray", origin='lower')
    axes[0].set_title("Input", fontweight='bold')
    axes[1].imshow(output_norm.T, cmap="gray", origin='lower')
    axes[1].set_title("Output", fontweight='bold')
    axes[2].imshow(target_norm.T, cmap="gray", origin='lower')
    axes[2].set_title("Target", fontweight='bold')

    for ax in axes:
        ax.axis('off')

    plt.suptitle(title, fontweight='bold')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()



    
   
