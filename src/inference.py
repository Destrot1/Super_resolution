import torch 
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def infer_slice(model, input_slice, device, target_shape):
    """
    input_slice: np.array float32 (H,W)
    """
    model.eval()
    input_slice = cv2.resize(input_slice, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

    input_tensor = torch.from_numpy(input_slice).unsqueeze(0).unsqueeze(0).float().to(device) # (1,1,H,W)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_slice = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
    return output_slice


def compare_slices(output, target, display=False):
    """
    compare output with target using MSE, PSNR, SSIM
    """
    mse_val = np.mean((output-target)**2)
    psnr_val = psnr(target, output, data_range=target.max() - target.min())
    ssim_val = ssim(target, output, data_range=target.max() - target.min())

    if display:
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(output, cmap='gray'); axs[0].set_title("Output")
        axs[1].imshow(target, cmap='gray'); axs[1].set_title("Target")
        axs[2].imshow(np.abs(output-target), cmap='hot'); axs[2].set_title("Abs Diff")
        for ax in axs: ax.axis('off')
        plt.show()
    
    return {'MSE': mse_val, 'PSNR': psnr_val, 'SSIM': ssim_val}


def infer_and_compare(model, input_slice, target_slice, device, display=True):
    output_slice = infer_slice(model, input_slice, device, target_shape=target_slice.shape)
    metrics = compare_slices(output_slice, target_slice, display=display)
    return output_slice, metrics


def infer_target_as_input(model, previous_target, new_target, device, display=True):
    """
    use the previous target as input to model and compare with new target
    """
    output_slice = infer_slice(model, previous_target, device, target_shape=new_target.shape)
    metrics = compare_slices(output_slice, new_target, display=display)
    return output_slice, metrics


    

