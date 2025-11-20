# SUPER RESOLUTION

**Transforming Images, Elevating Clarity Instantly**

## **Table of Contents**
1. [Overview](#overview)  
2. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
3. [Usage](#usage)  
4. [Testing](#testing)  
5. [Project Structure](#project-structure)  

## **Overview**
Super Resolution is a comprehensive developer tool designed to enhance medical images by increasing their resolution through advanced super-resolution techniques. It provides an end-to-end pipeline for training, validating, and deploying models tailored for medical imaging data, particularly volumetric scans.

### **Why Super_resolution?**
This project aims to improve image quality in medical diagnostics and research. The core features include:

- **Model Building & Training:** Seamlessly define, train, and evaluate super-resolution neural networks optimized for medical images.  
- **Data Preparation & Augmentation:** Efficiently generate and manage low- and very-low-resolution image patches from volumetric scans.  
- **Image Processing Utilities:** Utilities for visualizing, degrading, and comparing NIfTI slices to assess model performance.  
- **Inference & Evaluation:** Perform high-quality image prediction with metrics like PSNR and SSIM, supporting iterative improvements.  
- **Modular Architecture & Logging:** Well-structured codebase with comprehensive logging for debugging and monitoring.  

## **Getting Started**

### **Prerequisites**
- **Programming Language:** Python 3.10+  
- **Package Manager:** Conda  
- **Dependencies:** PyTorch, torchvision, nibabel, numpy, opencv-python, matplotlib, scikit-image, tensorboard, pytest  

### **Installation**
1. **Clone the repository:**  
```bash
git clone https://github.com/Destrot1/Super_resolution.git
```
2. **Navigate to the project directory:**  
```bash
cd Super_resolution
```
3. **Create and activate the Conda environment:**  
```bash
conda env create -f environment.yml
conda activate sr_projects
```

## **Usage**

**1. Train the model and monitor training:**  
Esegui il training del modello:
```bash
python main.py
```
Visualizza le metriche in tempo reale con TensorBoard:
```bash
tensorboard --logdir=runs
```

**2. Run classic inference on a slice:**  
```python
import nibabel as nib
from src import inference
from src.nifti import view_inference_slices

vlr_slice = nib.load("data/VLR_slice.nii.gz").get_fdata().astype(np.float32)
lr_slice  = nib.load("data/LR_slice.nii.gz").get_fdata().astype(np.float32)

output_slice, metrics = inference.infer_and_compare(model, vlr_slice, lr_slice, device)
print(metrics)

view_inference_slices(vlr_slice, output_slice, lr_slice, title="Classic Inference")
```

**3. Run 2x super-resolution using previous target as input:**  
```python
hr_slice = nib.load("data/HR_slice.nii.gz").get_fdata().astype(np.float32)

output_slice2, metrics2 = inference.infer_target_as_input(model, lr_slice, hr_slice, device)
print(metrics2)

view_inference_slices(lr_slice, output_slice2, hr_slice, title="2x Super-Resolution")
```

## **Testing**
Run the project tests to verify functionality. Make sure the Conda environment is active.

1. **Activate the Conda environment:**  
```bash
conda activate sr_projects
```

2. **Run the test suite:**  
```bash
pytest
```

3. **View results:**  
- The tests will show any errors or failures in the console.  
- Add the `-v` flag for verbose output:
```bash
pytest -v
```

**Note:**  
Tests include checks for dataset loading, model inference, and metric computation (MSE, PSNR, SSIM). Ensure that `data/`, `runs/`, and required files are present before running tests.

## **Project Structure**
The project is organized to be modular and clear. Here is an overview of the main folders and files:

```
Super_resolution/
├── src/                     # Main source code
│   ├── train.py             # Functions for model training
│   ├── inference.py         # Functions for inference and evaluation
│   ├── nifti.py             # Utilities for handling NIfTI files and visualization
│   └── __pycache__/         # Compiled Python files (ignored by git)
├── data/                    # Folder containing the datasets
│   ├── HR_volume.nii.gz     # High-resolution reference volume
│   ├── VLR_slice.nii.gz     # Very low-resolution slice
│   └── LR_slice.nii.gz      # Low-resolution slice
├── results/                 # Model outputs and saved images
├── runs/                    # TensorBoard logs for training monitoring
├── main.py                  # Main script for training and inference
├── model_weights/           # Saved model weights
├── train.log                # Training session log file
├── environment.yml          # Conda environment configuration
└── README.md                # Project documentation
```

**Quick description:**  
- `src/` contains all the project logic (training, inference, utilities).  
- `data/` hosts slices and volumes used for training and testing.  
- `results/` and `runs/` store outputs and logs, excluded from version control.  
- `main.py` is the entry point for training and inference.  
- `model_weights/` stores trained model weights for future use.
