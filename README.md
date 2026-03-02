# Senior Design — Lung Nodule Detection & Segmentation

A 3D deep learning pipeline that detects whether a lung CT scan contains a cancer nodule (detection) and — if detected — segments and highlights the nodule region (segmentation).

The system is trained on 3D CT patches extracted from the LUNA16 dataset and performs full-volume inference using a sliding-window approach.

---

## Project Overview

This project consists of two main deep learning models trained on 3D CT patches:

- **Detection (`detection_model.py`)**  
  A 3D CNN classifier that predicts whether a 3D CT patch contains a lung nodule (benign vs malignant / nodule vs non-nodule).

- **Segmentation (`segmentation_model.py`)**  
  A 3D U-Net–style model that segments the lung nodule within a positive CT patch.

- **Pipeline (`main.py`)**  
  Runs full-scan inference:  
  1. Loads a CT volume  
  2. Resamples and normalizes it  
  3. Slides a 3D window through the scan  
  4. Detection model classifies each patch  
  5. Positive patches → segmentation model  
  6. Aggregates predicted masks into full-volume nodule map  
  7. Visualizes and saves results  

- **Utilities**
  - `pre_processing.py` — CT resampling, normalization, patch extraction, dataset creation  
  - `helper.py` — Dice coefficient, Dice+BCE loss, confusion matrix plotting  
  - `demo.py` — example visualization of nodules from annotations  

---

## Dataset

Dataset source: https://zenodo.org/records/3723295  

Locally used subsets:
- `subset0`
- `subset1`  

(These are not included in GitHub due to size.)

### Dataset Contents

Each CT scan consists of:

- `.mhd` — metadata header  
- `.raw` — volumetric CT data  

CSV files:

- `annotations.csv` — nodule coordinates and diameter  
- `candidates.csv` — candidate locations and class label  
  - `1` → contains nodule  
  - `0` → no nodule  

---

## Preprocessing

`pre_processing.py` prepares training data by converting full CT scans into 3D patches.

Processing steps:

- Resample CT volume to 1 mm isotropic spacing  
- Clip HU values (−1000 to 400)  
- Normalize intensities to [0, 1]  
- Extract 3D patches centered at candidate coordinates  
- Generate spherical masks for nodules  
- Save patches as `.npy` files  

Output folders:

- data/
- pos/ ← positive patches
- neg/ ← negative patches
- mask/ ← segmentation masks


Positive and negative samples are balanced during extraction.  

---

## Detection Model

`detection_model.py` implements a 3D CNN classifier:

Architecture:

- 3D Conv → ReLU → MaxPool ×3  
- Fully connected layers  
- Dropout regularization  
- Softmax output (2 classes)

Input: 3D patch (32×32×32)  
Output: nodule present / absent  

Training:

- Loss: Cross-Entropy  
- Optimizer: Adam  
- Dataset: `data/pos` and `data/neg` patches  
- Train/val/test split: 80/10/10  

Evaluation metrics:

- Accuracy  
- Precision  
- Recall  
- F1 score  
- Confusion matrix  

---

## Segmentation Model

`segmentation_model.py` implements a lightweight 3D U-Net:

Architecture:

- Encoder: Conv blocks + MaxPool  
- Bottleneck  
- Decoder: Upsampling + skip connections  
- Final 3D conv → segmentation mask  

Input: positive CT patch  
Output: voxel-wise nodule mask  

Training:

- Loss: Dice + BCE  
- Metric: Dice coefficient  
- Dataset: `data/pos` and `data/mask`  
 

---

## Inference Pipeline

`main.py` performs full-volume lung nodule detection:

Steps:

1. Load CT `.mhd` scan  
2. Resample and normalize volume  
3. Slide 32³ patch window through scan  
4. Detection model classifies each patch  
5. Positive patches → segmentation model  
6. Merge patch masks into full 3D mask  
7. Save mask as `.mhd`  
8. Visualize slice with strongest prediction  

Output:

- `prediction_result.mhd` — full 3D predicted mask  
- `prediction_viz.png` — overlay visualization  

---

## How to Setup and Run

### 1. Download Dataset

Download the following files from the dataset website:  
https://zenodo.org/records/3723295  

Required files:

- `subset0.zip`
- `subset1.zip`
- `annotations.csv`
- `candidates.csv`

Extract the subsets and place everything in the **project root directory**:

- subset0/
- subset1/
- annotations.csv
- candidates.csv
- pre_processing.py
- detection_model.py
- segmentation_model.py
- main.py

---

### 2. Generate Training Data (Patches)

Run the preprocessing script to create patch datasets:

```bash
python pre_processing.py
```

This creates:

data/
 - pos/   ← positive patches
 - neg/   ← negative patches
 - mask/  ← segmentation masks
  
### 3. Train Detection Model

```bash
python detection_model.py
```

Output:

- nodule_detection_model.pth

### 4. Train Segmentation Model

```bash
python segmentation_model.py
```

Output:

- nodule_segmentation_model.pth

### 5. Run Full Inference Pipeline

```bash 
python main.py
```

The pipeline will:

- Load a CT scan
- Run detection + segmentation
- Save prediction mask (prediction_result.mhd)
- Save visualization (prediction_viz.png)

You can change which CT scan is analyzed by editing the file path in main.py:

```bash test_file = "subset0/subset0/<SCAN_ID>.mhd" ```
