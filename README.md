&nbsp;

<div align="center">

[Paper] | [Project Page] | [Models] | [Data]

<h2>REG-Gaussians: 4D-Gaussian Splatting for Sparse-View Dynamic CBCT Reconstruction</h2>

*Regularized 4D Gaussian splatting for sparse-view dynamic cone-beam CT reconstruction.*

</div>

&nbsp;

## Introduction

This repository contains the official implementation of **REG-Gaussians**, a framework for **sparse-view 4D cone-beam CT (4D-CBCT) reconstruction** based on **4D Gaussian splatting**.

REG-Gaussians extends recent radiative Gaussian splatting methods for tomographic reconstruction by introducing and evaluating multiple **regularization strategies** aimed at improving reconstruction quality and temporal consistency in dynamic CBCT settings. The framework focuses on **respiratory phase binned 4D-CBCT** and is evaluated on thoracic data.

The work builds upon and adapts existing 3D and 4D Gaussian splatting pipelines, most notably **R²-Gaussian** and **HexPlane-based spatiotemporal representations**, and evaluates the effect of additional regularization terms such as:
- HexPlane grid smoothing
- Motion coherence regularization
- Geometric regularization

---

## 1. Installation

We recommend using **Conda** to set up the environment. The code has been tested on **Ubuntu 20.04** and **Windows (WSL2)** with NVIDIA GPUs.

### 1.1 Clone repository

```sh
#Clone the repository
git clone https://github.com/JorisMentink/REG-gaussians.git --recursive
cd REG-gaussians

#Install environment
conda env create --file environment.yml
conda activate reg_gaussians
```

## 2. Dataset

REG-Gaussians is evaluated on **4D thoracic CT data** with simulated sparse-view CBCT projections.

**Primary dataset**
- DIR-Lung 4DCT dataset
- Patients 1 to 5

The dataset can be downloaded from the official source:
https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html

After downloading, the raw 4DCT data must be pre-processed and converted into a NeRF-style directory structure before training.

---

## 3. Initialization and Training

Initialization and training follow the same general pipeline as **R²-Gaussian**.  
Please refer to the original repository for additional background details:
https://github.com/Ruyi-Zha/r2_gaussian

### 3.1 Dataset compilation

After pre-processing the raw data into a NeRF-format directory structure, all respiratory phase bins are compiled into a single training dataset using:

```sh
python compile_training_dataset.py   --input_path  "path/to/parent/folder"   --output_path "path/to/output/folder"   --train_size  30   --test_size   30
```

Here, `train_size` and `test_size` denote the number of projections used for training and testing per respiratory phase, respectively.

---

### 3.2 Gaussian initialization

The compiled dataset is initialized by sampling Gaussians from a coarse reconstruction, typically obtained using the FDK algorithm:

```sh
python initialize_pcd.py   --data   "path/to/compiled/dataset"   --device 0
```

Proper initialization is important for stable convergence and reconstruction quality.

---

### 3.3 Training

Training can be started using:

```sh
python train.py   --source_path "path/to/compiled/dataset"   --iterations  30000   --model_path "path/to/output"
```

The number of training iterations and output directory can be adjusted depending on the desired reconstruction quality and available computational resources.


