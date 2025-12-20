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

If you find this repository useful, please consider giving it a star ⭐.

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

#### 1.2 Install TIGRE

It is necessary to download the TIGRE CT toolbox for data data generation and initialization. This can be done via their instructions HERE: https://github.com/CERN/TIGRE/blob/master/Frontispiece/python_installation.md

## 2. Dataset

REG-Gaussians is evaluated on 4D thoracic CT data with simulated sparse view CBCT projections.

Primary dataset:
DIR Lung 4DCT, Patients 1 to 5, downloadable HERE: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html

## 3. Initilization and training.

Initialization and training happens via the same methods as R2_gaussian, seen HERE: https://github.com/Ruyi-Zha/r2_gaussia

After pre-processing the raw data into NeRF format file structure, compile all processed phasebins using

```sh
python compile_training_dataset.py
--input_path    "path/to/parent/folder"
--output_path   "path/to/output/folder"
--train_size    30
--test_zie      30
```

Then, initialize the compiled dataset using:

```sh
python initialize_pcd.py
--data    "path/to/compiled/dataset"
--device  0
```

After which you can start training using:

```sh
python train.py "path/to/compiled/dataset"
--iterations 30000
--model_path "path/to/output"

```



