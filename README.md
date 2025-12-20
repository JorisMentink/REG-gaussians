&nbsp;

<div align="center">

<h2>REG-Gaussians: 4D-Gaussian Splatting for Sparse-View Dynamic CBCT Reconstruction</h2>

*Regularized 4D Gaussian splatting for sparse-view dynamic cone-beam CT reconstruction.*

[Project page](https://jorismentink.github.io/REG-gaussians/)


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

We recommend using **Conda** to set up the environment. The code has been tested on **Ubuntu 24.04** using the NVIDIA GeForce RTX 4080 Super GPU.

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

The dataset can be downloaded from the official source [here](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct)

After downloading, the raw 4DCT data must be pre-processed and converted into a NeRF-style directory structure before training.

---

## 3. Initialization and Training

Initialization and training follow the same general pipeline as [**R²-Gaussian**](https://github.com/Ruyi-Zha/r2_gaussian).  
Please refer to the original repository for additional background details:


### 3.1 Dataset compilation

After pre-processing the raw data into a NeRF-format directory structure, all respiratory phase bins are compiled into a single training dataset using:

```sh
python compile_training_dataset.py
--input_path  "path/to/parent/folder"
--output_path "path/to/output/folder"
--train_size  30
--test_size   30
```

Here, `train_size` and `test_size` denote the number of projections used for training and testing per respiratory phase, respectively.

---

### 3.2 Gaussian initialization

The compiled dataset is initialized by sampling Gaussians from a coarse reconstruction, typically obtained using the FDK algorithm:

```sh
python initialize_pcd.py
--data   "path/to/compiled/dataset"
--device 0
```

Proper initialization is important for stable convergence and reconstruction quality.

---

### 3.3 Training

Training can be started using:

```sh
python train.py
--source_path "path/to/compiled/dataset"
--iterations  30000
--model_path "path/to/output"
```

The number of training iterations and output directory can be adjusted depending on the desired reconstruction quality and available computational resources.

### 3.3 Evaluation

Model evaluation can be done via:

```sh
python test.py
--model_path "path/to/model/output/folder"
--iteration  30000
```
Where the iteration number denotes at what iteration checkpoint you want to evaluate the model. NOTE: you can only evaluate models at lower iterations if checkpoints exist.

### Acknowledgements

This repository builds upon and adapts code from several open-source projects, including:

- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)  
- [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian)  
- [HexPlane](https://github.com/Caoang327/HexPlane)
- [TIGRE Toolbox](https://github.com/CERN/TIGRE/tree/master)  

We thank the respective authors for making their work publicly available.

---

### License

This project falls under the third party licenses of the used open-source code:
[R²-Gaussian license](https://github.com/Ruyi-Zha/r2_gaussian/blob/main/LICENSE.md)
[HexPlane license](https://github.com/Caoang327/HexPlane/blob/main/LICENSE)


---

