# SEMAMBA - Fusion
this is the implementation of a Speech enhancement algorithm based on mamba architecture, fused with a time domain feature extraction.

---

⚠️  Notice: If you encounter CUDA-related issues while using the Mamba-1 framework, we suggest using the Mamba-2 framework (available in the mamba-2 branch).  
The Mamba-2 framework is designed to support both Mamba-1 and Mamba-2 model structures.

```bash
git checkout mamba-2
```

## Requirement
    * Python >= 3.9
    * CUDA >= 12.0
    * PyTorch == 2.2.2
## Model

## Speech Enhancement Results
![VoiceBand-Demand results](imgs/resluts.png)

## Additional Notes

1. Ensure that both the `nvidia-smi` and `nvcc -V` commands show CUDA version 12.0 or higher to verify proper installation and compatibility.

2. Currently, it supports only GPUs from the RTX series and newer models. Older GPU models, such as GTX 1080 Ti or Tesla V100, may not support the execution due to hardware limitations.

## Installation
### (Suggested:) Step 0 - Create a Python environment with Conda

It is highly recommended to create a separate Python environment to manage dependencies and avoid conflicts.
```bash
conda create --name mamba python=3.9
conda activate mamba
```

### Step 1 - Install PyTorch

Install PyTorch 2.2.2 from the official website. Visit [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) for specific installation commands based on your system configuration (OS, CUDA version, etc.).

### Step 2 - Install Required Packages

After setting up the environment and installing PyTorch, install the required Python packages listed in requirements.txt.

```bash
pip install -r requirements.txt
```

### Step 3 - Install the Mamba Package

Navigate to the mamba_install directory and install the package. This step ensures all necessary components are correctly installed.

```bash
cd mamba_install
pip install .
```

⚠️  Note: Installing from source (provided `mamba_install`) can help prevent package issues and ensure compatibility between different dependencies. It is recommended to follow these steps carefully to avoid potential conflicts.

⚠️  Notice: If you encounter CUDA-related issues while you already have `CUDA>=12.0` and installed `pytorch 2.2.2`, you could try mamba 1.2.0.post1 instead of mamba 1.2.0 as follow:
```bash
cd mamba-1_2_0_post1
pip install .
```


## Training the Model
### Step 1: Prepare Dataset JSON

Create the dataset JSON file using the script `sh make_dataset.sh`. You may need to modify `make_dataset.sh` and `data/make_dataset_json.py`.

Alternatively, you can directly modify the data paths in `data/train_clean.json`, `data/train_noisy.json`, etc.

### Step 2: Run the following script to train the model.

```bash
sh run_mamba_fusion.sh
```

Note: You can use `tensorboard --logdir exp/path_to_your_exp/logs` to check your training log

## Using the Pretrained Model

Modify the `--input_folder` and `--output_folder` parameters in `pretrained_fusion.sh` to point to your desired input and output directories. Then, run the script.

```bash
sh pretrained_fusion.sh
```


# Basic Mamba model trainign
```bash
sh pretrained_basic.sh
```


## References and Acknowledgements
we would like to thank to the original SEMamba paper and github repository on which this project is based on.
```
@article{chao2024investigation,
  title={An Investigation of Incorporating Mamba for Speech Enhancement},
  author={Chao, Rong and Cheng, Wen-Huang and La Quatra, Moreno and Siniscalchi, Sabato Marco and Yang, Chao-Han Huck and Fu, Szu-Wei and Tsao, Yu},
  journal={arXiv preprint arXiv:2405.06573},
  year={2024}
}
```
