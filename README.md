# Automated-Audio-Captioning

The code is a modification of [DCASE2024 Challenge Task 6 baseline system of AAC](https://github.com/Labbeti/dcase2024-task6-baseline).

<a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/-Python 3.11-blue?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
</a>
</div>

The main model is composed of a pretrained convolutional encoder to extract features and a transformer decoder to generate caption.
For more information, please refer to the corresponding [DCASE task page](https://dcase.community/challenge2024/task-automated-audio-captioning).

**This repository includes:**
- AAC model trained on the a dataset consisting of **Clotho** dataset and summarizations obtained using Gemini 1.5 Pro
- Extract features using **ConvNeXt**
- System reaches **0.545 FENSE** score on Clotho-eval (development-testing)
- Output detailed training characteristics (number of parameters, MACs, energy consumption...)


## Installation
First, you need to create an environment that contains **python>=3.11** and **pip**. You can use venv, conda, micromamba or other python environment tool.

Here is an example with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
```bash
micromamba env create -n env_dcase24 python=3.11 pip -c defaults
micromamba activate env_dcase24
```

Then, you can clone this repository and install it:
```bash
git clone https://github.com/KWencierski/Automated-Audio-Captioning
cd Automated-Audio-Captioning
pip install -e .
pre-commit install
```

You also need to install Java >= 1.8 and <= 1.13 on your machine to compute AAC metrics. If needed, you can override java executable path with the environment variable `AAC_METRICS_JAVA_PATH`.


## Usage

### Download external data, models and prepare

To download, extract and process data, you need to run:
```bash
dcase24t6-prepare
```
By default, the dataset is stored in `./data` directory. It will requires approximatively 33GB of disk space.

### Train the default model

```bash
dcase24t6-train +expt=baseline
```

By default, the model and results are saved in directory `./logs/SAVE_NAME`. `SAVE_NAME` is the name of the script with the starting date.
Metrics are computed at the end of the training with the best checkpoint.

### Test a pretrained model

```bash
dcase24t6-test resume=./logs/SAVE_NAME
```
or specify each path separtely:
```bash
dcase24t6-test resume=null model.checkpoint_path=./logs/SAVE_NAME/checkpoints/MODEL.ckpt tokenizer.path=./logs/SAVE_NAME/tokenizer.json
```
You need to replace `SAVE_NAME` by the save directory name and `MODEL` by the checkpoint filename.

If you want to load and test the baseline pretrained weights, you can specify the baseline checkpoint weights:

```bash
dcase24t6-test resume=~/.cache/torch/hub/checkpoints/Automated-Audio-Captioning
```

### Inference on a file
If you want to test the baseline model on a single file, you can use the `baseline_pipeline` function:

```python
from dcase24t6.nn.hub import baseline_pipeline

sr = 44100
audio = torch.rand(1, sr * 15)

model = baseline_pipeline()
item = {"audio": audio, "sr": sr}
outputs = model(item)
candidate = outputs["candidates"][0]

print(candidate)
```

