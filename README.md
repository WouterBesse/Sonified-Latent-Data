# Sonified Latent Data :microphone::abacus:	
## Experiments to sonify different latent distributions generated by encoders from multiple datasets.

This repository is made to assist my university thesis about building a WaveNet VAE. After this, my plan is to couple a WaveNet + VAE similar to [[Chorowski et al., 2019]](https://arxiv.org/abs/1901.08810) with latent distrubitions generated from multiple different dataset-encoder couples. 

The goal is to discover if the sonified audio results contains meaningful differences and features from the input datasets.
A lot of earlier experiments relevant to this project I tried using the WaveNet decoder can be found in my [denoising repository](https://github.com/WouterBesse/ConvDenoiser).

## How To Use

- First install all dependencies from requirements.txt
- Every model has its' own notebook containing training and inference instructions
- Models can be downloaded from here: (to be made)
- Datasets will are listed at the end of this readme

## Models

### Standard WaveNet VAE
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WM5do9HKMslXeQzKuPiklYVT-IwOogzL?usp=sharing)

<img src="https://github.com/WouterBesse/Sonified-Latent-Data/blob/main/media/WaveNetVae.jpg?raw=true" align="right" width="430px" alt="WaveNet VAE Diagram" />

The first result can be found in `media/FirstResult.wav`. It's very noisy, but it's definitely trying to make some patterns of speech.

This model is very similar to the one described by [Chorowski et al.](https://arxiv.org/abs/1901.08810) and follows the following model:
I decided to go with a normal VAE and not the quantized variant because it allows me to more easily interpolate and play with the latent space.

For the actual code I took inspriation, and sometimes flat out copied, from the following repositories:
- [hrbigelow/ae-wavenet](https://github.com/hrbigelow/ae-wavenet)
- [swasun/VQ-VAE-Speech](https://github.com/swasun/VQ-VAE-Speech)
- [DongyaoZhu/VQ-VAE-WaveNet](https://github.com/DongyaoZhu/VQ-VAE-WaveNet)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)


#### Training and model

My model is downloadable from 'n.b.t.', I trained it on the LJSpeech dataset. You can train your own model using `train.py` from the WaveNetVAE folder or by using the `WaveVaePlayground.ipynb` jupyter notebook.

Example usage of CLI train.py: 

`python3 train.py -tp "./traindatasetfolder/" -vp "./validationdatasetfolder/" -ep 100`
| **Short Flag** | **Long Flag**       | **Description**                    |
|----------------|---------------------|------------------------------------|
| `-tp`          | `--train_path`      | Path of training data              |
| `-vp`          | `--validation_path` | Path of validation data            |
| `-ep`          | `--epochs`          | Amount of epochs to train          |
| `-ex`          | `--export_path`     | Model export location              |
| `-bs`          | `--batch_size`      | Batch size                         |
| `-lr`          | `--learning_rate`   | Learning rate                      |
| `-kla`         | `--kl_anneal`       | KL multiplier increase per step    |
| `-mkl`         | `--max_kl`          | Maximum KL multiplier              |
| `-lpe`         | `--logs_per_epoch`  | Validation frequency               |
| `-d`           | `--device`          | Train device, e.g. `cuda:0`, `cpu` |
| `-mf`          | `--max_files`       | Maximum amount of files in dataset |

---
### Tybalt WaveNet VAE

<img src="https://github.com/WouterBesse/Sonified-Latent-Data/blob/main/media/Tybalt.svg?raw=true" align="right" width="430px" alt="Tybalt VAE Diagram" />

A alteration on the [Tybalt VAE model](https://github.com/greenelab/tybalt) by [Way et al.](https://www.biorxiv.org/content/10.1101/174474v2)
I gave it one extra linear layer to help reducing the data to a smaller latent space.

#### Training and model

My model is downloadable from the releases section, it's trained on the TCGA dataset. You can train your own model using `train.py` from the Tybalt model folder or by using the `TybaltPlayground.ipynb` jupyter notebook.
The acquisition and preprocessing scripts are available in the original [Tybalt GitHub](https://github.com/greenelab/tybalt).

Example usage of CLI train.py: 

`python3 train.py -dp "./traindatasetfolder/" -ep 100`
| **Short Flag** | **Long Flag**       | **Description**                    |
|----------------|---------------------|------------------------------------|
| `-dp`          | `--data_path`       | Path of all data                   |
| `-ep`          | `--epochs`          | Amount of epochs to train          |
| `-ex`          | `--export_path`     | Model export location              |
| `-bs`          | `--batch_size`      | Batch size                         |
| `-lr`          | `--learning_rate`   | Learning rate                      |
| `-kla`         | `--kl_anneal`       | KL multiplier increase per step    |
| `-mkl`         | `--max_kl`          | Maximum KL multiplier              |
| `-lpe`         | `--logs_per_epoch`  | Validation frequency               |
| `-d`           | `--device`          | Train device, e.g. `cuda:0`, `cpu` |
| `-mf`          | `--max_files`       | Maximum amount of files in dataset |
---
### Mocap WaveNet VAE


## Datasets

