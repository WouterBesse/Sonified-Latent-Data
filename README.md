# Sonified Latent Data :microphone::abacus:	
## Experiments to sonify different latent distributions generated by encoders from multiple datasets.

This repository is made to assist my university thesis about sonifying the latent space. We couple a WaveNet + VAE similar to [[Chorowski et al., 2019]](https://arxiv.org/abs/1901.08810) with latent distrubitions generated from multiple different dataset-encoder couples. 

The goal is to discover if the sonified audio results contains meaningfull differences and features from the input datasets.
A lot of earlier experiments relevant to this project I tried using the WaveNet decoder can be found in my [denoising repository](https://github.com/WouterBesse/ConvDenoiser).

## How To Use

- First install all dependencies from requirements.txt
- Every model has its' own notebook containing training and inference instructions
- Models can be downloaded from here: (to be made)
- Datasets will are listed at the end of this readme

## Models
### Standard WaveNet VAE

This model is very similar to the one described by [Chorowski et al.](https://arxiv.org/abs/1901.08810) and follows the following model:
I decided to go with a normal VAE and not the quantized variant because it allows me to more easily interpolate and play with the latent space.

![WaveNet VAE Diagram](https://github.com/WouterBesse/Sonified-Latent-Data/blob/main/media/WaveNetVae.jpg?raw=true)

#### Training and model

My model is downloadable from 'n.b.t.', I trained it on the LJSpeech dataset. You can train your own model using `train.py` from the WaveNetVAE folder or by using the `WaveVaePlayground.ipynb` jupyter notebook.

example usage of CLI train.py: 

`python3 train.py -tp "./traindatasetfolder/" -vp "./validationdatasetfolder/" -ep 100`
- `-tp`  | `--train_path` path of folder where training audio data is stored
- `-vp`  | `--validation_path` path of folder where validation audio data is stored
- `-ep`  | `--epochs` amount of epochs to train
- `-ex`  | `--export_path` path of folder to export model files to
- `-bs`  | `--batch_size` batch size
- `-lr`  | `--learning_rate` learning rate, I recommend 0.00001
- `-kla` | `--kl_anneal` how much the kl rate multiplier is increased after every log step
- `-mkl` | `--max_kl` what the maximum kl rate multiplier will be
- `-lpe` | `--logs_per_epoch` how often a tensorboard log is stored per epoch
- `-d`   | `--device` what device to train on, e.g. `cuda:0`, `cpu`
- `-mf`  | `--max_files` the maximum amount of files to use in the train dataset


### Tybalt WaveNet VAE

T.B.D.

### Mocap WaveNet VAE


## Datasets

