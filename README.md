# Sonified Latent Data
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

### Tybalt WaveNet VAE

T.B.D.

### Mocap WaveNet VAE


## Datasets

