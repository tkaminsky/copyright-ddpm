# STAT 98 Final Paper Code

The accompanying code for Thomas Kaminsky's STAT 98 final research project---most comes from Yi-Lun Wu and Peter Lorenz's implementation of Denoising Diffusion Probabilistic Models [1] (https://github.com/w86763777/pytorch-ddpm).

My contributions include copyright.py, which contains a custom implementation of the CP-k algorithm described in Provable Copyright Protection for Generative Models [3], as well as some minor changes to main.py.

To use:

1. Follow the instructions below to set up the environment.
2. Download the pretrained checkpoint and store it in /logs/

## Experiments:

### Training:

Training the copy dataset (for the cover):
```
python main.py --train --flagfile ./config/CIFAR10.txt --logdir ./logs/CIFAR10_copy --dataset copy --ckpt ./logs/DDPM_CIFAR10_EPS/
```

Training the safe datasete (for the cover):
```
python main.py --train --flagfile ./config/CIFAR10.txt --logdir ./logs/CIFAR10_safe --dataset safe --ckpt ./logs/DDPM_CIFAR10_EPS/
```

Training the full dataset:
```
python main.py --train --flagfile ./config/CIFAR10.txt --logdir ./logs/CIFAR10_full --dataset full --ckpt ./logs/DDPM_CIFAR10_EPS/
```

### Evaluation:

Full model performance:

```
python main.py \
    --flagfile ./logs/CIFAR10_full/flagfile.txt \
    --notrain \
    --eval
```

Safe model performance:

```
python main.py \
    --flagfile ./logs/CIFAR10_safe/flagfile.txt \
    --notrain \
    --eval
```

Copy model performance:

```
python main.py \
    --flagfile ./logs/CIFAR10_copy/flagfile.txt \
    --notrain \
    --eval
```

Evaluate the copyright-safe model:

```
python main.py \
    --flagfile ./logs/CIFAR10_full_final/flagfile.txt \
    --notrain \
    --eval \ --is_copyright_safe True
```

Baseline model performance:

```
python main.py \
    --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
    --notrain \
    --eval
```




# Denoising Diffusion Probabilistic Models

Unofficial PyTorch implementation of Denoising Diffusion Probabilistic Models [1].

This implementation follows the most of details in official TensorFlow
implementation [2]. I use PyTorch coding style to port [2] to PyTorch and hope
that anyone who is familiar with PyTorch can easily understand every
implementation details.

## TODO
- Datasets
    - [x] Support CIFAR10
    - [ ] Support LSUN
    - [ ] Support CelebA-HQ
- Featurex
    - [ ] Gradient accumulation
    - [x] Multi-GPU training
- Reproducing Experiment
    - [x] CIFAR10

## Requirements
- Python 3.6
- Packages
    Upgrade pip for installing latest tensorboard
    ```
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```
- Download precalculated statistic for dataset:

    [cifar10.train.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing)

    Create folder `stats` for `cifar10.train.npz`.
    ```
    stats
    └── cifar10.train.npz
    ```

## Train From Scratch
- Take CIFAR10 for example:
    ```
    python main.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Overwrite arguments
    ```
    python main.py --train \
        --flagfile ./config/CIFAR10.txt \
        --batch_size 64 \
        --logdir ./path/to/logdir
    ```
- [Optional] Select GPU IDs
    ```
    CUDA_VISIBLE_DEVICES=1 python main.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Multi-GPU training
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train \
        --flagfile ./config/CIFAR10.txt \
        --parallel
    ```

## Evaluate
- A `flagfile.txt` is autosaved to your log directory. The default logdir for `config/CIFAR10.txt` is `./logs/DDPM_CIFAR10_EPS`
- Start evaluation
    ```
    python main.py \
        --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
        --notrain \
        --eval
    ```
- [Optional] Multi-GPU evaluation
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
        --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
        --notrain \
        --eval \
        --parallel
    ```


## Reproducing Experiment

### CIFAR10
- FID: 3.249, Inception Score: 9.475(0.174)
![](./images/cifar10_samples.png)

The checkpoint can be downloaded from my [drive](https://drive.google.com/file/d/1IhdFcdNZJRosi3XRT7-qNmiPGTuyuEXr/view?usp=sharing).

## Reference

[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[2] [Official TensorFlow implementation](https://github.com/hojonathanho/diffusion)

[3] [Provable Copyright Protection for Generative Models](https://arxiv.org/abs/2302.10870)
