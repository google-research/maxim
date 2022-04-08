# [MAXIM](https://arxiv.org/abs/2201.02973): Multi-Axis MLP for Image Processing (CVPR 2022 Oral)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-j-1)](https://paperswithcode.com/sota/deblurring-on-realblur-j-1?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-r)](https://paperswithcode.com/sota/deblurring-on-realblur-r?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/photo-retouching-on-mit-adobe-5k)](https://paperswithcode.com/sota/photo-retouching-on-mit-adobe-5k?p=maxim-multi-axis-mlp-for-image-processing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-rain100h)](https://paperswithcode.com/sota/single-image-deraining-on-rain100h?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-rain100l)](https://paperswithcode.com/sota/single-image-deraining-on-rain100l?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-test100)](https://paperswithcode.com/sota/single-image-deraining-on-test100?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-test2800)](https://paperswithcode.com/sota/single-image-deraining-on-test2800?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-test1200)](https://paperswithcode.com/sota/single-image-deraining-on-test1200?p=maxim-multi-axis-mlp-for-image-processing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=maxim-multi-axis-mlp-for-image-processing)


This repo is the official implementation of [**CVPR 2022 Oral**] paper: ["**MAXIM**: Multi-Axis MLP for Image Processing"](https://arxiv.org/abs/2201.02973) by [Zhengzhong Tu](https://www.linkedin.com/in/vztu/), [Hossein Talebi](https://scholar.google.com/citations?hl=en&user=UOX9BigAAAAJ), [Han Zhang](https://sites.google.com/view/hanzhang), [Feng Yang](https://sites.google.com/view/feng-yang), [Peyman Milanfar](https://sites.google.com/view/milanfarhome/), [Alan Bovik](https://www.ece.utexas.edu/people/faculty/alan-bovik), and [Yinxiao Li](https://scholar.google.com/citations?user=kZsIU74AAAAJ&hl=en)

Google Research, University of Texas at Austin

*Disclaimer: This is not an officially supported Google product.*

<hr />

> **Abstract:** *Recent progress on Transformers and multi-layer perceptron (MLP) models provide new network architectural designs for computer vision tasks. Although these models proved to be effective in many vision tasks such as image recognition, there remain challenges in adapting them for low-level vision. The inflexibility to support high-resolution images and limitations of local attention are perhaps the main bottlenecks. In this work, we present a multi-axis MLP based architecture called MAXIM, that can serve as an efficient and flexible general-purpose vision backbone for image processing tasks. MAXIM uses a UNet-shaped hierarchical structure and supports long-range interactions enabled by spatially-gated MLPs. Specifically, MAXIM contains two MLP-based building blocks: a multi-axis gated MLP that allows for efficient and scalable spatial mixing of local and global visual cues, and a cross-gating block, an alternative to cross-attention, which accounts for cross-feature conditioning. Both these modules are exclusively based on MLPs, but also benefit from being both global and `fully-convolutional', two properties that are desirable for image processing. Our extensive experimental results show that the proposed MAXIM model achieves state-of-the-art performance on more than ten benchmarks across a range of image processing tasks, including denoising, deblurring, deraining, dehazing, and enhancement while requiring fewer or comparable numbers of parameters and FLOPs than competitive models.*
<hr />

## Architecture

![Model overview](maxim/images/overview.png)

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

Setup project:

```
pip install .
```

## Results and Pre-trained models

We provide all the pre-trained models and visual results.

| Task | Dataset | PSNR | SSIM | Model | #params | FLOPs | ckpt | outputs |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|
| Denoising | SIDD | 39.96 | 0.960 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Denoising/SIDD/) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Denoising/SIDD/) |
| Denoising | DND  | 39.84 | 0.954 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Denoising/SIDD/) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Denoising/DND/) |
| Deblurring | GoPro | 32.86 | 0.961 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/GoPro) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deblurring/GoPro/) |
| Deblurring | HIDE  | 32.83 | 0.956 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/GoPro) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deblurring/GoPro/) |
| Deblurring | REDS  | 28.93 | 0.865 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/REDS) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deblurring/REDS/) |
| Deblurring | RealBlur-R | 39.45 | 0.962 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/RealBlur_R) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deblurring/RealBlur/) |
| Deblurring | RealBlur-J | 32.84 | 0.935 | MAXIM-3S | 22.2M | 339G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/RealBlur_J) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deblurring/RealBlur/) |
| Deraining | Rain13k | 33.24 | 0.933 | MAXIM-2S | 14.1M | 216G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deraining/Rain13k) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deraining/Rain13k/) |
| Deraining | Raindrop | 31.87 | 0.935 | MAXIM-2S | 14.1M | 216G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deraining/Raindrop) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Deraining/Raindrop/) |
| Dehazing | RESIDE-Indoor | 38.11 | 0.991 | MAXIM-2S | 14.1M | 216G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Dehazing/SOTS-Indoor) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Dehazing/RESIDE-Indoor/) |
| Dehazing | RESIDE-Outdoor | 34.19 | 0.985 | MAXIM-2S | 14.1M | 216G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Dehazing/SOTS-Outdoor) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Dehazing/RESIDE-Outdoor/) |
| Enhancement | LOL | 23.43 | 0.863 | MAXIM-2S | 14.1M | 216G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Enhancement/LOL) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Enhancement/LOL/) |
| Enhancement | FiveK | 26.15 | 0.945 | MAXIM-2S | 14.1M  |  216G | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Enhancement/FiveK) | [images](https://console.cloud.google.com/storage/browser/gresearch/maxim/results/Enhancement/FiveK/) |

<!-- You can also download most of the training and test datasets we used on [gcloud](https://console.cloud.google.com/storage/browser/gresearch/maxim/datasets/). -->

## Demo

First download corresponding checkpoints and then go ahead and run:

<details>
  <summary><strong>Image Denoising</strong> (click to expand) </summary>

```
python3 maxim/run_eval.py --task Denoising --ckpt_path ${SIDD_CKPT_PATH} \
  --input_dir maxim/images/Denoising --output_dir maxim/images/Results --has_target=False
```
</details>

<details>
  <summary><strong>Image Deblurring</strong> (click to expand) </summary>

```
python3 maxim/run_eval.py --task Deblurring --ckpt_path ${GOPRO_CKPT_PATH} \
  --input_dir maxim/images/Deblurring --output_dir maxim/images/Results --has_target=False
```
</details>

<details>
  <summary><strong>Image Deraining</strong> (click to expand) </summary>

Rain streak:
```
python3 maxim/run_eval.py --task Deraining --ckpt_path ${RAIN13K_CKPT_PATH} \
  --input_dir maxim/images/Deraining --output_dir maxim/images/Results --has_target=False
```

Rain drop:
```
python3 maxim/run_eval.py --task Deraining --ckpt_path ${RAINDROP_CKPT_PATH} \
  --input_dir maxim/images/Deraining --output_dir maxim/images/Results --has_target=False
```
</details>

<details>
  <summary><strong>Image Dehazing</strong> (click to expand) </summary>

Indoor:
```
python3 maxim/run_eval.py --task Dehazing --ckpt_path ${REDISE_INDOOR_CKPT_PATH} \
  --input_dir maxim/images/Dehazing --output_dir maxim/images/Results --has_target=False
```

Outdoor:
```
python3 maxim/run_eval.py --task Dehazing --ckpt_path ${REDISE_OUTDOOR_CKPT_PATH} \
  --input_dir maxim/images/Dehazing --output_dir maxim/images/Results --has_target=False
```
</details>

<details>
  <summary><strong>Image Enhancement</strong> (click to expand) </summary>

Low-light enhancement:
```
python3 maxim/run_eval.py --task Enhancement --ckpt_path ${LOL_CKPT_PATH} \
  --input_dir maxim/images/Enhancement --output_dir maxim/images/Results --has_target=False
```

Retouching:
```
python3 maxim/run_eval.py --task Enhancement --ckpt_path ${FIVEK_CKPT_PATH} \
  --input_dir maxim/images/Enhancement --output_dir maxim/images/Results --has_target=False
```
</details>

## Results

<details>
  <summary><strong>Image Denoising</strong> (click to expand) </summary>

<img src = "https://user-images.githubusercontent.com/43280278/149262475-a73668f2-9fe1-4374-8ed3-4831acca8052.png" width="400">
</details>

<details>
<summary><strong>Image Deblurring</strong> (click to expand) </summary>

<table>
  <tr>
    <td> <img src = "https://user-images.githubusercontent.com/43280278/149261823-b77e9513-b3b5-4caf-a0eb-67bf18c2f681.png" width="500"> </td>
    <td> <img src = "https://user-images.githubusercontent.com/43280278/149261858-24664c33-dc8a-47c3-b84d-ba64b1c05937.png" width="500"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Synthetic blur</b></p></td>
    <td><p align="center"><b>Realistic blur</b></p></td>
  </tr>
</table>
</details>

<details>
<summary><strong>Image Deraining</strong> (click to expand) </summary>

<table>
  <tr>
    <td> <img src = "https://user-images.githubusercontent.com/43280278/149261908-8bce72cf-b343-4bf8-8462-8be363616cfa.png" width="700"> </td>
    <td> <p align="top"> <img src = "https://user-images.githubusercontent.com/43280278/149262066-7b93538a-2ccc-4ea0-9187-ef1b54734392.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Rain streak</b></p></td>
    <td><p align="center"><b>Rain drop</b></p></td>
  </tr>
</table>
</details>

<details>
<summary><strong>Image Dehazing</strong> (click to expand) </summary>

<img src = "https://user-images.githubusercontent.com/43280278/149261947-22954827-ce62-44e8-974a-0aa8d94a4bd9.png"  width="250">
</details>

<details>
<summary><strong>Image Enhancement</strong> (click to expand) </summary>

<img src = "https://user-images.githubusercontent.com/43280278/149262540-77d16592-9305-4fd7-80c6-b9d30000cc29.png" width="400">
</details>

## Citation
Should you find this repository useful, please consider citing:
```
@article{tu2022maxim,
  title={MAXIM: Multi-Axis MLP for Image Processing},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  journal={CVPR},
  year={2022},
}
```

## Acknowledgement

This repository is built on the [vision_transformer](https://github.com/google-research/vision_transformer) and [musiq](https://github.com/google-research/google-research/tree/master/musiq) repositories. Our work is also inspired by [HiT](https://github.com/google-research/hit-gan), [MPRNet](https://github.com/swz30/MPRNet), and [HINet](https://github.com/megvii-model/HINet).
