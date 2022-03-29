# [MAXIM](https://arxiv.org/abs/2201.02973): Multi-Axis MLP for Image Processing

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-j-1)](https://paperswithcode.com/sota/deblurring-on-realblur-j-1?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-rain100h)](https://paperswithcode.com/sota/single-image-deraining-on-rain100h?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-rain100l)](https://paperswithcode.com/sota/single-image-deraining-on-rain100l?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-r)](https://paperswithcode.com/sota/deblurring-on-realblur-r?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-test100)](https://paperswithcode.com/sota/single-image-deraining-on-test100?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-test2800)](https://paperswithcode.com/sota/single-image-deraining-on-test2800?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/single-image-deraining-on-test1200)](https://paperswithcode.com/sota/single-image-deraining-on-test1200?p=maxim-multi-axis-mlp-for-image-processing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maxim-multi-axis-mlp-for-image-processing/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=maxim-multi-axis-mlp-for-image-processing)


This repo is the official implementation of [CVPR 2022] paper: ["**MAXIM**: Multi-Axis MLP for Image Processing"](https://arxiv.org/abs/2201.02973) by [Zhengzhong Tu](https://www.linkedin.com/in/vztu/), [Hossein Talebi](https://scholar.google.com/citations?hl=en&user=UOX9BigAAAAJ), [Han Zhang](https://sites.google.com/view/hanzhang), [Feng Yang](https://sites.google.com/view/feng-yang), [Peyman Milanfar](https://sites.google.com/view/milanfarhome/), [Alan Bovik](https://www.ece.utexas.edu/people/faculty/alan-bovik), and [Yinxiao Li](https://scholar.google.com/citations?user=kZsIU74AAAAJ&hl=en)

Google Research, University of Texas at Austin

*Disclaimer: This is not an officially supported Google product.*

![Model overview](maxim/images/overview.png)


## Installation

Install dependencies:

```
pip install -r requirements.txt
```

## Results and Pre-trained models

TBD

## Evaluation

Example inference command:

```
python3 eval.py --task ${TASK} --ckpt_path ${CKPT_PATH} \
  --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR}
```

## Acknowledgement

This repository is built on the [vision_transformer](https://github.com/google-research/vision_transformer) and [musiq](https://github.com/google-research/google-research/tree/master/musiq) repositories.

## Citation
Should you find this repository useful, please consider citing:
```
@article{tu2022maxim,
  title={MAXIM: Multi-Axis MLP for Image Processing},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  journal={arXiv preprint arXiv:2201.02973},
  year={2022}
}
```
