## Please check [https://huggingface.co/somsubhra04/lowlight-denoise-sr4x](https://huggingface.co/somsubhra04/lowlight-denoise-sr4x)

## Overview

The goal of this competition is to develop robust deep learning models capable of denoising and enhancing the resolution of low-light images. Participants will work with a dataset of low-light images affected by noise and poor visibility, aiming to produce high-quality outputs that are both visually appealing and rich in detail. Specifically, the task is to denoise the images and perform 4x super-resolution, significantly improving their clarity and resolution.

## Dataset
The dataset will consist of three key components:

* Training Set: Contains low-resolution noisy images along with their corresponding high-resolution clean ground truth images.
* Evaluation Set: Similar to the training set, it includes low-resolution noisy images and their corresponding high-resolution clean images, allowing participants to validate the performance of their models during development.
* Test Set: Comprises only low-resolution noisy images without any corresponding high-resolution clean images. The performance of the models will be evaluated on this set.


## Evaluation
Submissions will be assessed based on Peak-Signal-to-Noise-Ratio (PSNR) score.

Examples of paths:
/kaggle/input/competitions/dlp-26t2-nppe3/archive/test/test_00001.png

/kaggle/input/competitions/dlp-26t2-nppe3/archive/train/gt/gt_00001.png

/kaggle/input/competitions/dlp-26t2-nppe3/archive/train/train/gt_00001.png

/kaggle/input/competitions/dlp-26t2-nppe3/archive/val/gt/gt_00001.png

/kaggle/input/competitions/dlp-26t2-nppe3/archive/val/val/val_00001.png

Submission csv Format: ID, pixel_0, ...
Also, I need to push the model file to a hf repo, so please add this as well in the code so that this happens automatically after the training ends. I have a hf api key.
