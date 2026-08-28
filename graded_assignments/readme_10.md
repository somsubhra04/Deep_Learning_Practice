# Low Light Object Detection and Classification of Mosquitoes

## Overview

Mosquitoes, small yet perilous insects, are responsible for transmitting diseases that pose a serious threat to humans and the environment. With over 3600 known species, a few of them have the ability to transmit various pathogens, leading to widespread illnesses such as Zika, Dengue, and Chikungunya. Controlling mosquito populations is vital to prevent disease outbreaks and protect communities worldwide.

Traditional mosquito surveillance methods are expensive and time-consuming, but community-based approaches empower citizens to report and collect mosquito specimens. By leveraging machine learning and deep learning techniques, we aim to automate the labor-intensive image validation process, making mosquito identification more efficient and accurate.

## Freedom to Innovate

Unlike the previous challenge, this competition places no restrictions on models, compute, or external data. You are encouraged to explore the full landscape of object detection techniques.

Whether you choose to fine-tune powerful pre-trained models like YOLO, Faster R-CNN etc. Leverage advanced vision foundation models, or design a novel detection architecture from scratch, the choice is yours.

You are free to use any tools, libraries, datasets, and computational resources at your disposal.

The goal is to push the boundaries of object detection performance on this dataset. We look forward to seeing the creative and powerful solutions you develop!

## Description

In this competition, the goal is to develop a model capable of detecting and classifying mosquitoes into six distinct types using images.

The dataset likely includes high-quality images of mosquitoes, and participants need to design a system that can accurately identify and differentiate between these species.

This problem involves computer vision tasks such as:

- Object detection (to locate the mosquito within the image)
- Classification (to determine the mosquito's type)

## Evaluation

Submissions are evaluated on **mAP score**.

mAP will be used to quantify the correctness of the bounding box predictions.

**Higher score is better.**

## Dataset Description

The dataset for this challenge is derived from a citizen science project focused on mosquito identification.

It comprises **8025 real-world images of mosquitos** captured by participants using mobile phone. These images offer a diverse representation of mosquitos in various scenarios and locations.

Each image is labeled with bounding box coordinates and mosquito class information.

The dataset has been split into training and testing sets, with **93.5% of the images allocated for training (7500 images)** and **6.5% for testing (525 images)**.

This division provides participants with ample training data to develop their models and a separate evaluation set to assess the performance and generalization of their AI algorithms.

## Files

- `train/images` - The training images
- `train/labels` - The training labels
- `test/images` - The testing images
- `sample_submission.csv` - A sample submission file in the correct format

## Label Format

In each `.txt` file in `train/labels/`:

- `class_label` - Label of the Image
- `bbx_xcenter` - X coordinate of center of Bounding Box
- `bbx_ycenter` - Y coordinate of center of Bounding Box
- `bbx_width` - Width of the Bounding Box
- `bbx_height` - Height of the Bounding Box

## Classes

There are **6 Classes**:

```text
{
    "aegypti": 0,
    "albopictus": 1,
    "anopheles": 2,
    "culex": 3,
    "culiseta": 4,
    "japonicus/koreicus": 5
}

Submission CSV Format

The submission CSV should follow the required format:

id  ImageID  LabelName  Conf  xcenter  ycenter  bbx_width  bbx_height
