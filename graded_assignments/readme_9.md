# Image Classification Competition

The topic of this competition is **Image Classification**. The dataset to be used in this competition consists of images from the world of flora and fauna. The details of the dataset are given in the Data Section.

## Goal

The aim of this competition is to use a model which is able to classify the images of animals and plants with best **F1 score**.

## Note

You are not allowed to use **transfer learning** or **transformer model**. Any submission generated using such models will be cancelled, and zero will be awarded. In this round, all students must build a **CNN model** to predict the labels of the test images.

## Evaluation

Submissions are evaluated on **weighted F1 score ("micro")**.

## Submission File

Submit a CSV file containing the predictions for **2000 images**. For each image in the test set, you must predict the label.

The `Image_ID` will be the image file name without extension and `Label` will be a number between `0` to `9`. Read the details about labels below.

```text
Image_ID, Label
Image_0001, 0
Image_1123, 5
etc.

Label Mapping

Use the following mapping for submitting labels:

Amphibia - 0
Animalia - 1
Arachnida - 2
Aves - 3
Fungi - 4
Insecta - 5
Mammalia - 6
Mollusca - 7
Plantae - 8
Reptilia - 9
Dataset Description

The dataset consists of two folders, train and test.

The train folder is subdivided into 10 subfolders, each containing 1,000 images, and the name of these subfolders is the label of the respective class.

For the test folder, all 2,000 images are in a single folder.

Your task is to predict the labels of these images.

Label Mapping

Use the following mapping for submitting labels:

Amphibia - 0
Animalia - 1
Arachnida - 2
Aves - 3
Fungi - 4
Insecta - 5
Mammalia - 6
Mollusca - 7
Plantae - 8
Reptilia - 9
Files
train.csv - The train folder consisting of 10,000 images subdivided equally into 10 folders representing 10 classes.
test.csv - The test folder consists of 2,000 images for which label has to be predicted.
sample_submission.csv - A sample submission file in the correct format.
metaData.csv - A sample submission file in the correct format.
Columns
Image_ID - filename without extension (e.g. Image_0001)
Label - Your prediction for the test images which is a value between 0 and 9.
