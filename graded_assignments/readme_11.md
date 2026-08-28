# Depth Estimation Challenge

## Overview

Welcome to the Depth Estimation Challenge! This competition invites participants to develop robust models capable of predicting depth from RGB images that have been intentionally corrupted with various degradations. The goal is to advance the field of depth estimation in challenging real-world scenarios.

## Description

### Challenge Overview

**Task:**

Your task is to predict accurate depth maps from RGB images affected by environmental conditions, such as:

- Low-Light
- Noise 

**Dataset:**

Participants will be provided with a dataset containing pairs of degraded RGB images and their corresponding ground truth depth maps. Each sample will feature one of the aforementioned degradations, making this a unique challenge for depth estimation models.

**Goals:**

- Develop innovative techniques to handle the effects of degradations.
- Explore and compare various deep learning architectures.
- Contribute to a deeper understanding of how environmental factors influence depth perception.

## Evaluation

Submissions will be evaluated based on the accuracy of the predicted depth maps using metrics like:

- Root Mean Squared Error (RMSE)

**Note:** Please note that after you generated your depth predictions, use the code in `imgs2csv.py` file to convert the directory of images to csv and submit that csv in kaggle.

```python
import os
import cv2
import pandas as pd
import numpy as np

def images_to_csv_with_metadata(image_folder, output_csv):
    # Initialize an empty list to store image data and metadata
    data = []

    # Loop through all images in the folder
    for idx, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.endswith(".png"):
            filepath = os.path.join(image_folder, filename)

            # Read the image
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (128, 128))
            image = image / 255
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
            image = np.uint8(image * 255.)

            # Flatten the image into a 1D array
            image_flat = image.flatten()

            # Add ID, ImageID (filename), and pixel values
            row = [idx, filename] + image_flat.tolist()
            data.append(row)

    # Create a DataFrame
    num_columns = len(data[0]) - 2 if data else 0
    column_names = ["id", "ImageID"] + [indx for indx in range(num_columns)]
    df = pd.DataFrame(data, columns=column_names)

    # Save to CSV
    df.to_csv(output_csv, index=False)


# Paths for prediction and ground truth images
predictions_folder = "data/sample_solution"

# Output CSV paths
predictions_csv = "predictions.csv"

# Convert prediction images to CSV
images_to_csv_with_metadata(predictions_folder, predictions_csv)

Dataset Description

The dataset for the Depth Estimation Challenge consists of pairs of RGB images and their corresponding ground truth depth maps, specifically designed to simulate real-world conditions. Each RGB image is intentionally corrupted with various environmental degradations.

Dataset Components
Data Splits
Training Set: Contains around 6686images with corresponding depth maps for model -training.
Validation Set: Contains around 836 images for evaluating model performance.
Testing Set: Contains around 836 images for testing the performance and for leaderboard upload.
Usage Guidelines

Participants are encouraged to explore various preprocessing techniques to mitigate the effects of degradations. The depth maps can be used for direct comparison against predicted outputs to assess model accuracy.

submission

Please note that after you generated your depth predictions, use the imgs2csv.py file to convert the directory of images to csv and submit that csv in kaggle.

Notes

Ensure to properly handle the variations introduced by the degradations in your model design. Remember to adhere to the competition rules regarding data usage and integrity. This dataset presents a unique challenge on robust algorithms capable of operating under real-world conditions.
