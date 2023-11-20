# Airbus Ship Detection Challenge

![Airbus Ship Detection]

## Overview

This repository contains my solutions and insights for the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) hosted on Kaggle. The challenge focuses on developing algorithms to detect and classify ships in satellite images provided by Airbus.

## Challenge Description

The goal of the competition is to create a model capable of identifying and segmenting ships within satellite images. The dataset includes a variety of images in resolution of 768x768 pixels taken from different perspectives, and the task involves both classification and segmentation of ships.

For more details, please visit the [competition page](https://www.kaggle.com/c/airbus-ship-detection).

## Repository Structure

- `images/`: Images used for README file.
- `utils/`: Utilities used in model building.
- `ship_detection_data/`: Placeholder for the dataset (you need to download it from Kaggle and place it here).
- `ship_eda.ipynb`: Exploratory data analysis of the dataset.
- `model_train.py`: Training the model.
- `model_inference.py`: Model inference on test images.
- `README.md`: README file.
- `requirements.txt`: List of dependencies for replicating the environment.

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/airbus-ship-detection.git
   ```

2. Set up the environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/airbus-ship-detection) and place it in the `ship_detection_data/` directory.

4. Run the `model_train.py` for training the model. Run the `model_inference.py` to evaluate the model on the test dataset.

## Model Architecture

The implemented model follows a modified U-Net architecture, a popular choice for semantic segmentation tasks. The architecture consists of a contracting path, a U-Net bottom, and an expansive path, designed to capture both high-level context and precise spatial information.

### Contracting Path

The contracting path begins with an input layer followed by a sequence of convolutional blocks, each composed of two convolutional layers with rectified linear unit (ReLU) activation. Max-pooling layers with a kernel size of (2, 2) are applied after each convolutional block to reduce spatial dimensions.

### U-Net Bottom

The U-Net bottom acts as a bottleneck, containing a convolutional block with increased filter size to retain rich feature information. This central block is crucial for maintaining context during the segmentation process.

### Expansive Path

The expansive path consists of transposed convolutions (Conv2DTranspose) to upsample the feature maps. Skip connections, concatenating feature maps from the contracting path, are employed to recover spatial resolution and enhance segmentation accuracy. Each upsampling step is followed by a convolutional block similar to those in the contracting path.

### Output Layer

The final layer is a 1x1 convolutional layer with a sigmoid activation function, producing a binary segmentation mask. The output represents the predicted probability of a pixel belonging to the target class (e.g., ship) in the input image.

The deprecated model summary shown below (number of layers is deprecated for the purpose of showing the image):

