# Airbus Ship Detection Challenge

## Table of Contents

1. [Overview](#overview)
2. [Challenge Description](#challenge-description)
3. [Repository Structure](#repository-structure)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Model Training and Evaluation](#model-training-and-evaluation)
   1. [Training](#training)
   2. [Evaluation](#evaluation)

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
   git clone https://github.com/poinkter/Airbus-Ship-Detection.git
   ```

2. Set up the environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/airbus-ship-detection) and place it in the `ship_detection_data/` directory.

4. Run the `model_train.py` for training the model. Run the `model_inference.py` to evaluate the model on the test images.

## Model Architecture

The implemented model follows a modified U-Net architecture, a popular choice for semantic segmentation tasks. The architecture consists of a contracting path, a U-Net bottom, and an expansive path, designed to capture both high-level context and precise spatial information.

- Contracting Path

  - Begins with an input layer
  - Sequence of convolutional blocks, each with:
    - Two convolutional layers with ReLU activation
    - Max-pooling layers (2x2) after each convolutional block to reduce spatial dimensions

- U-Net Bottom

  - Acts as a bottleneck
  - Contains a convolutional block with increased filter size to retain rich feature information

- Expansive Path

  - Consists of transposed convolutions (Conv2DTranspose) to upsample feature maps
  - Utilizes skip connections by concatenating feature maps from the contracting path
  - Followed by convolutional blocks similar to those in the contracting path after each upsampling step

- Output Layer

  - Final layer is a 1x1 convolutional layer with sigmoid activation
  - Produces a binary segmentation mask, representing the predicted probability of pixels belonging to the target class (e.g., ship) in the input image.

#### Customization

The model's flexibility is enhanced through the adjustable parameter `n_filters`, allowing users to control the number of filters in the convolutional and transposed convolutional layers.


The deprecated model summary shown below (number of layers is deprecated for the purpose of showing the image):![Example Image](images/model_summary_deprecated.png)

## Model Training and Evaluation

### Training

The U-Net model was trained using the Dice Loss (Diceloss) as the loss function. The training process involved optimizing the model parameters to minimize this loss, ultimately enhancing its ability to accurately segment ships in satellite images. Due to the computational intensity of training, the model was not trained locally on my computer, but instead on a machine with sufficient resources.

### Evaluation

The model's performance can be assessed using the provided evaluation script. Here's an overview of the evaluation process:

1. The trained model is loaded from the saved file (`model_diceloss.h5`).
2. Sample images from the test dataset (`./ship_detection_data/test_v2`) are loaded for inference.
3. The model predicts segmentation masks for the test images.
4. Optionally, the original images and predicted masks can be displayed side by side for visual inspection (set `display_images=True` in the `predict_masks` function).
5. The predicted masks are converted to Run-Length Encoding (RLE) format.
6. The results are saved in a CSV file (`submission.csv`), located in the specified save path (`./ship_detection_data/`).

Feel free to run the evaluation script locally to assess the model's performance and visualize the predictions.

Ensure that the necessary dependencies are installed, as listed in the `requirements.txt` file.

**Note:** Make sure to replace placeholders such as `./ship_detection_data/` with the actual paths relevant to your project.

For detailed visualizations, refer to the `display_images` and `predict_masks` functions in the evaluation script.
