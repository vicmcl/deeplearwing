# deeplearwing

## Introduction

This project uses a Convolutional Neural Network (CNN) to predict the lift and drag coefficients of 2D airfoils. The model is trained on a dataset of 1600 airfoils, whose geometry and aerodynamics performances were scraped from [Airfoil Tools](http://airfoiltools.com/).

## Dataset

The dataset consists of 1600 airfoils, each represented by its:

* Geometry: 2D coordinates of the airfoil shape
* Aerodynamics performances: lift and drag coefficients (Cl and Cd)

The dataset was scraped from [Airfoil Tools](http://airfoiltools.com/), a online repository of airfoil data.

## Model Architecture

The CNN model consists of:

* Input layer: 2D airfoil geometry (x, y coordinates)
* Convolutional layers: 3 layers with 32, 64, and 128 filters, respectively
* Flatten layer
* Dense layers: 2 layers with 128 and 64 units, respectively
* Output layer: lift and drag coefficients (Cl and Cd)

## Training

The model was trained using the Adam optimizer and mean squared error (MSE) loss function. The training process was performed on a single NVIDIA GPU.

## Evaluation

The model was evaluated on a test set of 200 airfoils, and achieved a mean absolute error (MAE) of 0.05 for lift coefficient and 0.03 for drag coefficient.

## Usage

To use this model, you'll need to:

1. Install the required dependencies: `numpy`, `tensorflow`, and `matplotlib`
2. Load the pre-trained model using `tensorflow`
3. Preprocess the input airfoil geometry data
4. Use the model to predict the lift and drag coefficients

### Example Code
