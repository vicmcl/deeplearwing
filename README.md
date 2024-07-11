# DeepLearWing

## Introduction

This project uses a Convolutional Neural Network (CNN) to predict the lift, drag and moment coefficients of 2D airfoils. The model is trained on a dataset of 1600 airfoils, which geometry and aerodynamics performances were scraped from [Airfoil Tools](http://airfoiltools.com/). 

## Deep Learning for Aerodynamics

In aerodynamics, deep learning is a relevant tool to complement CFD simulations and enrich results analysis, via the following aspects:

* **Handling High-Dimensional Data**
Aerodynamics simulations involve complex interactions between airfoils, turbulence, and other non-linear factors. Deep learning models can effectively handle this complexity by learning patterns and relationships in the data. This allows to capture subtle effects that may not be easily captured by traditional methods.

* **Scalability**
Deep learning models can be trained on large datasets, making them well-suited for large-scale aerodynamics simulations. This scalability allows to predict aerodynamics performances for a wide range of airfoils and operating conditions, without the need for extensive computational resources.

* **Flexibility**
Deep learning models can be designed to accommodate different types of data, including numerical simulations, experimental data, and even physical laws (via PINNs). This flexibility allows to integrate multiple sources of information and deliver more reliable results.

* **Integration with CFD Simulations**
Deep learning models can be used to augment and refine CFD simulations, providing a hybrid approach that leverages the strengths of both methods. By combining the physical insights from CFD simulations with the predictive power of deep learning, one can create a more accurate and efficient approach for aerodynamics predictions.

* **Reduced Computational Cost**
Deep learning models can be trained on a subset of the data and then used to make predictions on new, unseen data. This reduces the computational cost of simulations to make predictions more quickly and efficiently.


## Dataset

The dataset was scraped from [Airfoil Tools](http://airfoiltools.com/). This online public repository gathers airfoil data created using XFOIL, a program for design and analysis of subsonic airfoils. The scraped data consists of 1600 airfoils, each represented by its:

* Geometry: 2D coordinates of the airfoil shape
* Aerodynamics performances: lift, drag and moment coefficients (respectively Cl, Cd and Cm)

For each airfoil, these performances are given for a range of Reynolds numbers (from 50 000 to 1 000 000) and angles of attack (from -10 to +10 deg for most airfoils) resulting in 800k+ samples. The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/victorienmichel/deeplearwing)


## Model Architecture

The CNN model architecture is determined using Keras Tuner with the HyperBand algorithm, to determine hyperparameters such as:
* Number of convolution layers
* Number of dense layers
* Number of filters
* Kernel sizes
* Inclusion of batch normalization and dropout layers

## Training

The model was trained using the Adam optimizer and mean squared error (MSE) loss function.

## Evaluation

The model was evaluated on a test set of 200 airfoils, and achieved a mean absolute error (MAE) of 0.05 for lift coefficient and 0.03 for drag coefficient.
