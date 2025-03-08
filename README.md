# Image Recognition Project

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![scikit-image](https://img.shields.io/badge/scikit--image-latest-yellow.svg)
![NumPy](https://img.shields.io/badge/NumPy-latest-blue.svg)

A machine learning project that implements and compares various classification algorithms for image recognition tasks. This project provides a comprehensive framework for preprocessing images and evaluating multiple ML models.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Performance Optimization](#performance-optimization)
- [Future Improvements](#future-improvements)

## Overview

This project demonstrates the implementation and comparison of various machine learning algorithms for image classification. Each algorithm is tuned using cross-validation to find optimal hyperparameters, and performance is measured using precision scores.

## Key Features

- Automated image preprocessing pipeline
- K-fold cross-validation for model tuning
- Performance comparison across multiple algorithms
- Execution time measurement
- Standardized output format for easy comparison

## Algorithms Implemented

| Algorithm | Abbreviation | Description | Best For |
|-----------|--------------|-------------|----------|
| Random Forest | rf | Ensemble method using multiple decision trees | Complex datasets with many features |
| K-Nearest Neighbors | knn | Instance-based learning using distance metrics | Small to medium datasets with clear patterns |
| Decision Tree | dt | Tree-based classification model | Datasets with discrete features |
| Bagging Classifier | bg | Ensemble meta-estimator using bootstrapped samples | Reducing variance in unstable models |
| AdaBoost | ab | Boosting ensemble that focuses on misclassified samples | Improving weak classifiers |

## Project Structure

```
Image_Recognition/
├── src/                       # Source code
│   ├── main.py                # Main entry point
│   ├── rf_recognition.py      # Random Forest implementation
│   ├── knn_recognition.py     # K-Nearest Neighbors implementation
│   ├── dt_recognition.py      # Decision Tree implementation
│   ├── bg_recognition.py      # Bagging Classifier implementation
│   └── ab_recognition.py      # AdaBoost implementation
├── training/                  # Training dataset
│   └── [class_folders]/       # Image classes organized by folder
├── testing/                   # Test dataset
└── output/                    # Generated predictions
    └── *_predictions.txt      # Prediction results for each algorithm
```

## Requirements

- Python 3.6 or higher
- OpenCV
- scikit-learn
- scikit-image
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image_Recognition.git
cd Image_Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file, use:
```bash
pip install opencv-python scikit-image scikit-learn numpy
```

## Usage

1. Organize your dataset:
   - Place training images in `training/[class_name]/` directories
   - Place test images in the `testing/` directory

2. Run an algorithm using the main script:
```bash
python src/main.py <algorithm>
```

Where `<algorithm>` is one of:
- `rf` - Random Forest
- `knn` - K-Nearest Neighbors
- `dt` - Decision Tree
- `bg` - Bagging Classifier
- `ab` - AdaBoost

3. View results:
   - Predictions will be saved to `output/<algorithm>_predictions.txt`
   - Console output will show model performance metrics and execution time

Example command:
```bash
python src/main.py rf
```

## Model Evaluation

Each algorithm is evaluated using:
- 5-fold cross-validation
- Macro-averaged precision score
- Execution time measurement

Hyperparameters are tuned to maximize precision:
- Random Forest: Number of estimators (1-50)
- KNN: Number of neighbors (1-50)
- Decision Tree: Maximum depth (1-50)
- Bagging: Number of estimators (1-50)
- AdaBoost: Number of estimators (1-50)

## Performance Optimization

- Images are resized to 50x50 pixels to reduce computational complexity
- Flattened feature vectors are used for model training
- K-fold cross-validation prevents overfitting
- Hyperparameter tuning ensures optimal model configuration

## Future Improvements

- Implement additional algorithms (SVM, CNN, etc.)
- Add feature extraction techniques (HOG, SIFT, etc.)
- Incorporate data augmentation for improved model generalization
- Implement parallel processing for faster execution
- Add confusion matrix visualization for better model understanding