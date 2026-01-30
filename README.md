# Logistic Regression with a Neural Network Mindset

This project implements **logistic regression** for cat vs non-cat classification using a neural network mindset.

## Features
- Load and preprocess images
- Train logistic regression
- Predict custom images
- Plot learning curve

## Project Structure
- `src/model.py` → Functions for logistic regression
- `src/main.py` → Training, testing, prediction
- `lr_utils.py` → Dataset loader
- `data/catvnoncat.h5` → Dataset
- `images/` → Example images

## How to Run
1. Install libraries:
```bash
pip install numpy matplotlib pillow h5py
