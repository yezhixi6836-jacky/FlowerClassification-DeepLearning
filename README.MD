# Flower Classification Using Deep Learning

## Project Overview

This project focuses on 102-class flower image classification using deep learning.

The project compares different architectures, including:

- Basic CNN
- ResNet18
- EfficientNet-B0
- Vision Transformer (ViT)

The project also includes Grad-CAM visualization and an interactive Streamlit application.

---

## Dataset

The project uses the Oxford Flowers 102 dataset.

The dataset contains 102 flower categories with different visual characteristics such as color, shape, texture, and background.

The dataset itself is not included in this repository because of its size.

---

## Models

### 1. CNN

A basic convolutional neural network used as the baseline model.

### 2. ResNet18

A residual CNN architecture using shortcut connections.

### 3. EfficientNet-B0

An efficient CNN architecture designed to balance accuracy and computational cost.

### 4. Vision Transformer

A Transformer-based architecture that represents an image as a sequence of image patches.

---

## Project Pipeline

Image
↓
Preprocessing
↓
Tensor
↓
Feature / Patch Encoding
↓
Deep Learning Model
↓
Classification
↓
102 Flower Classes

---

## Explainability

Grad-CAM is used to visualize the important image regions contributing to CNN-based predictions.

---

## Application

A Streamlit application was developed to allow users to upload flower images and obtain:

- Top-1 prediction
- Confidence score
- Top-5 predictions
- Grad-CAM visualization

---

## Project Structure

```text
FlowerClassification-DeepLearning/
│
├── app.py
├── README.md
├── requirements.txt
│
├── notebooks/
├── models/
├── utils/
├── weights/
├── results/
└── demo_images/