# Integrating IDNet and Residual Blocks for Efficient Diabetic Retinopathy Grading

## Project Overview

This project introduces a novel approach to diagnosing Diabetic Retinopathy (DR) using advanced deep learning models, specifically Inception-Deconvolutional (IDNet) and Residual networks (ResNet). Diabetic Retinopathy is a significant ocular condition linked to diabetes mellitus, necessitating early detection for effective intervention. This project aims to automate the grading process, improving accuracy and reducing the resource intensity of manual diagnoses.

## Table of Contents

- [Project Description](#project-description)
- [Architecture](#architecture)
- [Data Preparation](#data-preparation)
- [Pre-Processing](#pre-processing)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Comparison with Other Models](#comparison-with-other-models)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [License](#license)
- [Contact](#contact)

## Project Description

The project integrates IDNet and Residual Blocks to efficiently grade Diabetic Retinopathy from fundus images. Key components include:

- **IDNet**: For extracting multi-scale features.
- **Residual Blocks**: For improved gradient flow and feature learning.
- **Deconvolutional Layers**: For upsampling and feature refinement.

The model leverages fundamental data augmentation techniques and advanced preprocessing methods to ensure robustness and accuracy in DR diagnosis.

## Architecture

The architecture combines Inception modules, deconvolutional layers, and residual networks:

- **Inception Modules**: Extract multi-scale features from retinal images.
- **Residual Blocks**: Alleviate vanishing gradient problems and enhance training stability.
- **Deconvolutional Layers**: Restore spatial details lost during downsampling.

![Architecture Diagram](link-to-your-architecture-diagram)

## Data Preparation

The dataset comprises fundus images sourced from Messidor, Messidor-2, DRISHTI-GS, and GitHub. Images are classified into stages of Diabetic Retinopathy:

- Grade 0: No DR
- Grade 1: Mild DR
- Grade 2: Moderate DR
- Grade 3: Severe DR
- Grade 4: Proliferative DR

Data preparation involved normalization, noise reduction, and manual annotation by expert ophthalmologists. Data augmentation techniques were applied to enhance model generalization.

## Pre-Processing

Preprocessing steps include:

- **Contrast Enhancement**: Using CLAHE for improved visibility of critical areas.
- **Illumination Correction**: To normalize lighting conditions.
- **Image Segmentation**: For accurate object detection.
- **Data Augmentation**: Techniques such as flipping and rotation to boost training data.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/DR-Detection.git
## Setup Instructions

* **Navigate to the Project Directory**:

    ```bash
    cd DR-Detection
    ```

* **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
## Usage

1. **Prepare Your Data**: Place your fundus images in the appropriate folders.

2. **Run the Training Script**:

    ```bash
    python train.py --data_dir path/to/your/data
    ```

3. **Evaluate the Model**:

    ```bash
    python evaluate.py --model_path path/to/your/saved/model
    ```
## Results

The model demonstrated strong performance with the following evaluation metrics:

- **Precision**: 0.924
- **Recall**: 0.955
- **F1-score**: 0.931
- **ROC AUC**: 0.946

The confusion matrix and performance metrics indicate robust classification capabilities across various stages of Diabetic Retinopathy.

## Comparison with Other Models

| Reference | DL Method                                 | AUC   | Accuracy |
|-----------|--------------------------------------------|-------|----------|
| [20]      | CNN (Inception-v3)                        | –     | 97.5%    |
| [21]      | CNN                                        | 0.980 | 87.0%    |
| [22]      | CNN                                        | –     | 75%      |
| [23]      | CNN                                        | –     | 94.5%    |
| [25]      | CNN-ResNet34                               | –     | 85%      |
| [27]      | CNN (InceptionNet V3, AlexNet, and VGG16) | –     | –        |

## Conclusion and Future Work

The integration of IDNet and Residual Blocks has significantly improved DR grading accuracy. Future work will focus on:

- Expanding the dataset for better generalization.
- Enhancing model performance with advanced architectures.
- Exploring real-time diagnostic applications.
## Contact

* **A.V.S.S.Nanditha**: nanditha.akkisetty@gmail.com
* **Talluri Laahiri**: laahiri1684@gmail.com
* **Mohammed Nurulla**: mohammednurulla1@gmail.com
