# Image Classification with MobileNetV2 and TensorFlow

## Introduction

This Python script is designed for image classification using a pre-trained MobileNetV2 model in TensorFlow. It finds applications in various real-world scenarios where image recognition and categorization are essential. The code can be adapted for a wide range of use cases, making it a valuable tool for machine learning and computer vision applications.

### Use Case: Medical Image Classification

One compelling use case for this code is in the field of medical image classification. Medical professionals and researchers often deal with vast amounts of medical imaging data, including X-rays, CT scans, MRIs, and pathology images. Using deep learning techniques, such as the MobileNetV2 model implemented here, these medical images can be categorized into different disease classes or conditions.

By adapting this code to specific medical imaging datasets and disease categories, it becomes a powerful tool for automating the analysis of medical images, assisting healthcare professionals, and improving patient care. Its flexibility and ease of use make it a valuable asset in the development of AI-driven solutions for medical image classification and diagnosis.

## Usage of Intel oneAPI for TensorFlow and sklearn Patch

### Intel oneAPI for TensorFlow
![Intel oneAPI for TensorFlow](https://user-images.githubusercontent.com/104119642/225605322-bf599380-9eea-4a8c-beb8-0b4ec467efd9.png)
This code takes advantage of Intel oneAPI for TensorFlow, an optimized version of TensorFlow designed to leverage Intel hardware acceleration, specifically oneDNN (Deep Neural Network Library). The following lines in the code enable the use of oneAPI for TensorFlow:

```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
```
### sklearn Patch

In addition to TensorFlow, the code utilizes the `sklearnex` package to patch the scikit-learn (sklearn) library. The `sklearnex` package extends scikit-learn to seamlessly integrate with TensorFlow models. While the specific usage of scikit-learn features is not shown in this code, you can easily incorporate scikit-learn preprocessing and evaluation steps by following the scikit-learn documentation.

### How to Patch scikit-learn using sklearnex

To enable the integration of scikit-learn with TensorFlow models, the following lines of code demonstrate how to patch scikit-learn using the `sklearnex` package:

```python
from sklearnex import patch_sklearn
patch_sklearn()
```
## Comparing Accuracy and Runtime

In this section, we compare the accuracy and runtime of the model when using Intel-optimized TensorFlow and regular TensorFlow.

### Model Accuracy Comparison

Here, we present a comparison of the model's accuracy when trained with both Intel-optimized TensorFlow and regular TensorFlow. The image below illustrates the accuracy achieved by each configuration.


### Runtime Comparison

We also measure and compare the runtime of the model training process for both configurations. The following table provides runtime statistics for each setup:

| Configuration            | Runtime (seconds) |
|--------------------------|--------------------|
| Intel-optimized TensorFlow | 789.04               |
| Regular TensorFlow        | 796.27               |

The results show the runtime differences between using Intel-optimized TensorFlow and regular TensorFlow for training your image classification model.

## Accuracy comparison

<img src='https://github.com/balasuriyaranganathan/brain_tumor_classification/blob/main/1.png'>

<img src='https://github.com/balasuriyaranganathan/brain_tumor_classification/blob/main/2.png'>

##compiling the code in intel dev cloud further gives better accuracy
<img src='https://github.com/balasuriyaranganathan/brain_tumor_classification/blob/main/3.png'>
