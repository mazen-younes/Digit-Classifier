# Digit Classifier

## Project Overview

This project is a **machine learning system** designed to recognize handwritten digits from 0 to 9. It uses a **Convolutional Neural Network (CNN)** trained on grayscale images of digits. The model learns the unique patterns of handwritten numbers and can accurately predict the digit in a new image.

The project covers the **full workflow**:

* Loading and preprocessing the data
* Building the CNN model
* Training the model
* Evaluating its performance
* Making real-time predictions

Additionally, it provides interactive tools to **draw a digit or upload an image** for classification.

---

## Features

* Complete **end-to-end digit classification system**
* **CNN model** built with TensorFlow/Keras
* **Data preprocessing** and normalization for consistent inputs
* **Model training** with monitoring and evaluation
* **Confusion matrix** and classification report for performance insights
* **Interactive prediction tools**: draw digits or upload images

---

## How to Run the Project

### 1. Install Dependencies

Ensure Python is installed, then run:

```
pip install tensorflow numpy matplotlib scikit-learn ipywidgets ipycanvas pillow seaborn
```

---

### 2. Open the Notebook

Open `Digits_CNN.ipynb` in either:

* **Jupyter Notebook**, or
* **Google Colab** (recommended for GPU support)

---

### 3. Data Preparation

* Run the **data loading and preprocessing section** to prepare the dataset.
* The dataset will be:

  * Extracted from the zip file
  * Resized to 28x28 pixels
  * Normalized to values between 0 and 1
  * Split into **training**, **validation**, and **test sets**

---

### 4. Train the Model

* Run the **CNN model training section**.
* The model will train using the training data and validate on the validation set.
* EarlyStopping callback is included to prevent overfitting.

---

### 5. Evaluate the Model

* Run the **evaluation section** to check:

  * Training & validation accuracy
  * Loss
  * Confusion matrix
  * Classification report

This helps identify which digits are classified well and which need improvement.

---

### 6. Make Predictions

The trained model can predict digits in two ways:

1. **Draw a Digit**

   * Use the interactive canvas to draw a digit.
   * The model will preprocess your drawing and output the predicted digit with confidence scores.

2. **Upload an Image**

   * Upload a 28x28 grayscale image of a digit.
   * The model will preprocess and classify it, showing prediction and confidence for each digit.

---

## Purpose of the Project

This project demonstrates a **complete machine learning pipeline**:

* Data preparation and preprocessing
* Building and training a neural network (CNN)
* Evaluating model performance
* Making real-time predictions with an interactive tool

It is ideal for learning about **image classification**, **CNNs**, and **end-to-end ML workflows**.

---

## Notes

* Ensure uploaded images are **28x28 grayscale** for accurate predictions.
* The interactive tool is designed for both **drawing** and **image uploads**.
* Misclassifications may occur for some digits; the model can be improved using **data augmentation** or more **complex architectures**.
