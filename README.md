# diabetespredictor
An ML model trained on PIMA dataset (from Kaggle) , which tries to predict whether a person has diabetes or not


# Diabetes Predictor

This Python code implements a simple diabetes predictor using the Support Vector Machine (SVM) algorithm. It uses the PIMA Diabetes Dataset to train the SVM model and make predictions on new input data.

## Prerequisites

Before running this code, ensure you have the following installed:

- Python 3.x
- Required Python libraries: numpy, pandas, sklearn

You can install the required libraries using `pip`:
```bash
pip install numpy pandas scikit-learn
```

## How to Use

1. Clone or download the repository to your local machine.

2. Ensure that the `diabetes.csv` file is present in the same directory as the Python script.

3. Run the Python script. The script will perform the following steps:

## Data Collection and Analysis

- Read the CSV file `diabetes.csv` using pandas.
- Display the first few rows and a summary of the dataset.
- Check the distribution of the target variable 'Outcome' (0 for non-diabetic, 1 for diabetic) to understand the class distribution.

## Data Standardization

- Standardize the features in the dataset using StandardScaler from sklearn.

## Training and Testing Split

- Split the dataset into training and testing sets for model evaluation.

## Training the Model

- Train an SVM classifier with a linear kernel using the training data.

## Model Evaluation

- Calculate the accuracy of the model on both the training and testing datasets.

## Making a Predictive System

- Create an example input data representing a person's medical information.
- Standardize the input data using the previously fitted scaler.
- Make a prediction on the standardized input data using the trained SVM model.
- Print whether the person is diabetic or not based on the prediction.

Please note that this is a basic diabetes predictor using a linear SVM model. The model's accuracy and performance can be improved by using more advanced techniques, feature engineering, hyperparameter tuning, and exploring different machine learning algorithms.

For more details on the implementation, you can refer to the original Colaboratory notebook where this code was generated:
[Original Colaboratory Notebook](https://colab.research.google.com/drive/1HMZkt79sCvEJfgebvhazzsAQFjEGpy_B)
