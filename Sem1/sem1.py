# ML4Net Seminar 1: Linear Regression
# Part I: Data Preparation & Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data

# Display statistics
def display_stats(data):
    print(data.describe())

# Plot data distributions
def plot_data_distribution(data):
    data.hist(bins=20, figsize=(12, 10))
    plt.tight_layout()
    plt.show()

# Data preprocessing
def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Data splitting
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Part II: Model Implementation

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return y_pred

# Plot model results
def plot_model_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = 'Seminar1.txt'
    target_column = 'Performance'  # Adjust this according to your dataset

    data = load_data(file_path)

    print("Dataset statistics:")
    display_stats(data)

    print("\nData distribution:")
    plot_data_distribution(data)

    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining the model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating the model...")
    y_pred = evaluate_model(model, X_test, y_test)

    plot_model_results(y_test, y_pred)
