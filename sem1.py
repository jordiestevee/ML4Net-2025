{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ML4Net Seminar 1: Linear Regression\n",
    "# Part I: Data Preparation & Analysis\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "def load_data(file_path):\n",
    "    data = pd.read_csv(file_path, sep='\\t')\n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display statistics\n",
    "def display_stats(data):\n",
    "    print(data.describe())\n",
    "\n",
    "# Plot data distributions\n",
    "def plot_data_distribution(data):\n",
    "    data.hist(bins=20, figsize=(12, 10))\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Data preprocessing\n",
    "def preprocess_data(data, target_column):\n",
    "    X = data.drop(columns=[target_column])\n",
    "    y = data[target_column]\n",
    "    scaler = StandardScaler()\n",
    "    X_scaled = scaler.fit_transform(X)\n",
    "    return X_scaled, y\n",
    "\n",
    "# Data splitting\n",
    "def split_data(X, y):\n",
    "    return train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'pd' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 58\u001b[0m\n\u001b[0;32m     55\u001b[0m file_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSeminar1.txt\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[0;32m     56\u001b[0m target_column \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPerformance\u001b[39m\u001b[38;5;124m'\u001b[39m  \u001b[38;5;66;03m# Adjust this according to your dataset\u001b[39;00m\n\u001b[1;32m---> 58\u001b[0m data \u001b[38;5;241m=\u001b[39m \u001b[43mload_data\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfile_path\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     60\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mDataset statistics:\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     61\u001b[0m display_stats(data)\n",
      "Cell \u001b[1;32mIn[1], line 3\u001b[0m, in \u001b[0;36mload_data\u001b[1;34m(file_path)\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mload_data\u001b[39m(file_path):\n\u001b[1;32m----> 3\u001b[0m     data \u001b[38;5;241m=\u001b[39m \u001b[43mpd\u001b[49m\u001b[38;5;241m.\u001b[39mread_csv(file_path, sep\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      4\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m data\n",
      "\u001b[1;31mNameError\u001b[0m: name 'pd' is not defined"
     ]
    }
   ],
   "source": [
    "# Part II: Model Implementation\n",
    "\n",
    "def train_model(X_train, y_train):\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train, y_train)\n",
    "    return model\n",
    "\n",
    "# Evaluate model\n",
    "def evaluate_model(model, X_test, y_test):\n",
    "    y_pred = model.predict(X_test)\n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    print(f\"Mean Squared Error: {mse}\")\n",
    "    print(f\"R^2 Score: {r2}\")\n",
    "    return y_pred\n",
    "\n",
    "# Plot model results\n",
    "def plot_model_results(y_test, y_pred):\n",
    "    plt.scatter(y_test, y_pred)\n",
    "    plt.xlabel(\"Actual\")\n",
    "    plt.ylabel(\"Predicted\")\n",
    "    plt.title(\"Actual vs Predicted\")\n",
    "    plt.grid(True)\n",
    "    plt.show()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main execution\n",
    "if __name__ == \"__main__\":\n",
    "    file_path = 'Seminar1.txt'\n",
    "    target_column = 'Performance'  # Adjust this according to your dataset\n",
    "\n",
    "    data = load_data(file_path)\n",
    "\n",
    "    print(\"Dataset statistics:\")\n",
    "    display_stats(data)\n",
    "\n",
    "    print(\"\\nData distribution:\")\n",
    "    plot_data_distribution(data)\n",
    "\n",
    "    X, y = preprocess_data(data, target_column)\n",
    "    X_train, X_test, y_train, y_test = split_data(X, y)\n",
    "\n",
    "    print(\"\\nTraining the model...\")\n",
    "    model = train_model(X_train, y_train)\n",
    "\n",
    "    print(\"\\nEvaluating the model...\")\n",
    "    y_pred = evaluate_model(model, X_test, y_test)\n",
    "\n",
    "    plot_model_results(y_test, y_pred)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
