
# Stock Price Predictor Using Neural Networks

This project is a Python script that uses a neural network to predict stock price movements based on historical data. The code trains a neural network on stock indices (like Apple and S&P 500), evaluates its accuracy, and makes future predictions on price changes.

## Overview

The primary objectives of this project are:
- Load and preprocess historical stock data for specified indices.
- Train a neural network to predict stock price movements.
- Evaluate the model's accuracy and loss over time.
- Generate visualizations comparing the model's predictions to actual stock price changes.
- Predict the stock price change for the following trading day.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Script Details](#script-details)
- [Notes](#notes)
- [License](#license)

## Prerequisites

Before running the script, ensure you have the following installed:
- Python 3.x
- Necessary Python libraries such as `matplotlib`, `numpy`, and others listed in `requirements.txt`.
- Custom modules: `datahandler` and `nethandler`.

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Install Dependencies**
   Ensure all the required Python packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Input Data Parameters
- **LAG_DAYS**: Number of days to use as input features for prediction.
- **startdate**: Start date for downloading the stock data (format: `YYYYMMDD`).
- **indices**: List of stock symbols to use as input features (e.g., `["AAPL", "^GSPC"]`).

### Neural Network Parameters
- **HIDDEN**: Number of hidden units in the neural network.
- **OUTPUT**: Number of output neurons (typically 1 for price change prediction).

### Training Parameters
- **EPOCHS**: Number of epochs for training the neural network.
- **LEARNING_RATE**: Learning rate for the training algorithm.
- **MOMENTUM**: Momentum factor for the training algorithm.

## Usage

To run the script:
```bash
python main.py
```

This will execute the following steps:
1. Load the stock data based on the indices provided.
2. Create and train a neural network model.
3. Evaluate the model's accuracy and plot results.
4. Predict stock price change for the next trading day.

### Output
- Two plots are generated and saved as:
  - **prediction_plot.png**: Comparison of actual vs predicted stock price changes.
  - **error_plot.png**: Training and validation error over epochs.

### Modules Breakdown

1. **Data Loading** (`dh.DataHandler`): The `DataHandler` class handles the loading and preprocessing of historical stock data. It fetches stock prices, applies lag to create features, and splits the data into training and testing sets.

2. **Neural Network Handling** (`NetHandler`): The `NetHandler` class builds and manages a neural network. It handles training, error calculation, prediction, and model evaluation.

3. **Training and Evaluation**:
   - The model is trained using the specified parameters (epochs, learning rate, momentum).
   - Training and validation errors are calculated over epochs, and the results are plotted.

4. **Prediction and Accuracy**:
   - After training, the model makes predictions on the test data.
   - The script calculates the directional accuracy — the percentage of times the model correctly predicted the direction of stock price change.
   - The prediction for the next trading day's price change is also computed.

## Notes

- **Data Handling**: Ensure that `datahandler` and `nethandler` are properly implemented and imported. The script assumes these modules handle data processing and neural network operations.
- **Hyperparameter Tuning**: The neural network parameters (like `HIDDEN`, `LEARNING_RATE`, etc.) can be adjusted based on the data and desired performance.
- **Plot Generation**: Generated plots for actual vs. predicted values and error over epochs are saved automatically in the project directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
