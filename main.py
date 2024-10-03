import matplotlib.pyplot as plt
import datahandler as dh
from nethandler import NetHandler
import numpy as np

def main():
    print("Script started")

    try:
        # Input Data
        LAG_DAYS = 3
        startdate = '20000101'  # YYYYMMDD
        indices = ["AAPL", "^GSPC"]  # Apple and S&P 500

        # Neural Network
        HIDDEN = 12
        OUTPUT = 1

        # Training
        EPOCHS = 100
        LEARNING_RATE = 0.001
        MOMENTUM = 0.9

        print("Loading data...")
        data = dh.DataHandler()
        data.load_indices(indices, startdate, LAG_DAYS)
        data.create_data(len(indices) * (LAG_DAYS + 1), OUTPUT)
        X_train, X_test, y_train, y_test = data.get_datasets()
        print(f"Data loaded successfully. Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        INPUT = X_train.shape[1]  # Number of input features
        print(f"INPUT size: {INPUT}")

        print("Creating neural network...")
        sp_net = NetHandler(INPUT, HIDDEN, OUTPUT, data)
        print("Neural network created")

        print("Starting training...")
        train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        train_errors, val_errors = sp_net.train(train_data, LEARNING_RATE, MOMENTUM, EPOCHS)
        print("Training completed")

        print("Evaluating model...")
        test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
        test_loss = sp_net.evaluate(test_data)
        print(f"Test loss: {test_loss}")

        print("Generating predictions...")
        predictions = sp_net.get_output(X_test)

        print("Calculating directional accuracy...")
        actual = y_test
        correct = np.sum(np.sign(predictions) == np.sign(actual))
        total = len(predictions)
        accuracy = (correct / total) * 100
        print(f"{accuracy:.2f}% Directional Accuracy")

        print("Generating plots...")
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual)), actual, label='Actual')
        plt.plot(range(len(predictions)), predictions, label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Price Change')
        plt.savefig('prediction_plot.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(train_errors, label='Training Error')
        plt.plot(val_errors, label='Validation Error')
        plt.legend()
        plt.title('Training and Validation Errors')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.savefig('error_plot.png')
        plt.close()

        print("Plots saved as 'prediction_plot.png' and 'error_plot.png'")

        print("Predicting change for tomorrow...")
        tomorrow_change = sp_net.predict_tomorrow()
        print(tomorrow_change)

        print("Script completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()