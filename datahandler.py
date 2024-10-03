import os.path
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self):
        self.sp = pd.DataFrame()
        self.filename = ''
        self.tickers = ''
        self.startdate = ''
        self.enddate = ''
        self.dataframe = pd.DataFrame()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_indices(self, tickers, startdate, lags):
        self.tickers = tickers
        self.filename = r"C:\Users\Juan Diego\Documents\Forex_Training_models_repositori\ai-financialANN-master\ai-financialANN-master\DATA.csv"
        self.startdate = startdate
        self.enddate = datetime.date.today().strftime("%Y-%m-%d")
        
        print(f"Loading data for tickers: {tickers}")
        print(f"Start date: {startdate}")
        print(f"End date: {self.enddate}")

        if os.path.isfile(self.filename):
            print(f"Loading data from file: {self.filename}")
            self.dataframe = pd.read_csv(self.filename, index_col=0, parse_dates=True)
            print(f"Loaded data shape: {self.dataframe.shape}")
        else:
            print("Downloading data from yfinance")
            for ticker in tickers:
                try:
                    data = yf.download(ticker, start=self.startdate, end=self.enddate)
                    print(f"Downloaded data for {ticker}. Shape: {data.shape}")
                    if data.empty:
                        print(f"No data available for {ticker}")
                        continue
                    
                    index = ticker + '1change'
                    data[index] = data['Adj Close'].pct_change(1)
                    data = data[[index]].dropna()
                    
                    data = data.apply(self.preprocess)
                    
                    for i in range(1, lags + 1):
                        label = f"{ticker}{i}lag"
                        data[label] = data[index].shift(i)
                    
                    data = data.dropna()
                    print(f"Processed data for {ticker}. Shape: {data.shape}")
                    
                    if self.sp.empty:
                        self.sp = data
                    else:
                        self.sp = pd.merge(self.sp, data, left_index=True, right_index=True, how='outer')
                
                except Exception as e:
                    print(f"Error downloading {ticker}: {str(e)}")
            
            self.dataframe = self.sp
            if not self.dataframe.empty:
                print(f"Saving data to file: {self.filename}")
                self.dataframe.to_csv(self.filename)
                print(f"Saved data shape: {self.dataframe.shape}")
            else:
                print("No data was loaded or processed.")

        if self.dataframe.empty:
            print("Warning: No data was loaded or processed.")
        else:
            print(f"Final data shape: {self.dataframe.shape}")


    def create_data(self, inputs, targets):
        if self.dataframe.empty:
            raise ValueError("No data available. Please load indices first.")
        
        X = self.dataframe.iloc[:, 1:].values  # All columns except the first one
        y = self.dataframe.iloc[:, 0].values   # First column
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def get_datasets(self):
        if self.X_train is None:
            raise ValueError("Data not created yet. Please call create_data() first.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self, vals):
        outs = np.log(np.abs(vals * 100) + 1) * np.sign(vals)
        scale = 1.8 / (np.max(outs) - np.min(outs))
        outs = (scale * (outs - np.min(outs))) - 0.9
        return outs - np.mean(outs)

handler = DataHandler()
handler.load_indices(['AAPL', '^GSPC'], '2020-01-01', 5)
handler.create_data(5, 1)
X_train, X_test, y_train, y_test = handler.get_datasets()

# Create and train a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")