from math import floor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from sklearn.preprocessing import StandardScaler

class NetHandler:
    def __init__(self, INS, HIDDEN, OUT, data):
        self.INS = INS
        self.HIDDEN = HIDDEN
        self.OUTS = OUT
        self.indata = data.dataframe
        self.model = self.assemble_rn()
        self.scaler = StandardScaler()
        self.scaler.fit(self.indata.iloc[:, :self.INS])

    def assemble_rn(self):
        model = Sequential([
            LSTM(self.HIDDEN, input_shape=(1, self.INS), return_sequences=False),
            Dense(self.OUTS, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self, data, LRATE, MOMENTUM, ITERATIONS):
        print("Training...")
        print(f"Input data shape: {data.shape}")
        X = data[:, :self.INS]
        y = data[:, self.INS:]
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = X_scaled.reshape(-1, 1, self.INS)
        
        print(f"X_scaled shape: {X_scaled.shape}")
        
        history = self.model.fit(
            X_scaled, y,
            epochs=ITERATIONS,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        return history.history['loss'], history.history['val_loss']

    def get_output(self, data):
        print(f"Input data shape in get_output: {data.shape}")
        X = data
        print(f"X shape: {X.shape}")
        X_scaled = self.scaler.transform(X)
        
        # Reshape for LSTM input
        X_scaled = X_scaled.reshape(-1, 1, self.INS)
        print(f"X_scaled shape: {X_scaled.shape}")
        
        outputs = self.model.predict(X_scaled).flatten()
        
        return outputs

    def predict_tomorrow(self):
        index = len(self.indata) - 1
        ins = self.indata.iloc[index].values[:self.INS].reshape(1, -1)
        ins_scaled = self.scaler.transform(ins)
        ins_scaled = ins_scaled.reshape(1, 1, -1)
        
        output = self.model.predict(ins_scaled)
        tomorchange = output[0][0]
        
        tomorrow = (self.indata.index[index] + BDay()).to_pydatetime().strftime("%a, %b %d, %Y")
        
        if tomorchange > 0:
            return f"\nOn {tomorrow} the market will increase.\n"
        else:
            return f"\nOn {tomorrow} the market will decrease.\n"

    def evaluate(self, test_data):
        print(f"Test data shape: {test_data.shape}")
        X_test = test_data[:, :self.INS]
        y_test = test_data[:, self.INS:]

        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = X_test_scaled.reshape(-1, 1, self.INS)

        print(f"X_test_scaled shape: {X_test_scaled.shape}")

        loss = self.model.evaluate(X_test_scaled, y_test)
        print(f"Test loss: {loss}")
        return loss