import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess the data
class H_DATA:
    def __init__(self,s_year,e_year):
        self.data = pd.read_csv('datas/omnidata8.txt', sep="\s+", skiprows=60, header=None)
        self.data=self.data[(self.data[0]>=s_year)&(self.data[0]<=e_year)]
        self.bz_gsm = self.data[15]
        self.placeholder_value = 999.9
        self.bz_gsm_cleaned = self.bz_gsm[self.bz_gsm != self.placeholder_value].values.reshape(-1, 1)

        # Scale the data to [0, 1] range
        self.scaler = MinMaxScaler()
        self.bz_gsm_scaled = self.scaler.fit_transform(self.bz_gsm_cleaned)

        # Define the enhanced autoencoder model
        self.input_dim = self.bz_gsm_scaled.shape[1]
        self.encoding_dim = 8  # increased the encoding dimension

        self.autoencoder = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(self.encoding_dim, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.input_dim, activation='sigmoid')
        ])
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Early Stopping callback
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the enhanced autoencoder
        self.autoencoder.fit(self.bz_gsm_scaled, self.bz_gsm_scaled, epochs=100, batch_size=256, shuffle=True, validation_split=0.2, callbacks=[self.early_stopping])

        # Calculate reconstruction errors
        self.reconstructed_data = self.autoencoder.predict(self.bz_gsm_scaled)
        self.mse = np.mean(np.power(self.bz_gsm_scaled - self.reconstructed_data, 2), axis=1)

        # ... [previous code for loading data, training the enhanced autoencoder, and calculating reconstruction errors]

        # Set a threshold for anomalies using the 95th percentile
        self.threshold = np.percentile(self.mse, 95)

        # Anomalies are data points with a reconstruction error above the threshold
        self.anomaly_indices = np.where(self.mse > self.threshold)[0]

        # Calculate the intervals between consecutive anomalies in hours
        self.intervals_in_hours = np.diff(self.anomaly_indices)

        # Convert the average interval from hours to days, hours, and minutes
        self.average_interval_in_days = np.mean(self.intervals_in_hours) // 24
        self.average_interval_in_hours_remainder = np.mean(self.intervals_in_hours) % 24
        self.average_interval_in_minutes = (np.mean(self.intervals_in_hours) % 1) * 60
    def print_dates(self):
            return f"Average interval between anomalies: {int(self.average_interval_in_days)} days, {int(self.average_interval_in_hours_remainder)} hours, and {int(self.average_interval_in_minutes)} minutes"
    def average(self):
        return f"Detected anomalies at indices: {self.anomaly_indices}"
