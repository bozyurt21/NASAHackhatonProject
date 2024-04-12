import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
class DATA:
    def __init__(self,s_year,e_year):
        # Load and preprocess the data
        self.data = pd.read_csv('datas/omnidata8.txt', sep="\s+", skiprows=60, header=None)
        self.data= self.data[(self.data[0]>=s_year)&(self.data[0]<=e_year)]
        self.bz_gsm = self.data[15]
        self. placeholder_value = 999.9
        self.bz_gsm_cleaned = self.bz_gsm[self.bz_gsm != self.placeholder_value].values.reshape(-1, 1)

        # Scale the data to [0, 1] range
        self.scaler = MinMaxScaler()
        self.bz_gsm_scaled = self.scaler.fit_transform(self.bz_gsm_cleaned)

        # Define the enhanced autoencoder model
        self.input_dim = self.bz_gsm_scaled.shape[1]
        self.encoding_dim = 8

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

        # Set a threshold for anomalies using the 95th percentile
        self.threshold = np.percentile(self.mse, 95)

        # Anomalies are data points with a reconstruction error above the threshold
        self.anomaly_indices = np.where(self.mse > self.threshold)[0]

        # Extract the year and day-of-year columns
        self.years = self.data[0].values
        self.days_of_year = self.data[1].values

        # Extract the year and day-of-year for the anomaly indices
        self.anomaly_years = self.years[self.anomaly_indices]
        self.anomaly_days = self.days_of_year[self.anomaly_indices]

    # Print the anomalies with their respective dates
    def print_year_day(self):
        for year, day in zip(self.anomaly_years, self.anomaly_days):
            return f"At the {day}th day of the year {year}, magnetic reconnection may have occurred."
    def print_average_anomalies(self):
        # Calculate the intervals between consecutive anomalies in days
        intervals_in_days = np.diff(self.anomaly_indices)
        # Calculate the average interval in days
        average_interval = np.mean(intervals_in_days)
        return f"Average interval between anomalies: {int(average_interval)} days"

