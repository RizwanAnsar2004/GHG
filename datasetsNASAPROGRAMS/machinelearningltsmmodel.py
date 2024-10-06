import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to load data from NetCDF and create DataFrame
def load_data(file_path, var_name):
    dataset = Dataset(file_path, mode='r')
    emissions_data = dataset.variables[var_name][:]
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    latitudes_mesh, longitudes_mesh = np.meshgrid(latitudes, longitudes)
    df = pd.DataFrame({
        'Latitude': latitudes_mesh.flatten(),
        'Longitude': longitudes_mesh.flatten(),
        'Emissions': emissions_data.flatten()
    }).dropna()
    dataset.close()
    return df

# Define file paths for each year's dataset
file_paths = {
    '2012': '/Users/admin/Downloads/Gridded Methane Data 2012/GEPA_Annual.nc',
    '2013': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2013.nc',
    '2014': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2014.nc',
    '2015': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2015.nc',
    '2016': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2016.nc',
    '2017': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2017.nc',
    '2018': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2018.nc'
}

# Define the variable for emissions data
variables = 'emissions_1A_Combustion_Mobile'  # Change to other variables as needed

# Combine all years into one DataFrame
dfs = []
for year, file_path in file_paths.items():
    df = load_data(file_path, variables)
    df['Year'] = int(year)
    dfs.append(df)

combined_df = pd.concat(dfs)

# Feature scaling (scaling emissions data to a range between 0 and 1)
scaler = MinMaxScaler()
combined_df[['Emissions']] = scaler.fit_transform(combined_df[['Emissions']])

# Create sequences for LSTM
def create_sequences(data, time_steps=3):
    sequences = []
    for i in range(len(data) - time_steps):
        sequence = data[i:i + time_steps]
        target = data[i + time_steps, 2]  # Emissions value as target
        sequences.append((sequence[:, :2], target))  # Use lat, lon, year as input
    return sequences

# Prepare data for LSTM
combined_np = combined_df[['Latitude', 'Longitude', 'Emissions']].values
sequences = create_sequences(combined_np, time_steps=3)

X = np.array([seq[0] for seq in sequences])  # Inputs (lat, lon, emissions over time)
y = np.array([seq[1] for seq in sequences])  # Targets (emissions)

# Reshape for LSTM (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))  # Predicting a single emissions value
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Predict future emissions (e.g., for 2019)
def predict_future_emission(lat, lon, year):
    new_data = np.array([[lat, lon, year]])  # Provide new lat, lon, and year
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = new_data_scaled.reshape((1, new_data_scaled.shape[0], new_data_scaled.shape[1]))
    predicted_emission = model.predict(new_data_scaled)
    predicted_emission = scaler.inverse_transform(predicted_emission)  # Inverse scaling to get actual value
    return predicted_emission[0][0]

# Example: Predict methane emissions for a specific latitude and longitude in 2019
lat = 40.7128   # Example latitude
lon = -74.0060  # Example longitude
predicted_emission_2019 = predict_future_emission(lat, lon, 2019)
print(f"Predicted Emission for 2019 at (lat: {lat}, lon: {lon}): {predicted_emission_2019}")
