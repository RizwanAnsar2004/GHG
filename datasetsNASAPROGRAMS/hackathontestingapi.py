from netCDF4 import Dataset
import numpy as np
import pandas as pd  # Ensure pandas is imported
import plotly.express as px

# Path to your NetCDF file
file_path = '/Users/admin/Downloads/Gridded_GHGI_Methane_v2_2015.nc'

# Open the NetCDF file
dataset = Dataset(file_path, mode='r')

# Print file information
print("NetCDF File Information:")
print("File Name:", dataset.filepath())
print("Dimensions:", dataset.dimensions)
print("Variables:", dataset.variables)
print("Global Attributes:", dataset.ncattrs())

# Print details about each variable
print("\nVariables Details:")
for var_name in dataset.variables:
    var = dataset.variables[var_name]
    print(f"Variable Name: {var_name}")
    print(f"  Dimensions: {var.dimensions}")
    print(f"  Shape: {var.shape}")
    print(f"  Data Type: {var.dtype}")
    print(f"  Attributes: {var.ncattrs()}")

# Extract the data for 'emi_ch4_1A_Combustion_Mobile'
ch4_data = dataset.variables['emi_ch4_1A_Combustion_Mobile'][0, :, :].data  # Selecting the first time step
latitudes = dataset.variables['lat'][:]
longitudes = dataset.variables['lon'][:]

# Create a DataFrame for Plotly
latitudes_mesh, longitudes_mesh = np.meshgrid(latitudes, longitudes)
df = pd.DataFrame({
    'Latitude': latitudes_mesh.flatten(),
    'Longitude': longitudes_mesh.flatten(),
    'CH4_Emission': ch4_data.flatten()
})

# Filter out NaN values
df = df.dropna()

# Close the dataset
dataset.close()

# Create an interactive heatmap
fig = px.density_mapbox(
    df, 
    lat='Latitude', 
    lon='Longitude', 
    z='CH4_Emission', 
    radius=15,  # Adjust the radius for better visibility
    mapbox_style="open-street-map", 
    title='CH4 Emissions from Mobile Combustion (2015)',
    center={"lat": np.mean(latitudes), "lon": np.mean(longitudes)},
    zoom=3,  # Adjust the zoom level to focus on specific areas
    opacity=0.6,  # Set opacity
    color_continuous_scale=px.colors.sequential.Plasma  # Use a different color scale
)

# Show the figure
fig.show()
