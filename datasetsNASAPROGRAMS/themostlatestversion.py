from netCDF4 import Dataset
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Paths to your NetCDF files
file_paths = {
    '2012': '/Users/admin/Desktop/datasetsNASAPROGRAMS/gridded GHGI/Gridded_GHGI_Methane_v2_2012.nc',
    '2013': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2013.nc',
    '2014': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2014.nc',
    '2015': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2015.nc',
    '2016': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2016.nc',
    '2017': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2017.nc',
    '2018': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2018.nc'
}

# Updated variables dictionary
variables = {
    'Mobile Combustion': 'emi_ch4_1A_Combustion_Mobile',
    'Stationary Combustion': 'emi_ch4_1A_Combustion_Stationary',
    'Natural Gas Production': 'emi_ch4_1B2b_Natural_Gas_Production',
    'Enteric Fermentation': 'emi_ch4_3A_Enteric_Fermentation',
    'Municipal Landfills': 'emi_ch4_5A1_Landfills_MSW',
}

# Function to load data from a NetCDF file
def load_data(file_path, var_name):
    dataset = Dataset(file_path, mode='r')
    try:
        # Check if variable exists in the dataset
        if var_name in dataset.variables:
            emissions_data = dataset.variables[var_name][:]
            latitudes = dataset.variables['lat'][:]
            longitudes = dataset.variables['lon'][:]
            latitudes_mesh, longitudes_mesh = np.meshgrid(latitudes, longitudes)
            df = pd.DataFrame({
                'Latitude': latitudes_mesh.flatten(),
                'Longitude': longitudes_mesh.flatten(),
                'Emissions': emissions_data.flatten()
            }).dropna()
        else:
            st.warning(f"Variable '{var_name}' not found in the dataset for {file_path}. Skipping this variable.")
            df = pd.DataFrame()  # Return an empty DataFrame if the variable is not found
    except Exception as e:
        st.error(f"An error occurred while loading data from '{file_path}': {e}")
        df = pd.DataFrame()
    dataset.close()
    return df

# Create Streamlit app
st.title('Methane Emissions Visualization')
st.sidebar.header('Settings')

# Sidebar inputs
selected_years = st.sidebar.multiselect('Select Year:', list(file_paths.keys()), default=['2012'])
selected_vars = st.sidebar.multiselect('Select Emission Source:', list(variables.keys()), default=['Mobile Combustion'])

# Load data for selected years and variables
df_dict = {var: [] for var in selected_vars}  # Initialize a dictionary to hold data for each variable

for year in selected_years:
    for var_name in selected_vars:
        file_path = file_paths[year]
        df = load_data(file_path, variables[var_name])
        if not df.empty:
            df['Year'] = year  # Add year information
            df['Emission Source'] = var_name  # Add source information
            df_dict[var_name].append(df)

# Create a heatmap for each selected variable
for var_name in selected_vars:
    if df_dict[var_name]:
        combined_df = pd.concat(df_dict[var_name])
        
        # Display heatmap for the current variable
        fig = px.density_mapbox(
            combined_df,
            lat='Latitude',
            lon='Longitude',
            z='Emissions',
            radius=15,
            mapbox_style="open-street-map",
            title=f'Methane Emissions Visualization: {var_name}',
            center={"lat": np.mean(combined_df['Latitude']), "lon": np.mean(combined_df['Longitude'])},
            zoom=3,  # Set default zoom level
            opacity=0.6,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for the emission source: {var_name}.")

# Notify if no data is available for selected options
if not any(df_dict.values()):
    st.warning("No data available for the selected options.")
