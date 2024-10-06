from netCDF4 import Dataset
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Paths to your NetCDF files
file_paths = {
    '2012': '/Users/admin/Downloads/Gridded Methane Data 2012/GEPA_Annual.nc',
    '2013': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2013.nc',
    '2014': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2014.nc',
    '2015': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2015.nc',
    '2016': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2016.nc',
    '2017': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2017.nc',
    '2018': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2018.nc'
}

# Updated variables dictionary
variables = {
    'Mobile Combustion': 'emissions_1A_Combustion_Mobile',
    'Stationary Combustion': 'emissions_1A_Combustion_Stationary',
    'Natural Gas Production': 'emissions_1B2b_Natural_Gas_Production',
    'Enteric Fermentation': 'emissions_4A_Enteric_Fermentation',
    'Municipal Landfills': 'emissions_6A_Landfills_Municipal',
}

# Function to load data from a NetCDF file
def load_data(file_path, var_name):
    dataset = Dataset(file_path, mode='r')
    try:
        emissions_data = dataset.variables[var_name][:]
        latitudes = dataset.variables['lat'][:]
        longitudes = dataset.variables['lon'][:]
        latitudes_mesh, longitudes_mesh = np.meshgrid(latitudes, longitudes)
        df = pd.DataFrame({
            'Latitude': latitudes_mesh.flatten(),
            'Longitude': longitudes_mesh.flatten(),
            'Emissions': emissions_data.flatten()
        }).dropna()
    except KeyError:
        st.error(f"Variable '{var_name}' not found in the dataset.")
        df = pd.DataFrame()  # Return an empty DataFrame if the variable is not found
    dataset.close()
    return df

# Create Streamlit app
st.title('Methane Emissions Visualization')
st.sidebar.header('Settings')

# Sidebar inputs
selected_years = st.sidebar.multiselect('Select Year:', list(file_paths.keys()), default=['2012'])
selected_vars = st.sidebar.multiselect('Select Emission Source:', list(variables.keys()), default=['Mobile Combustion'])

# Load data for selected years and variables
df_list = []
for year in selected_years:
    for var_name in selected_vars:
        file_path = file_paths[year]
        df = load_data(file_path, variables[var_name])
        if not df.empty:
            df['Year'] = year  # Add year information
            df['Emission Source'] = var_name  # Add source information
            df_list.append(df)

if df_list:
    combined_df = pd.concat(df_list)
else:
    combined_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'Emissions'])

# Display heatmap
if not combined_df.empty:
    fig = px.density_mapbox(
        combined_df,
        lat='Latitude',
        lon='Longitude',
        z='Emissions',
        radius=15,
        mapbox_style="open-street-map",
        title='Methane Emissions Visualization',
        center={"lat": np.mean(combined_df['Latitude']), "lon": np.mean(combined_df['Longitude'])},
        zoom=3,  # Set default zoom level
        opacity=0.6,
        color_continuous_scale=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected options.")

# Zoom controls
if st.button('Zoom In'):
    st.session_state.zoom_level = st.session_state.get('zoom_level', 3) + 1

if st.button('Zoom Out'):
    st.session_state.zoom_level = st.session_state.get('zoom_level', 3) - 1

# Display the current zoom level
st.write(f"Current Zoom Level: {st.session_state.get('zoom_level', 3)}")

