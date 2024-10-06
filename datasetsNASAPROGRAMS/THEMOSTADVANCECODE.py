from netCDF4 import Dataset
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from groq import Groq
import xarray as xr
import matplotlib.pyplot as plt
import os

# Set your API key
API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Paths to your NetCDF files
base_path = '/Users/admin/Downloads/gridded GHGI/'
years = [str(year) for year in range(2012, 2019)]
file_paths = {year: os.path.join(base_path, f'Gridded_GHGI_Methane_v2_{year}.nc') for year in years}

# Automated variable extraction from the first dataset
def get_variables(file_path):
    with Dataset(file_path, mode='r') as dataset:
        return list(dataset.variables.keys())

# Dataset information
variables = {
    'Mobile Combustion': 'emi_ch4_1A_Combustion_Mobile',
    'Stationary Combustion': 'emi_ch4_1A_Combustion_Stationary',
    'Natural Gas Production': 'emi_ch4_1B2b_Natural_Gas_Production',
    'Enteric Fermentation': 'emi_ch4_3A_Enteric_Fermentation',
    'Municipal Landfills': 'emi_ch4_5A1_Landfills_MSW',
}

# Load variables dynamically
available_variables = get_variables(file_paths[years[0]])  # Check the first year's dataset

# Function to load data from a NetCDF file
def load_data(file_path, var_name):
    dataset = Dataset(file_path, mode='r')
    try:
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
            st.warning(f"Variable '{var_name}' not found in the dataset for {file_path}.")
            df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {e}")
        df = pd.DataFrame()
    dataset.close()
    return df

# Function to interact with Groq API for chatbot-like conversation
def chat_with_bot(user_input):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

# Streamlit app
st.title('Methane Emissions Visualization and Chatbot')
st.sidebar.header('Settings')

# Sidebar inputs for years and emission sources
selected_years = st.sidebar.multiselect('Select Year:', years, default=['2012'])
selected_vars = st.sidebar.multiselect('Select Emission Source:', list(variables.keys()), default=['Mobile Combustion'])

# Load data for selected years and variables
df_dict = {var: [] for var in selected_vars}

for year in selected_years:
    for var_name in selected_vars:
        file_path = file_paths[year]
        df = load_data(file_path, variables[var_name])
        if not df.empty:
            df['Year'] = year
            df['Emission Source'] = var_name
            df_dict[var_name].append(df)

# Create heatmap for each selected variable
for var_name in selected_vars:
    if df_dict[var_name]:
        combined_df = pd.concat(df_dict[var_name])
        fig = px.density_mapbox(
            combined_df,
            lat='Latitude',
            lon='Longitude',
            z='Emissions',
            radius=15,
            mapbox_style="open-street-map",
            title=f'Methane Emissions Visualization: {var_name}',
            center={"lat": np.mean(combined_df['Latitude']), "lon": np.mean(combined_df['Longitude'])},
            zoom=3,
            opacity=0.6,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for the emission source: {var_name}.")

# Notify if no data is available for selected options
if not any(df_dict.values()):
    st.warning("No data available for the selected options.")

# Chatbot input
user_input = st.text_input("Ask about methane emissions or chat with the bot:")

if user_input:
    bot_response = chat_with_bot(user_input)
    st.text_area("Bot Response:", value=bot_response, height=300)
