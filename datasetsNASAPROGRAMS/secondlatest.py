import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from groq import Groq

# Set your API key for Groq
API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Paths to your NetCDF files for methane emissions
base_path = '/Users/admin/Downloads/gridded GHGI/'
years = [str(year) for year in range(2012, 2019)]
file_paths = {year: os.path.join(base_path, f'Gridded_GHGI_Methane_v2_{year}.nc') for year in years}

# Function to get available variables from the NetCDF dataset
def get_variables(file_path):
    with Dataset(file_path, mode='r') as dataset:
        return list(dataset.variables.keys())

# Dataset information for methane emissions
variables = {
    'Mobile Combustion': 'emi_ch4_1A_Combustion_Mobile',
    'Stationary Combustion': 'emi_ch4_1A_Combustion_Stationary',
    'Natural Gas Production': 'emi_ch4_1B2b_Natural_Gas_Production',
    'Enteric Fermentation': 'emi_ch4_3A_Enteric_Fermentation',
    'Municipal Landfills': 'emi_ch4_5A1_Landfills_MSW',
}

# Load variables dynamically from the first dataset
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

# Load CO2 emissions data
co2_data = pd.read_csv("/Users/admin/Desktop/datasetsNASAPROGRAMS/annual-co2-emissions-per-country.csv")

# Streamlit app
st.title('🌍 Methane Emissions and CO2 Emissions Visualization with Chatbot')

# Sidebar for methane emissions
st.sidebar.header('Methane Emissions Settings')
selected_years = st.sidebar.multiselect('Select Year:', years, default=['2012'])
selected_vars = st.sidebar.multiselect('Select Emission Source:', list(variables.keys()), default=['Mobile Combustion'])

# Load data for selected years and variables for methane emissions
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

# Chatbot input for methane emissions
user_input = st.text_input("Ask about methane emissions or chat with the bot:")
if user_input:
    bot_response = chat_with_bot(user_input)
    st.text_area("Bot Response:", value=bot_response, height=300)

# Sidebar for CO2 emissions
st.sidebar.header("CO2 Emissions Settings")

# Multi-country selection
countries = co2_data['Entity'].unique()
selected_countries = st.sidebar.multiselect('Select Countries', countries, default=countries[:3])

# Year range selection
year_min = int(co2_data['Year'].min())
year_max = int(co2_data['Year'].max())
selected_years_co2 = st.sidebar.slider('Select Year Range', year_min, year_max, (year_min, year_max))

# Filter the data by selected countries and years
filtered_data = co2_data[(co2_data['Entity'].isin(selected_countries)) & 
                          (co2_data['Year'] >= selected_years_co2[0]) & 
                          (co2_data['Year'] <= selected_years_co2[1])]

# Display the title and dataset
st.title("🌍 CO2 Emissions Data Visualization and Analysis")
st.write(f"**CO2 Emissions for selected countries between {selected_years_co2[0]} and {selected_years_co2[1]}**")

# Non-Visualization Feature 1: Download filtered data as CSV
st.sidebar.markdown("### Download Data")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_data)
st.sidebar.download_button(label="Download Filtered Data as CSV", 
                           data=csv, 
                           file_name='filtered_CO2_data.csv',
                           mime='text/csv')

# Non-Visualization Feature 2: Summary Statistics
st.subheader("📊 Summary Statistics")
st.write(f"**Total CO2 Emissions (selected period):** {filtered_data['Annual CO₂ emissions'].sum():,.2f} Metric Tons")
st.write(f"**Average CO2 Emissions (per year):** {filtered_data['Annual CO₂ emissions'].mean():,.2f} Metric Tons")
st.write(f"**Total Countries Selected:** {len(selected_countries)}")

# Non-Visualization Feature 3: Emissions percentage change
st.subheader("📈 Percentage Change in CO2 Emissions")
def calculate_percentage_change(df):
    df = df.sort_values('Year')
    df['Emissions Change (%)'] = df['Annual CO₂ emissions'].pct_change() * 100
    return df

# Display the percentage change for each country
for country in selected_countries:
    st.write(f"### {country}")
    country_data = filtered_data[filtered_data['Entity'] == country]
    country_data = calculate_percentage_change(country_data)
    st.dataframe(country_data[['Year', 'Annual CO₂ emissions', 'Emissions Change (%)']].dropna())

# Visualization Section for CO2 emissions
st.subheader("📊 CO2 Emissions Visualizations")

# Visualization selection
visualization_type = st.selectbox("Select Visualization Type", [
    "Bar Chart", "Stacked Bar Chart", "Pie Chart",
    "Box Plot", "Radar Chart", "Violin Plot"
])

# Bar Chart for CO2 Emissions
if visualization_type == "Bar Chart":
    fig, ax = plt.subplots()
    filtered_data.groupby('Entity')['Annual CO₂ emissions'].sum().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Total CO2 Emissions per Country')
    ax.set_ylabel('CO2 Emissions (Metric Tons)')
    ax.set_xlabel('Countries')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Stacked Bar Chart
elif visualization_type == "Stacked Bar Chart":
    fig, ax = plt.subplots()
    for country in selected_countries:
        country_data = filtered_data[filtered_data['Entity'] == country]
        ax.bar(country_data['Year'], country_data['Annual CO₂ emissions'], label=country, alpha=0.7)
    ax.set_title('CO2 Emissions (Stacked) per Country Over Time')
    ax.set_ylabel('CO2 Emissions (Metric Tons)')
    ax.set_xlabel('Year')
    ax.legend()
    st.pyplot(fig)

# Pie Chart of CO2 Emissions (Country-wise Contribution)
elif visualization_type == "Pie Chart":
    emissions_sum = filtered_data.groupby('Entity')['Annual CO₂ emissions'].sum()
    fig = go.Figure(data=[go.Pie(labels=emissions_sum.index, values=emissions_sum.values, hole=.3)])
    fig.update_layout(title_text='CO2 Emissions Contribution by Country')
    st.plotly_chart(fig)

# Box Plot for CO2 Emissions
elif visualization_type == "Box Plot":
    fig = go.Figure()
    for country in selected_countries:
        country_data = filtered_data[filtered_data['Entity'] == country]
        fig.add_trace(go.Box(y=country_data['Annual CO₂ emissions'], name=country))
    fig.update_layout(title='CO2 Emissions Box Plot by Country', yaxis_title='CO2 Emissions (Metric Tons)')
    st.plotly_chart(fig)

# Radar Chart for CO2 Emissions (if applicable)
elif visualization_type == "Radar Chart":
    fig = go.Figure()
    for country in selected_countries:
        country_data = filtered_data[filtered_data['Entity'] == country]
        fig.add_trace(go.Scatterpolar(
            r=country_data['Annual CO₂ emissions'],
            theta=country_data['Year'],
            fill='toself',
            name=country
        ))
    fig.update_layout(title='Radar Chart for CO2 Emissions', polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig)

# Violin Plot for CO2 Emissions
elif visualization_type == "Violin Plot":
    fig = go.Figure()
    for country in selected_countries:
        country_data = filtered_data[filtered_data['Entity'] == country]
        fig.add_trace(go.Violin(y=country_data['Annual CO₂ emissions'], name=country, box_visible=True))
    fig.update_layout(title='Violin Plot for CO2 Emissions by Country')
    st.plotly_chart(fig)

st.sidebar.markdown("### Contact")
st.sidebar.markdown("For any queries, contact: [your_email@example.com]")
