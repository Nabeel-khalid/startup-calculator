import streamlit as st
import pandas as pd
import numpy as np
import plotly as pl
import plotly.express as px
import sklearn as skl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
import requests

# Set page configuration for wide mode and dark theme
st.set_page_config(layout='wide', page_title='Startup Category Size & Success Forecasting', page_icon='📊')


# Set Data Commons API key
API_KEY = 'AIzaSyCTI4Xz-UW_G2Q2RfknhcfdAnTHq5X5XuI'

# Function to fetch data from Data Commons API
def fetch_population(dcid, year, api_key):
    url = f'https://api.datacommons.org/v2/observation/point/linked'
    params = {
        'dcids': dcid,
        'linked_property': 'containedInPlace',
        'linked_entity': 'country/USA',
        'stat_vars': 'Count_Person',
        'date': year,
        'key': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'observations' in data:
            return data['observations'][0]['value']
    return None

# Load custom dataset
st.title('Startup Category Size & Success Forecasting')

# Option to choose dataset source
data_source = st.sidebar.radio("Select Data Source", ('Demo Data', 'Data Commons', 'Excel File'))

if data_source == 'Demo Data':
    # Load demo dataset
    years = list(range(2010, 2024))
    geographies = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
    sectors = ['Fintech', 'Healthcare', 'Edtech', 'SaaS', 'Biotech', 'E-commerce']
    data = []

    for year in years:
        for geography in geographies:
            for sector in sectors:
                size = random.randint(50, 500)
                capture_size = random.randint(10, 100)
                yoy_growth = random.uniform(5, 30)
                market_size = random.randint(500, 5000)
                total_addressable_market = market_size * random.uniform(1.5, 3)
                serviceable_addressable_market = total_addressable_market * random.uniform(0.6, 0.9)
                serviceable_obtainable_market = serviceable_addressable_market * random.uniform(0.4, 0.7)
                data.append({
                    'geography': geography,
                    'sector': sector,
                    'year': year,
                    'size': size,
                    'capture_size': capture_size,
                    'yoy_growth': yoy_growth,
                    'market_size': market_size,
                    'total_addressable_market': total_addressable_market,
                    'serviceable_addressable_market': serviceable_addressable_market,
                    'serviceable_obtainable_market': serviceable_obtainable_market
                })
    data = pd.DataFrame(data)

elif data_source == 'Data Commons':
    # Load dataset with historical data from Data Commons
    years = list(range(2010, 2024))
    geographies = ['geoId/06', 'country/IND', 'country/CHN', 'country/DEU', 'country/BRA']
    sectors = ['Fintech', 'Healthcare', 'Edtech', 'SaaS', 'Biotech', 'E-commerce']
    data = []

    # Fetching sample statistics for each geography and year
    for geography in geographies:
        for year in years:
            for sector in sectors:
                try:
                    population = fetch_population(geography, str(year), API_KEY)
                    if population is None:
                        population = random.randint(50000, 1000000)
                    gdp = random.uniform(1e5, 1e7)  # GDP as a placeholder, since fetching GDP might require a different approach
                except Exception as e:
                    population = random.randint(50000, 1000000)
                    gdp = random.uniform(1e5, 1e7)

                size = random.randint(50, 500)
                capture_size = random.randint(10, 100)
                yoy_growth = random.uniform(5, 30)
                market_size = population * random.uniform(0.01, 0.1)
                total_addressable_market = market_size * random.uniform(1.5, 3)
                serviceable_addressable_market = total_addressable_market * random.uniform(0.6, 0.9)
                serviceable_obtainable_market = serviceable_addressable_market * random.uniform(0.4, 0.7)
                data.append({
                    'geography': geography,
                    'sector': sector,
                    'year': year,
                    'population': population,
                    'gdp': gdp,
                    'size': size,
                    'capture_size': capture_size,
                    'yoy_growth': yoy_growth,
                    'market_size': market_size,
                    'total_addressable_market': total_addressable_market,
                    'serviceable_addressable_market': serviceable_addressable_market,
                    'serviceable_obtainable_market': serviceable_obtainable_market
                })

    data = pd.DataFrame(data)

elif data_source == 'Excel File':
    # Allow user to upload an Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        # Load the 'market size' sheet from the Excel file
        xlsx = pd.ExcelFile(uploaded_file)
        if 'market size' in xlsx.sheet_names:
            market_size_data = pd.read_excel(xlsx, sheet_name='market size')
            # Extract relevant input cells (e.g., Sales, Population, etc.)
            data = market_size_data[['Region', 'Sales', 'Population']]
        else:
            st.error("The uploaded file does not contain a 'market size' sheet.")
    else:
        st.warning("Please upload an Excel file to proceed.")
        data = pd.DataFrame()

# Title and description
st.write("""
This app allows you to model startup category sizes and predict their future success based on historical data.
You can explore different geographies, sectors, and other factors to understand trends and make forecasts.
""")

# Sidebar for user inputs
if not data.empty:
    st.sidebar.header('Filter Options')
    selected_geography = st.sidebar.selectbox('Select Geography', data['geography'].unique()) if 'geography' in data.columns else None
    selected_sector = st.sidebar.selectbox('Select Sector', data['sector'].unique()) if 'sector' in data.columns else None
    selected_years = st.sidebar.slider('Select Year Range', min_value=data['year'].min(), max_value=data['year'].max(), value=(data['year'].min(), data['year'].max())) if 'year' in data.columns else None

    # Filter data based on user input
    filtered_data = data
    if selected_geography:
        filtered_data = filtered_data[filtered_data['geography'] == selected_geography]
    if selected_sector:
        filtered_data = filtered_data[filtered_data['sector'] == selected_sector]
    if selected_years:
        filtered_data = filtered_data[filtered_data['year'].between(selected_years[0], selected_years[1])]

    # Show filtered data with editing enabled
    st.subheader('Filtered Data (Editable)')
    editable_data = st.data_editor(filtered_data, num_rows="dynamic")

    # Graph view of the filtered data
    st.subheader('Graph View of Filtered Data')
    fig_table = px.bar(editable_data, x='year', y=['market_size', 'total_addressable_market', 'serviceable_addressable_market', 'serviceable_obtainable_market'], title='Market Size Overview (TAM, SAM, SOM)')
    st.plotly_chart(fig_table)

    # Export filtered data
    st.sidebar.subheader('Export Data')
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(editable_data)
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv'
    )

    # Visualize historical trends
    st.subheader('Historical YoY Growth')
    fig = px.line(editable_data, x='year', y=['yoy_growth', 'market_size', 'total_addressable_market', 'serviceable_addressable_market', 'serviceable_obtainable_market'], title='Historical YoY Growth & Market Size Cuts (TAM, SAM, SOM)')
    st.plotly_chart(fig)

    # Prepare data for modeling
    X = editable_data[['year', 'size', 'capture_size', 'market_size', 'total_addressable_market', 'serviceable_addressable_market', 'serviceable_obtainable_market']]

    y = editable_data['yoy_growth']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Show predictions and error
    st.subheader('Model Predictions')
    for i, prediction in enumerate(predictions):
        st.write(f'Test Sample {i + 1}: Predicted YoY Growth: {prediction:.2f}%')
    st.write(f'Mean Squared Error: {mse}')

    # Future Prediction
    st.subheader('Future Forecasting')
    forecasting_option = st.selectbox('Select Forecasting Method', ['Point Estimate', 'Variance Range'])
    future_year = st.slider('Select a Year for Prediction', min_value=2024, max_value=2030, step=1)
    future_years = list(range(editable_data['year'].max() + 1, future_year + 1))

    # Forecast for multiple future years with growth trends
    growth_rate = editable_data['yoy_growth'].mean() / 100
    forecast_data = pd.DataFrame({
        'year': future_years,
        'size': [editable_data['size'].mean() * (1 + growth_rate) ** (i + 1) for i in range(len(future_years))],
        'capture_size': [editable_data['capture_size'].mean() * (1 + growth_rate) ** (i + 1) for i in range(len(future_years))],
        'market_size': [editable_data['market_size'].mean() * (1 + growth_rate) ** (i + 1) for i in range(len(future_years))],
        'total_addressable_market': [editable_data['total_addressable_market'].mean() * (1 + growth_rate) ** (i + 1) for i in range(len(future_years))],
        'serviceable_addressable_market': [editable_data['serviceable_addressable_market'].mean() * (1 + growth_rate) ** (i + 1) for i in range(len(future_years))],
        'serviceable_obtainable_market': [editable_data['serviceable_obtainable_market'].mean() * (1 + growth_rate) ** (i + 1) for i in range(len(future_years))]
    })
    forecast_predictions = model.predict(forecast_data)

    # Plotting forecast graph
    st.subheader('Forecast Graph')
    forecast_df = pd.DataFrame({'year': future_years, 'predicted_yoy_growth': forecast_predictions})

    if forecasting_option == 'Point Estimate':
        fig_forecast = px.line(forecast_df, x='year', y='predicted_yoy_growth', title='Future YoY Growth Forecast')
        st.plotly_chart(fig_forecast)

    elif forecasting_option == 'Variance Range':
        variance = np.std(y) * 0.1  # Assume 10% of the standard deviation as variance
        forecast_df['lower_bound'] = forecast_df['predicted_yoy_growth'] - variance
        forecast_df['upper_bound'] = forecast_df['predicted_yoy_growth'] + variance
        fig_forecast = px.line(forecast_df, x='year', y=['predicted_yoy_growth', 'lower_bound', 'upper_bound'], title='Future YoY Growth Forecast with Variance')
        st.plotly_chart(fig_forecast)

# Example Data Commons Integration
st.subheader('Data Commons Integration Example')
st.write('Displaying population data fetched from Data Commons API...')
try:
    population = fetch_population('geoId/06', '2020', API_KEY)
    if population:
        st.write(f'Population of California in 2020: {population}')
    else:
        st.write('Data not available.')
except Exception as e:
    st.write('Error fetching data from Data Commons:', str(e))

