import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'data/Cleaned_Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(file_path)

# Convert Model Year to integer
df['Model Year'] = df['Model Year'].astype(int)

# Set Streamlit background style
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #141e30, #243b55);
            color: white;
        }
        .stTitle, .stSubheader {
            color: #ffcc00;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title('Electric Vehicle Market Analysis')

# Data Filtering Options
st.sidebar.header("Filter Data")
selected_make = st.sidebar.selectbox("Select Make", ['All'] + sorted(df['Make'].unique()))
selected_model = st.sidebar.selectbox("Select Model", ['All'] + sorted(df['Model'].unique()))
selected_year = st.sidebar.selectbox("Select Year", ['All'] + sorted(df['Model Year'].unique()))

# Apply Filters
filtered_df = df.copy()
if selected_make != 'All':
    filtered_df = filtered_df[filtered_df['Make'] == selected_make]
if selected_model != 'All':
    filtered_df = filtered_df[filtered_df['Model'] == selected_model]
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Model Year'] == selected_year]

# EV Registrations Over Time
st.subheader('EV Registrations Over Time')
historical_trend = filtered_df.groupby('Model Year').size().reset_index(name='Count')
historical_trend = historical_trend[historical_trend['Model Year'] >= 2000]

fig = px.line(historical_trend, x='Model Year', y='Count', markers=True, title='EV Registrations Over Time', labels={'Count': 'Number of EV Registrations'}, hover_data=['Model Year', 'Count'], color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig)

# Future Prediction Module
st.subheader('Future Prediction of EV Market Growth')
prediction_range = st.slider("Select Prediction Range", 2025, 2045, (2025, 2045))
future_years = np.arange(prediction_range[0], prediction_range[1] + 1).reshape(-1, 1)

X = historical_trend['Model Year'].values.reshape(-1, 1)
y = historical_trend['Count'].values
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

future_years_poly = poly.transform(future_years)
predictions = model.predict(future_years_poly)

fig = px.scatter(historical_trend, x='Model Year', y='Count', title='Predicted EV Market Growth', labels={'Count': 'EV Registrations'}, hover_data=['Model Year', 'Count'], color_discrete_sequence=px.colors.qualitative.Bold)
fig.add_scatter(x=future_years.flatten(), y=predictions, mode='lines', name='Predicted Growth')
st.plotly_chart(fig)

# Additional Visualizations with Improved Colors and Hover Functionality
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 EV Models Sold")
    top_models = filtered_df['Model'].value_counts().nlargest(10).reset_index()
    top_models.columns = ['Model', 'Count']
    fig = px.bar(top_models, x='Model', y='Count', title='Top 10 EV Models Sold', labels={'Count': 'Number of Registrations'}, hover_data=['Model', 'Count'], color='Count', color_continuous_scale='Plasma')
    st.plotly_chart(fig)

with col2:
    st.subheader("EV Registrations by Make")
    ev_by_make = filtered_df['Make'].value_counts().reset_index()
    ev_by_make.columns = ['Make', 'Count']
    fig = px.bar(ev_by_make, x='Make', y='Count', title='EV Registrations by Make', labels={'Count': 'Number of Registrations'}, hover_data=['Make', 'Count'], color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("EV Adoption Rate Over Time")
    ev_adoption = filtered_df.groupby('Model Year').size().reset_index(name='Count')
    ev_adoption['Cumulative'] = ev_adoption['Count'].cumsum()
    ev_adoption = ev_adoption[ev_adoption['Model Year'] >= 2000]
    fig = px.line(ev_adoption, x='Model Year', y='Cumulative', title='EV Adoption Rate Over Time', labels={'Cumulative': 'Cumulative EV Registrations'}, hover_data=['Model Year', 'Cumulative'], color_discrete_sequence=['blue'])
    st.plotly_chart(fig)

with col4:
    st.subheader("EV Growth Rate Comparison")
    ev_adoption['Growth Rate'] = ev_adoption['Count'].pct_change() * 100
    fig = px.bar(ev_adoption, x='Model Year', y='Growth Rate', title='EV Growth Rate Comparison', labels={'Growth Rate': 'Growth Rate (%)'}, hover_data=['Model Year', 'Growth Rate'], color='Growth Rate', color_continuous_scale='Cividis')
    st.plotly_chart(fig)

# Enable interactive zoom for all plots via Streamlit configuration
st.write("You can zoom into any graph for better visualization!")
