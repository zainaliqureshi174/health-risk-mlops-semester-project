"""
Health Risk Prediction Dashboard
Two views: Health Authorities & Citizens
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Health Risk Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/federated_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/federated_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_models()

# Load data
@st.cache_data
def load_data():
    outcomes = pd.read_csv('data/raw/health_outcomes.csv')
    air = pd.read_csv('data/raw/air_quality.csv')
    return outcomes, air

outcomes, air = load_data()

# Sidebar - View Selection
st.sidebar.title("🏥 Health Risk System")
view = st.sidebar.radio(
    "Select View:",
    ["🏛️ Health Authority Dashboard", "👤 Citizen Personal Alert"]
)

# ==========================================
# VIEW 1: HEALTH AUTHORITY DASHBOARD
# ==========================================

if view == "🏛️ Health Authority Dashboard":
    st.title("🏛️ Health Authority Dashboard")
    st.markdown("**Population-level health risk monitoring and alerts**")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        illness_rate = outcomes['respiratory_illness'].mean() * 100
        st.metric("Overall Illness Rate", f"{illness_rate:.2f}%", 
                  delta="-0.5% vs last week", delta_color="inverse")
    
    with col2:
        total_cases = outcomes['respiratory_illness'].sum()
        st.metric("Total Cases", f"{int(total_cases):,}", 
                  delta="+12 today")
    
    with col3:
        air['timestamp'] = pd.to_datetime(air['timestamp'])
        avg_aqi = air['aqi'].mean()
        st.metric("Average AQI", f"{avg_aqi:.0f}", 
                  delta="+5 vs yesterday", delta_color="inverse")
    
    with col4:
        high_risk_cities = (outcomes.groupby('city')['respiratory_illness'].mean() > 0.13).sum()
        st.metric("High Risk Cities", high_risk_cities, 
                  delta="+1 this week", delta_color="inverse")
    
    st.markdown("---")
    
    # Row 1: Risk Map and Trend
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Risk Heat Map by City")
        
        # Calculate risk by city
        city_risk = outcomes.groupby('city').agg({
            'respiratory_illness': 'mean'
        }).reset_index()
        city_risk['risk_percentage'] = city_risk['respiratory_illness'] * 100
        city_risk['risk_level'] = pd.cut(city_risk['risk_percentage'], 
                                          bins=[0, 10, 13, 100], 
                                          labels=['Low', 'Medium', 'High'])
        
        # Create bar chart
        fig = px.bar(city_risk, x='city', y='risk_percentage',
                     color='risk_level',
                     color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                     title='Respiratory Illness Rate by City',
                     labels={'risk_percentage': 'Illness Rate (%)', 'city': 'City'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Risk Distribution")
        
        risk_counts = city_risk['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     color=risk_counts.index,
                     color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                     title='Cities by Risk Level')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Air Quality Trends
    st.subheader("🌫️ Air Quality Trends")
    
    air['date'] = pd.to_datetime(air['timestamp']).dt.date
    air_daily = air.groupby('date')['aqi'].mean().reset_index()
    air_daily = air_daily.tail(30)  # Last 30 days
    
    fig = px.line(air_daily, x='date', y='aqi',
                  title='Average AQI - Last 30 Days',
                  labels={'aqi': 'Air Quality Index', 'date': 'Date'})
    fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Threshold")
    fig.add_hline(y=150, line_dash="dash", line_color="red", 
                  annotation_text="Unhealthy Threshold")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Alerts and Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Active Alerts")
        
        high_risk = city_risk[city_risk['risk_level'] == 'High']
        if len(high_risk) > 0:
            for _, city in high_risk.iterrows():
                st.error(f"⚠️ **{city['city']}**: High illness rate ({city['risk_percentage']:.2f}%)")
        
        medium_risk = city_risk[city_risk['risk_level'] == 'Medium']
        if len(medium_risk) > 0:
            for _, city in medium_risk.iterrows():
                st.warning(f"⚡ **{city['city']}**: Moderate illness rate ({city['risk_percentage']:.2f}%)")
        
        if len(high_risk) == 0 and len(medium_risk) == 0:
            st.success("✅ No active alerts. All cities at low risk.")
    
    with col2:
        st.subheader("📋 City Details")
        
        city_stats = outcomes.groupby('city').agg({
            'respiratory_illness': ['sum', 'mean', 'count']
        }).round(2)
        city_stats.columns = ['Cases', 'Rate', 'Population']
        city_stats['Rate'] = (city_stats['Rate'] * 100).round(2)
        city_stats = city_stats.sort_values('Rate', ascending=False)
        
        st.dataframe(city_stats, use_container_width=True)

# ==========================================
# VIEW 2: CITIZEN PERSONAL ALERT
# ==========================================

else:
    st.title("👤 Personal Health Risk Assessment")
    st.markdown("**Enter your health and environmental data for personalized risk prediction**")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure federated_model.pkl exists.")
    else:
        st.markdown("---")
        
        # Input Form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌫️ Environmental Data")
            pm25 = st.slider("PM2.5 Level (μg/m³)", 0, 200, 50)
            pm10 = st.slider("PM10 Level (μg/m³)", 0, 300, 80)
            aqi = st.slider("Air Quality Index", 0, 300, 150)
            temperature = st.slider("Temperature (°C)", -10, 45, 25)
            
        with col2:
            st.subheader("🌤️ Weather Data")
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            precipitation = st.slider("Precipitation (mm)", 0, 50, 2)
            st.markdown("")
            st.markdown("")
            
        with col3:
            st.subheader("⌚ Health Metrics")
            heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 75)
            sleep_hours = st.slider("Sleep Hours", 3, 12, 7)
            spo2 = st.slider("Blood Oxygen (%)", 85, 100, 98)
            stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        
        st.markdown("---")
        
        # Predict Button
        if st.button("🔮 Assess My Risk", type="primary", use_container_width=True):
            
            # Prepare input
            features = np.array([[pm25, pm10, aqi, temperature, humidity, 
                                precipitation, heart_rate, sleep_hours, spo2, stress]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
                color = "green"
                icon = "✅"
            elif probability < 0.6:
                risk_level = "Medium"
                color = "orange"
                icon = "⚠️"
            else:
                risk_level = "High"
                color = "red"
                icon = "🚨"
            
            st.markdown("---")
            
            # Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Level", f"{icon} {risk_level}")
            
            with col2:
                st.metric("Risk Probability", f"{probability*100:.1f}%")
            
            with col3:
                st.metric("Prediction", "At Risk" if prediction == 1 else "Healthy")
            
            st.markdown("---")
            
            # Recommendations
            st.subheader("💡 Personalized Recommendations")
            
            if risk_level == "High":
                st.error("🚨 **High Risk Detected**")
                st.markdown("""
                - 🏥 **Consult a healthcare provider immediately**
                - 😷 **Wear a mask when going outdoors**
                - 🏠 **Stay indoors during high pollution hours**
                - 💊 **Keep medications handy if prescribed**
                """)
            elif risk_level == "Medium":
                st.warning("⚠️ **Moderate Risk - Take Precautions**")
                st.markdown("""
                - 👀 **Monitor your symptoms closely**
                - 🚶 **Limit outdoor physical activities**
                - 💧 **Stay well hydrated**
                - 😴 **Ensure adequate sleep (7-8 hours)**
                """)
            else:
                st.success("✅ **Low Risk - Stay Healthy!**")
                st.markdown("""
                - 🏃 **Continue regular physical activity**
                - 🥗 **Maintain a healthy diet**
                - 😴 **Keep consistent sleep schedule**
                - 📊 **Monitor air quality regularly**
                """)
            
            # Risk Factors
            st.markdown("---")
            st.subheader("📊 Your Risk Factors")
            
            risk_factors = []
            if pm25 > 55:
                risk_factors.append(("High PM2.5", pm25, "Air pollution"))
            if aqi > 150:
                risk_factors.append(("Poor Air Quality", aqi, "Environmental"))
            if heart_rate > 90 or heart_rate < 60:
                risk_factors.append(("Abnormal Heart Rate", heart_rate, "Health"))
            if sleep_hours < 6:
                risk_factors.append(("Insufficient Sleep", sleep_hours, "Lifestyle"))
            if spo2 < 95:
                risk_factors.append(("Low Blood Oxygen", spo2, "Health"))
            if stress > 7:
                risk_factors.append(("High Stress", stress, "Mental Health"))
            
            if risk_factors:
                for factor, value, category in risk_factors:
                    st.warning(f"⚡ **{factor}**: {value} ({category})")
            else:
                st.success("✅ All metrics within healthy ranges!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🏥 Health Risk Prediction System | Powered by Federated Learning | MLOps Project 2024
    </div>
    """,
    unsafe_allow_html=True
)
