"""
Data Collection Script for Health Risk Prediction System
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class HealthDataCollector:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_air_quality_data(self):
        print("📊 Collecting Air Quality Data...")
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='h')
        
        air_quality_data = pd.DataFrame({
            'timestamp': dates,
            'city': np.random.choice(['City_A', 'City_B', 'City_C', 'City_D', 'City_E'], len(dates)),
            'pm25': np.random.normal(50, 20, len(dates)).clip(0, 500),
            'pm10': np.random.normal(80, 30, len(dates)).clip(0, 600),
            'no2': np.random.normal(40, 15, len(dates)).clip(0, 200),
            'so2': np.random.normal(10, 5, len(dates)).clip(0, 100),
            'co': np.random.normal(1.2, 0.5, len(dates)).clip(0, 10),
            'ozone': np.random.normal(50, 10, len(dates)).clip(0, 200),
            'aqi': np.random.randint(0, 300, len(dates))
        })
        
        filepath = os.path.join(self.data_dir, 'air_quality.csv')
        air_quality_data.to_csv(filepath, index=False)
        print(f"✅ Air quality data saved: {filepath}")
        print(f"   Shape: {air_quality_data.shape}")
        return air_quality_data
    
    def download_weather_data(self):
        print("🌤️  Collecting Weather Data...")
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='h')
        
        weather_data = pd.DataFrame({
            'timestamp': dates,
            'city': np.random.choice(['City_A', 'City_B', 'City_C', 'City_D', 'City_E'], len(dates)),
            'temperature': np.random.normal(25, 8, len(dates)),
            'humidity': np.random.normal(60, 15, len(dates)).clip(0, 100),
            'pressure': np.random.normal(1013, 10, len(dates)),
            'wind_speed': np.random.exponential(5, len(dates)),
            'precipitation': np.random.exponential(2, len(dates)),
            'cloud_cover': np.random.uniform(0, 100, len(dates)),
            'uv_index': np.random.randint(0, 12, len(dates))
        })
        
        filepath = os.path.join(self.data_dir, 'weather.csv')
        weather_data.to_csv(filepath, index=False)
        print(f"✅ Weather data saved: {filepath}")
        print(f"   Shape: {weather_data.shape}")
        return weather_data
    
    def generate_wearable_data(self):
        print("⌚ Generating Wearable Health Data...")
        n_users = 1000
        days = 365
        wearable_data = []
        
        for user_id in range(n_users):
            city = np.random.choice(['City_A', 'City_B', 'City_C', 'City_D', 'City_E'])
            age = np.random.randint(18, 80)
            
            for day in range(days):
                date = datetime(2023, 1, 1) + timedelta(days=day)
                wearable_data.append({
                    'user_id': f'user_{user_id:04d}',
                    'city': city,
                    'date': date.strftime('%Y-%m-%d'),
                    'age': age,
                    'heart_rate_avg': np.random.normal(75, 10),
                    'heart_rate_max': np.random.normal(120, 20),
                    'heart_rate_min': np.random.normal(60, 8),
                    'steps': np.clip(np.random.normal(7000, 3000), 0, 30000),
                    'calories_burned': np.random.normal(2200, 400),
                    'sleep_hours': np.clip(np.random.normal(7, 1.5), 3, 12),
                    'sleep_quality': np.random.uniform(0, 1),
                    'stress_level': np.random.randint(1, 11),
                    'spo2': np.clip(np.random.normal(98, 1.5), 85, 100),
                    'active_minutes': np.clip(np.random.normal(30, 20), 0, 300)
                })
        
        wearable_df = pd.DataFrame(wearable_data)
        filepath = os.path.join(self.data_dir, 'wearable_health.csv')
        wearable_df.to_csv(filepath, index=False)
        print(f"✅ Wearable health data saved: {filepath}")
        print(f"   Shape: {wearable_df.shape}")
        return wearable_df
    
    def generate_health_outcomes(self):
        print("🏥 Generating Health Outcomes Data...")
        wearable_path = os.path.join(self.data_dir, 'wearable_health.csv')
        
        if not os.path.exists(wearable_path):
            print("⚠️  Wearable data not found. Generating it first...")
            self.generate_wearable_data()
        
        wearable_df = pd.read_csv(wearable_path)
        unique_records = wearable_df.groupby(['user_id', 'date']).first().reset_index()
        outcomes = []
        
        for _, row in unique_records.iterrows():
            base_risk = 0.05
            age_risk = (row['age'] - 40) / 200 if row['age'] > 40 else 0
            
            health_risk = 0
            if row['heart_rate_avg'] > 90 or row['heart_rate_avg'] < 60:
                health_risk += 0.02
            if row['sleep_hours'] < 6:
                health_risk += 0.03
            if row['stress_level'] > 7:
                health_risk += 0.02
            if row['spo2'] < 95:
                health_risk += 0.05
            
            total_risk = base_risk + age_risk + health_risk
            illness = 1 if np.random.random() < total_risk else 0
            
            outcomes.append({
                'user_id': row['user_id'],
                'date': row['date'],
                'city': row['city'],
                'respiratory_illness': illness,
                'illness_severity': np.random.choice(['mild', 'moderate', 'severe']) if illness else 'none'
            })
        
        outcomes_df = pd.DataFrame(outcomes)
        filepath = os.path.join(self.data_dir, 'health_outcomes.csv')
        outcomes_df.to_csv(filepath, index=False)
        print(f"✅ Health outcomes data saved: {filepath}")
        print(f"   Shape: {outcomes_df.shape}")
        return outcomes_df
    
    def collect_all_data(self):
        print("="*60)
        print("🚀 Starting Data Collection Process")
        print("="*60)
        
        air_quality = self.download_air_quality_data()
        weather = self.download_weather_data()
        wearable = self.generate_wearable_data()
        outcomes = self.generate_health_outcomes()
        
        print("\n" + "="*60)
        print("✨ Data Collection Complete!")
        print("="*60)
        print(f"\n📁 All data saved in: {self.data_dir}/")
        print("\n📊 Dataset Summary:")
        print(f"   - Air Quality: {air_quality.shape[0]:,} records")
        print(f"   - Weather: {weather.shape[0]:,} records")
        print(f"   - Wearable Health: {wearable.shape[0]:,} records")
        print(f"   - Health Outcomes: {outcomes.shape[0]:,} records")
        
        return {
            'air_quality': air_quality,
            'weather': weather,
            'wearable': wearable,
            'outcomes': outcomes
        }

if __name__ == "__main__":
    collector = HealthDataCollector()
    datasets = collector.collect_all_data()
    
    print("\n🎯 Next Steps:")
    print("   1. Run EDA: python notebooks/01_eda.py")
    print("   2. Check data/raw/ folder for CSV files")
