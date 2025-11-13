"""
EDA - Health Risk Prediction System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
os.makedirs('docs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("="*60)
print("📊 EXPLORATORY DATA ANALYSIS")
print("="*60)

# Load data
print("\n📥 Loading datasets...")
air_quality = pd.read_csv('data/raw/air_quality.csv')
weather = pd.read_csv('data/raw/weather.csv')
wearable = pd.read_csv('data/raw/wearable_health.csv')
outcomes = pd.read_csv('data/raw/health_outcomes.csv')
print("✅ Data loaded!\n")

# Basic info
print("="*60)
print("📋 DATASET SHAPES")
print("="*60)
print(f"Air Quality: {air_quality.shape}")
print(f"Weather: {weather.shape}")
print(f"Wearable: {wearable.shape}")
print(f"Outcomes: {outcomes.shape}\n")

# Check missing values
print("="*60)
print("🔍 MISSING VALUES CHECK")
print("="*60)
print(f"Air Quality: {air_quality.isnull().sum().sum()} missing")
print(f"Weather: {weather.isnull().sum().sum()} missing")
print(f"Wearable: {wearable.isnull().sum().sum()} missing")
print(f"Outcomes: {outcomes.isnull().sum().sum()} missing\n")

# Summary statistics
print("="*60)
print("📊 AIR QUALITY STATISTICS")
print("="*60)
print(air_quality[['pm25', 'pm10', 'aqi']].describe().round(2))

print("\n" + "="*60)
print("⌚ WEARABLE HEALTH STATISTICS")
print("="*60)
print(wearable[['heart_rate_avg', 'sleep_hours', 'steps', 'spo2']].describe().round(2))

# Illness rate
print("\n" + "="*60)
print("🏥 HEALTH OUTCOMES")
print("="*60)
illness_rate = outcomes['respiratory_illness'].mean() * 100
print(f"Overall Illness Rate: {illness_rate:.2f}%")
print(f"\nIllness by City:")
print(outcomes.groupby('city')['respiratory_illness'].mean().mul(100).round(2))

# Visualizations
print("\n" + "="*60)
print("📈 CREATING VISUALIZATIONS")
print("="*60)

# 1. Air Quality by City
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
air_quality.boxplot(column='pm25', by='city', ax=axes[0])
axes[0].set_title('PM2.5 by City')
axes[0].set_ylabel('PM2.5 (μg/m³)')
plt.suptitle('')

air_quality.boxplot(column='aqi', by='city', ax=axes[1])
axes[1].set_title('AQI by City')
axes[1].set_ylabel('Air Quality Index')
plt.tight_layout()
plt.savefig('docs/air_quality_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: docs/air_quality_analysis.png")
plt.close()

# 2. Wearable Health Metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(wearable['heart_rate_avg'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Heart Rate (bpm)')
axes[0, 0].set_title('Heart Rate Distribution')

axes[0, 1].hist(wearable['sleep_hours'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Sleep Hours')
axes[0, 1].set_title('Sleep Duration Distribution')

axes[1, 0].scatter(wearable['steps'].sample(1000), wearable['calories_burned'].sample(1000), alpha=0.3, s=1)
axes[1, 0].set_xlabel('Steps')
axes[1, 0].set_ylabel('Calories')
axes[1, 0].set_title('Steps vs Calories')

axes[1, 1].hist(wearable['spo2'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axvline(95, color='red', linestyle='--', label='Critical (95%)')
axes[1, 1].set_xlabel('SpO2 (%)')
axes[1, 1].set_title('Blood Oxygen Distribution')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('docs/wearable_health_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: docs/wearable_health_analysis.png")
plt.close()

# 3. Illness Rate by City
illness_by_city = outcomes.groupby('city')['respiratory_illness'].mean() * 100
plt.figure(figsize=(10, 5))
illness_by_city.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Respiratory Illness Rate by City')
plt.ylabel('Illness Rate (%)')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.axhline(illness_rate, color='red', linestyle='--', label=f'Average ({illness_rate:.2f}%)')
plt.legend()
plt.tight_layout()
plt.savefig('docs/illness_by_city.png', dpi=150, bbox_inches='tight')
print("✅ Saved: docs/illness_by_city.png")
plt.close()

# Merge datasets for correlation
print("\n" + "="*60)
print("🔗 MERGING DATASETS FOR CORRELATION")
print("="*60)

air_quality['timestamp'] = pd.to_datetime(air_quality['timestamp'])
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
wearable['date'] = pd.to_datetime(wearable['date'])
outcomes['date'] = pd.to_datetime(outcomes['date'])

air_daily = air_quality.copy()
air_daily['date'] = air_daily['timestamp'].dt.date
air_daily = air_daily.groupby(['city', 'date']).agg({
    'pm25': 'mean',
    'pm10': 'mean',
    'aqi': 'mean'
}).reset_index()

weather_daily = weather.copy()
weather_daily['date'] = weather_daily['timestamp'].dt.date
weather_daily = weather_daily.groupby(['city', 'date']).agg({
    'temperature': 'mean',
    'humidity': 'mean',
    'precipitation': 'sum'
}).reset_index()

wearable_agg = wearable.groupby(['city', 'date']).agg({
    'heart_rate_avg': 'mean',
    'sleep_hours': 'mean',
    'spo2': 'mean',
    'stress_level': 'mean'
}).reset_index()
wearable_agg['date'] = pd.to_datetime(wearable_agg['date']).dt.date

outcomes_agg = outcomes.groupby(['city', 'date'])['respiratory_illness'].mean().reset_index()
outcomes_agg['date'] = pd.to_datetime(outcomes_agg['date']).dt.date

merged = air_daily.merge(weather_daily, on=['city', 'date'], how='inner')
merged = merged.merge(wearable_agg, on=['city', 'date'], how='inner')
merged = merged.merge(outcomes_agg, on=['city', 'date'], how='inner')

print(f"Merged dataset shape: {merged.shape}")

# Correlation matrix
corr_cols = ['pm25', 'aqi', 'temperature', 'humidity', 'precipitation',
             'heart_rate_avg', 'sleep_hours', 'spo2', 'stress_level', 
             'respiratory_illness']

corr_matrix = merged[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('docs/correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved: docs/correlation_matrix.png")
plt.close()

# Save merged dataset
merged.to_csv('data/processed/merged_daily_data.csv', index=False)
print(f"✅ Saved: data/processed/merged_daily_data.csv\n")

print("="*60)
print("✨ EDA COMPLETE!")
print("="*60)
print("\n📁 Check 'docs/' folder for visualizations")
print("📁 Merged data saved in 'data/processed/'\n")
