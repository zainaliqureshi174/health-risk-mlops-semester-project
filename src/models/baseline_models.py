"""
Baseline Models - Train centralized models for comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import pickle
import os
import json

print("="*60)
print("🤖 BASELINE MODELS TRAINING")
print("="*60)

# Load individual user-level data for proper binary labels
print("\n📥 Loading data...")
wearable = pd.read_csv('data/raw/wearable_health.csv')
outcomes = pd.read_csv('data/raw/health_outcomes.csv')
air = pd.read_csv('data/raw/air_quality.csv')
weather = pd.read_csv('data/raw/weather.csv')

# Convert dates
wearable['date'] = pd.to_datetime(wearable['date'])
outcomes['date'] = pd.to_datetime(outcomes['date'])
air['timestamp'] = pd.to_datetime(air['timestamp'])
weather['timestamp'] = pd.to_datetime(weather['timestamp'])

# Aggregate environmental data by city and date
air['date'] = air['timestamp'].dt.date
air_daily = air.groupby(['city', 'date']).agg({
    'pm25': 'mean', 'pm10': 'mean', 'aqi': 'mean'
}).reset_index()

weather['date'] = weather['timestamp'].dt.date
weather_daily = weather.groupby(['city', 'date']).agg({
    'temperature': 'mean', 'humidity': 'mean', 'precipitation': 'sum'
}).reset_index()

# Merge all data
wearable['date_only'] = wearable['date'].dt.date
df = wearable.merge(outcomes[['user_id', 'date', 'respiratory_illness']], 
                    on=['user_id', 'date'], how='inner')
df = df.merge(air_daily, left_on=['city', 'date_only'], 
              right_on=['city', 'date'], how='left', suffixes=('', '_air'))
df = df.merge(weather_daily, left_on=['city', 'date_only'], 
              right_on=['city', 'date'], how='left', suffixes=('', '_weather'))

# Drop duplicate date columns
df = df.drop(['date_air', 'date_weather', 'date_only'], axis=1, errors='ignore')

print(f"Dataset shape: {df.shape}")
print(f"Illness rate: {df['respiratory_illness'].mean()*100:.2f}%")

# Prepare features and target
print("\n🔧 Preparing features...")
feature_cols = ['pm25', 'pm10', 'aqi', 'temperature', 'humidity', 
                'precipitation', 'heart_rate_avg', 'sleep_hours', 
                'spo2', 'stress_level']

# Remove rows with missing values
df_clean = df[feature_cols + ['respiratory_illness']].dropna()
print(f"After removing NaN: {len(df_clean):,} samples")

X = df_clean[feature_cols]
y = df_clean['respiratory_illness'].astype(int)

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X):,}")
print(f"Positive class: {y.sum():,} ({y.mean()*100:.2f}%)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved")

# Store results
results = {}

# ==========================================
# Model 1: Logistic Regression
# ==========================================
print("\n" + "="*60)
print("📊 MODEL 1: LOGISTIC REGRESSION")
print("="*60)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n🎯 Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

results['logistic_regression'] = {
    'accuracy': float(accuracy_score(y_test, y_pred_lr)),
    'roc_auc': float(roc_auc_score(y_test, y_proba_lr))
}

# Save model
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✅ Model saved: models/logistic_regression.pkl")

# ==========================================
# Model 2: Random Forest
# ==========================================
print("\n" + "="*60)
print("🌲 MODEL 2: RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n🎯 Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 5 Important Features:")
print(feature_importance.head())

results['random_forest'] = {
    'accuracy': float(accuracy_score(y_test, y_pred_rf)),
    'roc_auc': float(roc_auc_score(y_test, y_proba_rf)),
    'feature_importance': feature_importance.to_dict('records')
}

# Save model
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n✅ Model saved: models/random_forest.pkl")

# ==========================================
# Summary
# ==========================================
print("\n" + "="*60)
print("📊 MODELS COMPARISON")
print("="*60)

results_df = pd.DataFrame(results).T
print(results_df[['accuracy', 'roc_auc']])

# Save results
with open('models/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✨ BASELINE TRAINING COMPLETE!")
print("="*60)
print(f"\n📁 Models saved in: models/")
print(f"📊 Results saved: models/baseline_results.json")
print("\n🎯 Next: Implement federated learning")
