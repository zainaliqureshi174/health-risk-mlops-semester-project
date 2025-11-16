"""
MLflow Experiment Tracking - Track all model training experiments
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import json

print("="*60)
print("📊 MLFLOW EXPERIMENT TRACKING")
print("="*60)

# Set tracking URI (local by default)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("health-risk-prediction")

print("\n✅ MLflow initialized")
print(f"   Tracking URI: ./mlruns")
print(f"   Experiment: health-risk-prediction")

# Load data
print("\n📥 Loading data...")
wearable = pd.read_csv('data/raw/wearable_health.csv')
outcomes = pd.read_csv('data/raw/health_outcomes.csv')
air = pd.read_csv('data/raw/air_quality.csv')
weather = pd.read_csv('data/raw/weather.csv')

# Prepare data (same as baseline)
wearable['date'] = pd.to_datetime(wearable['date'])
outcomes['date'] = pd.to_datetime(outcomes['date'])
air['timestamp'] = pd.to_datetime(air['timestamp'])
weather['timestamp'] = pd.to_datetime(weather['timestamp'])

air['date'] = air['timestamp'].dt.date
air_daily = air.groupby(['city', 'date']).agg({
    'pm25': 'mean', 'pm10': 'mean', 'aqi': 'mean'
}).reset_index()

weather['date'] = weather['timestamp'].dt.date
weather_daily = weather.groupby(['city', 'date']).agg({
    'temperature': 'mean', 'humidity': 'mean', 'precipitation': 'sum'
}).reset_index()

wearable['date_only'] = wearable['date'].dt.date
df = wearable.merge(outcomes[['user_id', 'date', 'respiratory_illness']], 
                    on=['user_id', 'date'], how='inner')
df = df.merge(air_daily, left_on=['city', 'date_only'], 
              right_on=['city', 'date'], how='left', suffixes=('', '_air'))
df = df.merge(weather_daily, left_on=['city', 'date_only'], 
              right_on=['city', 'date'], how='left', suffixes=('', '_weather'))

df = df.drop(['date_air', 'date_weather', 'date_only'], axis=1, errors='ignore')

feature_cols = ['pm25', 'pm10', 'aqi', 'temperature', 'humidity', 
                'precipitation', 'heart_rate_avg', 'sleep_hours', 
                'spo2', 'stress_level']

df_clean = df[feature_cols + ['respiratory_illness']].dropna()
X = df_clean[feature_cols]
y = df_clean['respiratory_illness'].astype(int)

print(f"✅ Data prepared: {len(X):,} samples")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ==========================================
# Experiment 1: Logistic Regression
# ==========================================
print("\n" + "="*60)
print("🧪 EXPERIMENT 1: Logistic Regression")
print("="*60)

with mlflow.start_run(run_name="logistic_regression"):
    
    # Log parameters
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Logged metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   F1-Score: {f1:.4f}")

# ==========================================
# Experiment 2: Random Forest
# ==========================================
print("\n" + "="*60)
print("🧪 EXPERIMENT 2: Random Forest")
print("="*60)

with mlflow.start_run(run_name="random_forest"):
    
    n_estimators = 100
    
    # Log parameters
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    mlflow.log_dict(feature_importance, "feature_importance.json")
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Logged metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   F1-Score: {f1:.4f}")

# ==========================================
# Experiment 3: Federated Model (from saved)
# ==========================================
print("\n" + "="*60)
print("🧪 EXPERIMENT 3: Federated Learning")
print("="*60)

try:
    import pickle
    with open('models/federated_model.pkl', 'rb') as f:
        fed_model = pickle.load(f)
    with open('models/federated_scaler.pkl', 'rb') as f:
        fed_scaler = pickle.load(f)
    
    with mlflow.start_run(run_name="federated_learning"):
        
        # Log parameters
        mlflow.log_param("model_type", "federated_logistic_regression")
        mlflow.log_param("n_nodes", 5)
        mlflow.log_param("n_rounds", 5)
        mlflow.log_param("n_features", len(feature_cols))
        
        # Predict
        X_test_fed = fed_scaler.transform(X_test)
        y_pred = fed_model.predict(X_test_fed)
        y_proba = fed_model.predict_proba(X_test_fed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model
        mlflow.sklearn.log_model(fed_model, "model")
        
        print(f"✅ Logged federated model metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
except Exception as e:
    print(f"⚠️  Could not log federated model: {e}")

# ==========================================
# Summary
# ==========================================
print("\n" + "="*60)
print("✨ MLFLOW TRACKING COMPLETE!")
print("="*60)
print("\n📊 View experiments:")
print("   Run: mlflow ui")
print("   Open: http://localhost:5000")
print("\n💡 All experiments logged to ./mlruns/")
