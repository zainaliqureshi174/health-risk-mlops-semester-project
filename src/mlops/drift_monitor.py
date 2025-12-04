"""
Data Drift Detection and Monitoring
"""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import json
from datetime import datetime
from evidently.options import DataDriftOptions

# Add this before creating the report
drift_options = DataDriftOptions(threshold=0.4)

report = Report(
    metrics=[DataDriftPreset(), DataQualityPreset()],
    options=[drift_options]  # Add this line
)

print("="*60)
print("📊 DATA DRIFT MONITORING")
print("="*60)

# Load reference data (training data)
print("\n📥 Loading reference data...")
reference_data = pd.read_csv('data/processed/merged_daily_data.csv')
print(f"Reference data: {len(reference_data):,} samples")

# Simulate current data (last 20% as "new" data for demo)
split_point = int(len(reference_data) * 0.8)
reference = reference_data[:split_point]
current = reference_data[split_point:]

print(f"Reference set: {len(reference):,} samples")
print(f"Current set: {len(current):,} samples")

# Features to monitor
feature_cols = ['pm25', 'pm10', 'aqi', 'temperature', 'humidity', 
                'precipitation', 'heart_rate_avg', 'sleep_hours', 
                'spo2', 'stress_level']

print(f"\n🔍 Monitoring {len(feature_cols)} features...")

# Create drift report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

print("\n⏳ Generating drift report...")
report.run(reference_data=reference[feature_cols + ['respiratory_illness']], 
           current_data=current[feature_cols + ['respiratory_illness']])

# Save HTML report
html_path = 'docs/drift_report.html'
report.save_html(html_path)
print(f"✅ HTML report saved: {html_path}")

# Extract drift results
drift_results = report.as_dict()

# Count drifted features
drift_summary = {
    'timestamp': datetime.now().isoformat(),
    'reference_size': len(reference),
    'current_size': len(current),
    'features_monitored': len(feature_cols),
    'drifted_features': []
}

print("\n" + "="*60)
print("📈 DRIFT DETECTION RESULTS")
print("="*60)

try:
    metrics = drift_results['metrics']
    for metric in metrics:
        if metric['metric'] == 'DatasetDriftMetric':
            dataset_drift = metric['result']['dataset_drift']
            drift_share = metric['result']['drift_share']
            
            print(f"\nDataset Drift Detected: {'YES ⚠️' if dataset_drift else 'NO ✅'}")
            print(f"Drift Share: {drift_share:.2%}")
            
            drift_summary['dataset_drift'] = dataset_drift
            drift_summary['drift_share'] = drift_share
            
            # Feature-level drift
            if 'drift_by_columns' in metric['result']:
                print(f"\n{'Feature':<20} {'Drift':<10} {'Score':<10}")
                print("-"*40)
                
                for col, details in metric['result']['drift_by_columns'].items():
                    is_drifted = details.get('drift_detected', False)
                    drift_score = details.get('drift_score', 0)
                    
                    status = "⚠️  YES" if is_drifted else "✅ NO"
                    print(f"{col:<20} {status:<10} {drift_score:.3f}")
                    
                    if is_drifted:
                        drift_summary['drifted_features'].append({
                            'feature': col,
                            'drift_score': drift_score
                        })
except Exception as e:
    print(f"Warning: Could not parse all drift metrics: {e}")

# Save drift summary
summary_path = 'models/drift_summary.json'
with open(summary_path, 'w') as f:
    json.dump(drift_summary, f, indent=2)

print(f"\n✅ Drift summary saved: {summary_path}")

# Alert if drift detected
if drift_summary.get('dataset_drift', False):
    print("\n⚠️  ALERT: Data drift detected!")
    print("   Recommendation: Retrain model with recent data")
else:
    print("\n✅ No significant drift detected")
    print("   Model performance should remain stable")

print("\n" + "="*60)
print("✨ DRIFT MONITORING COMPLETE")
print("="*60)
print(f"\n📄 View detailed report: {html_path}")
print(f"📊 View summary: {summary_path}")
