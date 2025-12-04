"""
Data Validation Tests for CI/CD Pipeline
"""
import pandas as pd
import os
import sys

def test_data_exists():
    """Test that all required data files exist"""
    print("Testing data file existence...")
    required_files = [
        'data/raw/air_quality.csv',
        'data/raw/weather.csv',
        'data/raw/health_outcomes.csv',
        'data/processed/merged_daily_data.csv'
    ]
    
    for file in required_files:
        assert os.path.exists(file), f"Missing file: {file}"
        print(f"  ✓ {file} exists")
    
    print("✅ All data files exist\n")
    return True

def test_data_quality():
    """Test data quality and integrity"""
    print("Testing data quality...")
    
    # Test merged data
    df = pd.read_csv('data/processed/merged_daily_data.csv')
    
    # Check shape
    assert len(df) > 1000, f"Insufficient data: {len(df)} rows"
    print(f"  ✓ Data has {len(df):,} rows")
    
    # Check required columns
    required_cols = ['pm25', 'aqi', 'temperature', 'humidity', 
                     'heart_rate_avg', 'sleep_hours', 'spo2', 
                     'respiratory_illness']
    
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  ✓ All {len(required_cols)} required columns present")
    
    # Check for missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    assert missing_pct < 50, f"Too many missing values: {missing_pct:.1f}%"
    print(f"  ✓ Missing values: {missing_pct:.1f}% (acceptable)")
    
    # Check target distribution
    illness_rate = df['respiratory_illness'].mean()
    assert 0.05 < illness_rate < 0.25, f"Unusual illness rate: {illness_rate:.2%}"
    print(f"  ✓ Illness rate: {illness_rate:.2%} (realistic)")
    
    print("✅ Data quality tests passed\n")
    return True

def test_data_ranges():
    """Test that data values are in valid ranges"""
    print("Testing data value ranges...")
    
    df = pd.read_csv('data/processed/merged_daily_data.csv')
    
    # PM2.5 should be 0-500
    assert df['pm25'].min() >= 0 and df['pm25'].max() <= 500, "PM2.5 out of range"
    print(f"  ✓ PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f}")
    
    # Temperature should be -20 to 50
    assert df['temperature'].min() >= -20 and df['temperature'].max() <= 50, "Temperature out of range"
    print(f"  ✓ Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}")
    
    # SpO2 should be 85-100
    assert df['spo2'].min() >= 85 and df['spo2'].max() <= 100, "SpO2 out of range"
    print(f"  ✓ SpO2 range: {df['spo2'].min():.1f} - {df['spo2'].max():.1f}")
    
    # Respiratory illness should be 0 or 1
    assert set(df['respiratory_illness'].unique()).issubset({0, 1}), "Invalid illness labels"
    print(f"  ✓ Target labels: binary (0/1)")
    
    print("✅ Data range tests passed\n")
    return True

if __name__ == "__main__":
    try:
        test_data_exists()
        test_data_quality()
        test_data_ranges()
        print("="*50)
        print("ALL DATA TESTS PASSED ✅")
        print("="*50)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
