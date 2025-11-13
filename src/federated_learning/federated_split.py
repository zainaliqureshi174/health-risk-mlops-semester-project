"""
Federated Data Split - Distribute data across nodes by city
Each city = one federated node (simulates hospital/city keeping data local)
"""

import pandas as pd
import os
import json

print("="*60)
print("🔀 FEDERATED DATA SPLIT")
print("="*60)

# Load merged data
print("\n📥 Loading merged dataset...")
df = pd.read_csv('data/processed/merged_daily_data.csv')
print(f"Total records: {len(df):,}")
print(f"Cities: {df['city'].unique()}")

# Create federated nodes directory
nodes_dir = 'data/federated_nodes'
os.makedirs(nodes_dir, exist_ok=True)

# Split by city
cities = df['city'].unique()
node_info = {}

print("\n" + "="*60)
print("📦 SPLITTING DATA BY CITY")
print("="*60)

for i, city in enumerate(cities, 1):
    node_name = f"node_{i}_{city}"
    node_path = os.path.join(nodes_dir, node_name)
    os.makedirs(node_path, exist_ok=True)
    
    # Get city data
    city_data = df[df['city'] == city].copy()
    
    # Save to node directory
    filepath = os.path.join(node_path, 'local_data.csv')
    city_data.to_csv(filepath, index=False)
    
    # Store node info
    node_info[node_name] = {
        'city': city,
        'records': len(city_data),
        'illness_rate': float(city_data['respiratory_illness'].mean()),
        'filepath': filepath
    }
    
    print(f"\n✅ {node_name}")
    print(f"   City: {city}")
    print(f"   Records: {len(city_data):,}")
    print(f"   Illness rate: {city_data['respiratory_illness'].mean()*100:.2f}%")
    print(f"   Saved to: {filepath}")

# Save node configuration
config_path = os.path.join(nodes_dir, 'nodes_config.json')
with open(config_path, 'w') as f:
    json.dump(node_info, f, indent=2)

print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)
print(f"Total nodes: {len(node_info)}")
print(f"Configuration saved: {config_path}")

# Verify data distribution
print("\n" + "="*60)
print("🔍 DATA DISTRIBUTION VERIFICATION")
print("="*60)

total_records = sum(info['records'] for info in node_info.values())
print(f"Original dataset: {len(df):,} records")
print(f"Sum of all nodes: {total_records:,} records")
print(f"Match: {'✅ YES' if total_records == len(df) else '❌ NO'}")

print("\n📋 Records per node:")
for node, info in node_info.items():
    pct = (info['records'] / total_records) * 100
    print(f"  {node}: {info['records']:,} ({pct:.1f}%)")

print("\n" + "="*60)
print("✨ FEDERATED SPLIT COMPLETE!")
print("="*60)
print(f"\n📁 Data distributed across {len(node_info)} nodes")
print(f"📁 Location: {nodes_dir}/")
print("\n🎯 Next: Train federated model or baseline models")
