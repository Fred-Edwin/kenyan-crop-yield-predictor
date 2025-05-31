"""
Kenyan Crop Yield Prediction - Simple Data Collection
Generates synthetic dataset for the ML project
"""

import pandas as pd
import numpy as np
import os

# Create data directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("🌾 Kenyan Crop Yield Prediction - Data Generation")
print("=" * 50)

# Sample data for major maize-producing counties in Kenya
counties = ['Nakuru', 'Uasin Gishu', 'Trans Nzoia', 'Kitale', 'Eldoret']
years = list(range(2018, 2024))

# Create synthetic dataset based on realistic Kenyan agricultural data
np.random.seed(42)  # For reproducibility

data = []

for county in counties:
    for year in years:
        for season in ['Long Rains', 'Short Rains']:
            # Base yield varies by county (bags per hectare)
            base_yields = {
                'Nakuru': 25, 'Uasin Gishu': 30, 'Trans Nzoia': 28, 
                'Kitale': 26, 'Eldoret': 29
            }
            
            # Season affects yield
            season_multiplier = 1.2 if season == 'Long Rains' else 0.8
            
            # Generate realistic weather data
            if season == 'Long Rains':  # March-July
                rainfall = np.random.normal(400, 100)  # mm
                temp_max = np.random.normal(24, 2)     # Celsius
                temp_min = np.random.normal(12, 2)
            else:  # Short Rains: October-December
                rainfall = np.random.normal(250, 80)
                temp_max = np.random.normal(26, 2)
                temp_min = np.random.normal(14, 2)
            
            # Soil and farming factors
            soil_ph = np.random.normal(6.2, 0.5)
            fertilizer_used = np.random.choice([0, 1], p=[0.3, 0.7])
            farm_size = np.random.exponential(2)  # hectares
            
            # Calculate yield with realistic relationships
            yield_base = base_yields[county] * season_multiplier
            
            # Weather impact
            rainfall_effect = max(0, min(2, rainfall / 300))  # Optimal around 300mm
            temp_effect = max(0.5, min(1.5, 1 - abs(temp_max - 25) / 10))
            
            # Farming practices impact
            fertilizer_effect = 1.3 if fertilizer_used else 1.0
            
            # Soil impact
            soil_effect = max(0.7, min(1.3, 1 + (soil_ph - 6.5) / 5))
            
            # Random factors (weather variability, pests, etc.)
            random_factor = np.random.normal(1, 0.15)
            
            # Final yield calculation
            final_yield = (yield_base * rainfall_effect * temp_effect * 
                          fertilizer_effect * soil_effect * random_factor)
            
            # Ensure realistic bounds
            final_yield = max(5, min(50, final_yield))
            
            data.append({
                'county': county,
                'year': year,
                'season': season,
                'rainfall_mm': max(0, rainfall),
                'temp_max_c': temp_max,
                'temp_min_c': temp_min,
                'soil_ph': max(4.5, min(8.5, soil_ph)),
                'fertilizer_used': fertilizer_used,
                'farm_size_ha': max(0.1, farm_size),
                'yield_bags_per_ha': round(final_yield, 1)
            })

# Create DataFrame
df = pd.DataFrame(data)

print(f"✓ Generated dataset with {len(df)} records")
print(f"✓ Counties: {list(df['county'].unique())}")
print(f"✓ Years: {sorted(df['year'].unique())}")
print(f"✓ Seasons: {list(df['season'].unique())}")

# Save the synthetic dataset
df.to_csv('data/raw/synthetic_crop_data.csv', index=False)
print(f"✓ Dataset saved to 'data/raw/synthetic_crop_data.csv'")

print("\n📊 Dataset Preview:")
print(df.head(10))

print("\n📈 Basic Statistics:")
print(df.describe().round(2))

print("\n🎉 Data generation complete! Ready for the next step.")