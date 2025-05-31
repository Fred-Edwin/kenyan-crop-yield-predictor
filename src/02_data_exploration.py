"""
Kenyan Crop Yield Prediction - Data Exploration
Analyze patterns in our crop yield dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("🔍 Kenyan Crop Yield Data Exploration")
print("=" * 40)

# Load the dataset
df = pd.read_csv('data/raw/synthetic_crop_data.csv')
print(f"✓ Loaded dataset with {len(df)} records")

# Basic dataset info
print("\n📊 Dataset Information:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Display basic statistics
print("\n📈 Descriptive Statistics:")
print(df.describe().round(2))

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# 1. Yield Distribution by County
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='county', y='yield_bags_per_ha')
plt.xticks(rotation=45)
plt.title('Crop Yield Distribution by County')
plt.ylabel('Yield (bags per hectare)')

# 2. Yield by Season
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='season', y='yield_bags_per_ha')
plt.title('Crop Yield by Season')
plt.ylabel('Yield (bags per hectare)')

plt.tight_layout()
plt.savefig('visualizations/yield_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Weather Factors vs Yield
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(df['rainfall_mm'], df['yield_bags_per_ha'], alpha=0.6)
plt.xlabel('Rainfall (mm)')
plt.ylabel('Yield (bags/ha)')
plt.title('Rainfall vs Yield')

plt.subplot(1, 3, 2)
plt.scatter(df['temp_max_c'], df['yield_bags_per_ha'], alpha=0.6, color='orange')
plt.xlabel('Max Temperature (°C)')
plt.ylabel('Yield (bags/ha)')
plt.title('Temperature vs Yield')

plt.subplot(1, 3, 3)
plt.scatter(df['soil_ph'], df['yield_bags_per_ha'], alpha=0.6, color='green')
plt.xlabel('Soil pH')
plt.ylabel('Yield (bags/ha)')
plt.title('Soil pH vs Yield')

plt.tight_layout()
plt.savefig('visualizations/weather_vs_yield.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Correlation Matrix
plt.figure(figsize=(10, 8))
# Select numeric columns for correlation
numeric_cols = ['rainfall_mm', 'temp_max_c', 'temp_min_c', 'soil_ph', 
                'fertilizer_used', 'farm_size_ha', 'yield_bags_per_ha']
correlation_matrix = df[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Fertilizer Impact
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fertilizer_yield = df.groupby('fertilizer_used')['yield_bags_per_ha'].mean()
plt.bar(['No Fertilizer', 'Fertilizer Used'], fertilizer_yield.values, 
        color=['lightcoral', 'lightgreen'])
plt.title('Average Yield: Fertilizer vs No Fertilizer')
plt.ylabel('Average Yield (bags/ha)')

# Add value labels on bars
for i, v in enumerate(fertilizer_yield.values):
    plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='fertilizer_used', y='yield_bags_per_ha')
plt.xlabel('Fertilizer Used (0=No, 1=Yes)')
plt.title('Yield Distribution by Fertilizer Use')

plt.tight_layout()
plt.savefig('visualizations/fertilizer_impact.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Yearly Trends
plt.figure(figsize=(12, 6))
yearly_yield = df.groupby(['year', 'season'])['yield_bags_per_ha'].mean().reset_index()

for season in ['Long Rains', 'Short Rains']:
    season_data = yearly_yield[yearly_yield['season'] == season]
    plt.plot(season_data['year'], season_data['yield_bags_per_ha'], 
             marker='o', linewidth=2, label=season)

plt.xlabel('Year')
plt.ylabel('Average Yield (bags/ha)')
plt.title('Yield Trends Over Years by Season')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/yearly_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# Key Insights Summary
print("\n🔑 Key Insights:")
print("-" * 30)

# Best performing county
best_county = df.groupby('county')['yield_bags_per_ha'].mean().idxmax()
best_yield = df.groupby('county')['yield_bags_per_ha'].mean().max()
print(f"• Highest yielding county: {best_county} ({best_yield:.1f} bags/ha)")

# Season comparison
season_yields = df.groupby('season')['yield_bags_per_ha'].mean()
print(f"• Long Rains average yield: {season_yields['Long Rains']:.1f} bags/ha")
print(f"• Short Rains average yield: {season_yields['Short Rains']:.1f} bags/ha")

# Fertilizer impact
fert_impact = df.groupby('fertilizer_used')['yield_bags_per_ha'].mean()
improvement = ((fert_impact[1] - fert_impact[0]) / fert_impact[0]) * 100
print(f"• Fertilizer increases yield by {improvement:.1f}%")

# Correlation insights
strongest_corr = correlation_matrix['yield_bags_per_ha'].abs().sort_values(ascending=False)
print(f"• Strongest yield predictor: {strongest_corr.index[1]} ({strongest_corr.iloc[1]:.2f})")

print(f"\n✓ Visualizations saved in 'visualizations/' folder")
print("📊 Data exploration complete! Ready for model building.")