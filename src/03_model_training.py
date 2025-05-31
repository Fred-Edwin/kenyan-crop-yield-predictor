"""
Kenyan Crop Yield Prediction - Model Training
Build and evaluate ML models for crop yield prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

print("🤖 Kenyan Crop Yield Prediction - Model Training")
print("=" * 50)

# Load the dataset
df = pd.read_csv('data/raw/synthetic_crop_data.csv')
print(f"✓ Loaded dataset with {len(df)} records")

# Create models directory
os.makedirs('models', exist_ok=True)

# Data Preprocessing
print("\n🔧 Data Preprocessing...")

# Encode categorical variables
df_processed = df.copy()

# Encode county (target encoding based on mean yield)
county_means = df.groupby('county')['yield_bags_per_ha'].mean().to_dict()
df_processed['county_encoded'] = df_processed['county'].map(county_means)

# Encode season
season_encoder = LabelEncoder()
df_processed['season_encoded'] = season_encoder.fit_transform(df_processed['season'])

# Feature Engineering
print("⚙️ Feature Engineering...")

# Create new features
df_processed['temp_range'] = df_processed['temp_max_c'] - df_processed['temp_min_c']
df_processed['rainfall_per_hectare'] = df_processed['rainfall_mm'] / df_processed['farm_size_ha']

# Create categorical features for rainfall levels
df_processed['rainfall_category'] = pd.cut(df_processed['rainfall_mm'], 
                                         bins=[0, 200, 350, 500, 1000], 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
rainfall_dummies = pd.get_dummies(df_processed['rainfall_category'], prefix='rainfall')
df_processed = pd.concat([df_processed, rainfall_dummies], axis=1)

# Select features for modeling
feature_columns = [
    'county_encoded', 'season_encoded', 'year',
    'rainfall_mm', 'temp_max_c', 'temp_min_c', 'temp_range',
    'soil_ph', 'fertilizer_used', 'farm_size_ha', 'rainfall_per_hectare'
] + list(rainfall_dummies.columns)

X = df_processed[feature_columns]
y = df_processed['yield_bags_per_ha']

print(f"✓ Features selected: {len(feature_columns)}")
print(f"✓ Feature names: {feature_columns[:5]}... (and {len(feature_columns)-5} more)")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_processed['county']
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Model Training
print("\n🏗️ Training Models...")

models = {}
results = {}

# 1. Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr_model

# 2. Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling
models['Random Forest'] = rf_model

# Model Evaluation
print("\n📊 Model Evaluation...")

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 20)
    
    # Choose appropriate features based on model
    if name == 'Linear Regression':
        train_features = X_train_scaled
        test_features = X_test_scaled
    else:
        train_features = X_train
        test_features = X_test
    
    # Predictions
    y_train_pred = model.predict(train_features)
    y_test_pred = model.predict(test_features)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    if name == 'Linear Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                  scoring='neg_mean_squared_error')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Store results
    results[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse': cv_rmse,
        'predictions': y_test_pred
    }
    
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"CV RMSE: {cv_rmse:.2f}")

# Select best model
best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"Test RMSE: {results[best_model_name]['test_rmse']:.2f}")
print(f"Test R²: {results[best_model_name]['test_r2']:.3f}")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(season_encoder, 'models/season_encoder.pkl')

# Feature Importance (for Random Forest)
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Feature Importances:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Model Comparison Visualization
plt.figure(figsize=(12, 8))

# Plot 1: RMSE Comparison
plt.subplot(2, 2, 1)
model_names = list(results.keys())
test_rmse_values = [results[name]['test_rmse'] for name in model_names]
plt.bar(model_names, test_rmse_values, color=['skyblue', 'lightcoral'])
plt.title('Test RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

# Plot 2: R² Comparison
plt.subplot(2, 2, 2)
test_r2_values = [results[name]['test_r2'] for name in model_names]
plt.bar(model_names, test_r2_values, color=['lightgreen', 'gold'])
plt.title('Test R² Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

# Plot 3: Actual vs Predicted (Best Model)
plt.subplot(2, 2, 3)
best_predictions = results[best_model_name]['predictions']
plt.scatter(y_test, best_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title(f'Actual vs Predicted ({best_model_name})')

# Plot 4: Residuals
plt.subplot(2, 2, 4)
residuals = y_test - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Yield')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.savefig('visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Model training complete!")
print(f"✅ Best model saved as 'models/best_model.pkl'")
print(f"✅ Scaler saved as 'models/scaler.pkl'")
print(f"✅ Visualizations saved in 'visualizations/' folder")