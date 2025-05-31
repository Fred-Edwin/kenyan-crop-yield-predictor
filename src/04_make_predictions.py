"""
Kenyan Crop Yield Prediction - Make Predictions
Use the trained model to predict crop yields for new data
"""

import pandas as pd
import numpy as np
import joblib

print("🌾 Kenyan Crop Yield Predictor")
print("=" * 40)

# Load the trained model and preprocessors
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    season_encoder = joblib.load('models/season_encoder.pkl')
    print("✓ Model and preprocessors loaded successfully")
except:
    print("❌ Error: Please run the model training script first!")
    exit()

# Load original data to get county means for encoding
df_original = pd.read_csv('data/raw/synthetic_crop_data.csv')
county_means = df_original.groupby('county')['yield_bags_per_ha'].mean().to_dict()

def predict_yield(county, season, year, rainfall_mm, temp_max_c, temp_min_c, 
                 soil_ph, fertilizer_used, farm_size_ha):
    """
    Predict crop yield based on input parameters
    """
    
    # Validate inputs
    if county not in county_means:
        print(f"Warning: County '{county}' not in training data. Using average encoding.")
        county_encoded = np.mean(list(county_means.values()))
    else:
        county_encoded = county_means[county]
    
    # Encode season
    try:
        season_encoded = season_encoder.transform([season])[0]
    except:
        print(f"Warning: Season '{season}' not recognized. Using 0.")
        season_encoded = 0
    
    # Feature engineering
    temp_range = temp_max_c - temp_min_c
    rainfall_per_hectare = rainfall_mm / farm_size_ha
    
    # Create rainfall categories
    if rainfall_mm <= 200:
        rainfall_Low, rainfall_Medium, rainfall_High, rainfall_VeryHigh = 1, 0, 0, 0
    elif rainfall_mm <= 350:
        rainfall_Low, rainfall_Medium, rainfall_High, rainfall_VeryHigh = 0, 1, 0, 0
    elif rainfall_mm <= 500:
        rainfall_Low, rainfall_Medium, rainfall_High, rainfall_VeryHigh = 0, 0, 1, 0
    else:
        rainfall_Low, rainfall_Medium, rainfall_High, rainfall_VeryHigh = 0, 0, 0, 1
    
    # Create feature array
    features = np.array([[
        county_encoded, season_encoded, year,
        rainfall_mm, temp_max_c, temp_min_c, temp_range,
        soil_ph, fertilizer_used, farm_size_ha, rainfall_per_hectare,
        rainfall_Low, rainfall_Medium, rainfall_High, rainfall_VeryHigh
    ]])
    
    # Make prediction (Random Forest doesn't need scaling)
    prediction = model.predict(features)[0]
    
    return round(prediction, 1)

# Example predictions
print("\n📊 Example Predictions:")
print("-" * 30)

examples = [
    {
        'county': 'Nakuru',
        'season': 'Long Rains',
        'year': 2024,
        'rainfall_mm': 400,
        'temp_max_c': 25,
        'temp_min_c': 12,
        'soil_ph': 6.5,
        'fertilizer_used': 1,
        'farm_size_ha': 2.0
    },
    {
        'county': 'Uasin Gishu',
        'season': 'Short Rains',
        'year': 2024,
        'rainfall_mm': 250,
        'temp_max_c': 27,
        'temp_min_c': 14,
        'soil_ph': 6.0,
        'fertilizer_used': 0,
        'farm_size_ha': 1.5
    },
    {
        'county': 'Trans Nzoia',
        'season': 'Long Rains',
        'year': 2024,
        'rainfall_mm': 500,
        'temp_max_c': 23,
        'temp_min_c': 11,
        'soil_ph': 7.0,
        'fertilizer_used': 1,
        'farm_size_ha': 3.0
    }
]

for i, example in enumerate(examples, 1):
    predicted_yield = predict_yield(**example)
    
    print(f"\nExample {i}:")
    print(f"Location: {example['county']} ({example['season']} {example['year']})")
    print(f"Weather: {example['rainfall_mm']}mm rainfall, {example['temp_max_c']}°C max temp")
    print(f"Farming: {example['farm_size_ha']}ha farm, {'fertilizer' if example['fertilizer_used'] else 'no fertilizer'}")
    print(f"Soil pH: {example['soil_ph']}")
    print(f"🎯 Predicted Yield: {predicted_yield} bags per hectare")

# Interactive prediction function
def interactive_prediction():
    """
    Interactive function to get user input and make predictions
    """
    print("\n🔮 Make Your Own Prediction!")
    print("-" * 35)
    
    try:
        county = input("Enter county (Nakuru/Uasin Gishu/Trans Nzoia/Kitale/Eldoret): ")
        season = input("Enter season (Long Rains/Short Rains): ")
        year = int(input("Enter year (e.g., 2024): "))
        rainfall_mm = float(input("Enter expected rainfall (mm): "))
        temp_max_c = float(input("Enter max temperature (°C): "))
        temp_min_c = float(input("Enter min temperature (°C): "))
        soil_ph = float(input("Enter soil pH (4.5-8.5): "))
        fertilizer_used = int(input("Will you use fertilizer? (1=Yes, 0=No): "))
        farm_size_ha = float(input("Enter farm size (hectares): "))
        
        predicted_yield = predict_yield(county, season, year, rainfall_mm, 
                                      temp_max_c, temp_min_c, soil_ph, 
                                      fertilizer_used, farm_size_ha)
        
        print(f"\n🎯 Predicted Yield: {predicted_yield} bags per hectare")
        
        # Provide context
        if predicted_yield > 30:
            print("🌟 Excellent yield expected!")
        elif predicted_yield > 25:
            print("✅ Good yield expected!")
        elif predicted_yield > 20:
            print("⚠️ Average yield expected. Consider optimizing conditions.")
        else:
            print("⚠️ Below average yield. Review farming practices and conditions.")
            
    except ValueError:
        print("❌ Invalid input. Please enter numeric values where required.")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

# Run interactive prediction
if __name__ == "__main__":
    interactive_prediction()