# 🌾 Kenyan Crop Yield Predictor

AI-powered maize yield predictions for Kenyan smallholder farmers using machine learning.

## 🇰🇪 About

This web application helps Kenyan farmers predict maize crop yields based on weather conditions, soil properties, and farming practices. Built to address food security challenges and support agricultural decision-making in Kenya.



## ✨ Features

- 🎯 **Yield Prediction**: Get accurate crop yield predictions in bags per hectare
- 📊 **Data Visualization**: Interactive charts showing yield patterns and trends
- 🤖 **ML Model Performance**: View model accuracy and feature importance
- 🌦️ **Weather Analysis**: Understand how rainfall and temperature affect yields
- 📱 **User-Friendly Interface**: Easy-to-use web interface built with Streamlit

## 🛠️ Technology Stack

- **Python 3.12+**
- **Machine Learning**: Scikit-learn (Random Forest, Linear Regression)
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Development**: VS Code, Git

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Fred-Edwin/kenyan-crop-yield-prediction.git
   cd kenyan-crop-yield-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate dataset and train model**
   ```bash
   python src/simple_data_collection.py
   python src/03_model_training.py
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
kenyan-crop-yield-prediction/
├── data/
│   ├── raw/
│   │   └── synthetic_crop_data.csv    # Generated dataset
│   └── processed/                     # Processed data
├── models/
│   ├── best_model.pkl                # Trained ML model
│   ├── scaler.pkl                    # Data scaler
│   └── season_encoder.pkl            # Season encoder
├── src/
│   ├── simple_data_collection.py     # Data generation
│   ├── 02_data_exploration.py        # Data analysis
│   ├── 03_model_training.py          # Model training
│   └── 04_make_predictions.py        # CLI predictions
├── visualizations/                   # Generated charts
├── streamlit_app.py                  # Main web application
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 📊 Model Performance

- **Algorithm**: Random Forest Regression
- **Accuracy**: ~89% R² score
- **Error**: ±2.8 bags per hectare average
- **Features**: Weather, soil, farming practices, location

## 🌍 Geographic Coverage

The model covers major maize-producing counties in Kenya:
- Nakuru
- Uasin Gishu  
- Trans Nzoia
- Kitale
- Eldoret

## 🎯 Use Cases

- **Farmers**: Plan planting seasons and resource allocation
- **Agricultural Extension**: Provide data-driven advice
- **Researchers**: Study yield patterns and climate impacts
- **Policy Makers**: Support food security planning

## 🔮 Future Enhancements

- Real-time weather API integration
- Mobile-responsive design
- Additional crops (wheat, beans, etc.)
- Satellite imagery analysis
- Economic profitability predictions

## 📈 Demo

🌐 **Live Demo**: [Deployed App URL - Coming Soon]

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kenya Meteorological Department for weather data inspiration
- Kenyan farmers for the agricultural insights
- Open source community for the amazing tools

## 📞 Contact 

For questions or collaboration opportunities:
- 📧 Email: edwinfredofficial@gmail.com
- 🐙 GitHub: github.com/Fred-Edwin
- 🌐 LinkedIn: linkedin.com/in/edwinfred
---

🇰🇪 **Built with ❤️ for Kenya's agricultural future**