# 🏠 House Price Prediction using Regression Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![Dashboard](https://img.shields.io/badge/Dashboard-Next.js-blue.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning system that predicts house prices using multiple regression models, complete with a FastAPI inference service and Next.js dashboard.

## 🎯 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for house price prediction, including:

- **Data Processing**: Automated data cleaning, feature engineering, and preprocessing
- **Model Training**: Multiple regression algorithms (Linear, Ridge, Random Forest, XGBoost)
- **Model Evaluation**: Comprehensive performance metrics and comparison
- **Prediction System**: RESTful API for real-time predictions
- **Interactive Dashboard**: Web interface for data exploration and predictions
- **Deployment Ready**: Docker containerization and production-ready code

## 🚀 Key Features

### 🤖 Machine Learning
- **4 Regression Models**: Linear Regression, Ridge Regression, Random Forest, XGBoost
- **Automated Feature Engineering**: Age calculations, area ratios, quality scores
- **Model Comparison**: RMSE, MAE, R² metrics with cross-validation
- **Feature Importance**: SHAP-like analysis for model interpretability

### 📊 Data Analysis
- **Exploratory Data Analysis**: Comprehensive statistical analysis and visualizations
- **Correlation Analysis**: Feature relationships and importance ranking
- **Outlier Detection**: IQR-based outlier identification and handling
- **Missing Data Handling**: Intelligent imputation strategies

### 🌐 API & Dashboard
- **FastAPI Backend**: RESTful API with automatic documentation
- **Next.js Frontend**: Modern, responsive web dashboard
- **Real-time Predictions**: Instant house price estimates
- **Market Insights**: Neighborhood analysis and investment recommendations

### 🐳 Deployment
- **Docker Support**: Containerized deployment
- **Environment Configuration**: Flexible setup for different environments
- **Health Checks**: API monitoring and status endpoints

## 📁 Project Structure

```
House-Price-Prediction/
│
├── 📁 api/                     # FastAPI application
│   ├── app.py                  # Main API server
│   ├── requirements.txt        # API dependencies
│   └── Dockerfile             # Docker configuration
│
├── 📁 dashboard/               # Next.js frontend
│   ├── pages/                  # React pages
│   ├── components/             # React components
│   ├── package.json           # Frontend dependencies
│   └── tailwind.config.js     # Styling configuration
│
├── 📁 src/                      # Core ML modules
│   ├── data_processing.py      # Data cleaning & preprocessing
│   ├── eda.py                  # Exploratory data analysis
│   ├── models.py               # Model training & evaluation
│   ├── predict.py              # Prediction system
│   ├── dashboard.py            # Streamlit dashboard
│   └── generate_dataset.py     # Synthetic data generator
│
├── 📁 data/                     # Data storage
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned data
│   └── external/               # External data sources
│
├── 📁 models/                   # Trained models
│   ├── *.joblib               # Serialized models
│   └── best_model_info.joblib # Model metadata
│
├── 📁 outputs/                  # Analysis results
│   ├── plots/                  # Visualizations
│   ├── reports/                # Analysis reports
│   └── predictions/            # Sample predictions
│
├── 📁 tests/                    # Unit tests
├── 📁 images/                   # Screenshots & diagrams
├── 📄 README.md                # Project documentation
├── 📄 requirements.txt          # Python dependencies
└── 📄 .gitignore               # Git ignore rules
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **FastAPI**: Modern web framework for APIs
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **Next.js 13**: React framework for web applications
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Chart library for data visualization
- **Axios**: HTTP client for API requests

### Data & Visualization
- **Matplotlib & Seaborn**: Statistical plotting
- **Plotly**: Interactive visualizations
- **Streamlit**: Alternative dashboard framework

### Deployment
- **Docker**: Containerization platform
- **Joblib**: Model serialization
- **Pydantic**: Data validation and settings management

## 📊 Model Performance

| Model | RMSE | MAE | R² Score | CV RMSE |
|-------|------|-----|----------|---------|
| XGBoost | ~$25,000 | ~$18,000 | ~0.92 | ~$26,000 |
| Random Forest | ~$28,000 | ~$20,000 | ~0.89 | ~$29,000 |
| Ridge Regression | ~$32,000 | ~$23,000 | ~0.85 | ~$33,000 |
| Linear Regression | ~$35,000 | ~$25,000 | ~0.82 | ~$36,000 |

*Results based on synthetic dataset with 2,000 house records*

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ (for dashboard)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train Models
```bash
cd src
python models.py
```

### 4. Start API Server
```bash
cd ../api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start Dashboard (Optional)
```bash
# Streamlit Dashboard
cd ../src
streamlit run dashboard.py

# Next.js Dashboard
cd ../dashboard
npm install
npm run dev
```

### 6. Access Applications
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **Next.js Dashboard**: http://localhost:3000

## 📖 Usage Guide

### API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "house": {
    "LotArea": 5000,
    "GrLivArea": 1800,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "YearBuilt": 2000,
    "OverallQual": 7,
    "Neighborhood": "Suburbs",
    "HouseStyle": "Ranch"
  }
}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
-H "Content-Type: application/json" \
-d '{
  "houses": [
    {"LotArea": 5000, "GrLivArea": 1800, ...},
    {"LotArea": 6000, "GrLivArea": 2000, ...}
  ]
}'
```

### Python SDK Usage
```python
from src.predict import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor()

# Single prediction
house_data = {
    "LotArea": 5000,
    "GrLivArea": 1800,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "YearBuilt": 2000,
    "OverallQual": 7,
    "Neighborhood": "Suburbs",
    "HouseStyle": "Ranch"
}

result = predictor.predict_single_house(house_data)
print(f"Predicted Price: {result['price_formatted']}")
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Database Configuration (if using real database)
DATABASE_URL=postgresql://user:password@localhost/housing_db

# Model Configuration
MODEL_PATH=./models
BEST_MODEL=xgboost
```

### Custom Dataset
To use your own dataset:

1. Place CSV file in `data/raw/`
2. Update column mappings in `src/data_processing.py`
3. Run preprocessing pipeline
4. Retrain models

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### API Tests
```bash
# Health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/models

# Sample data
curl http://localhost:8000/sample
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
cd api
docker build -t house-price-api .
```

### Run Container
```bash
docker run -p 8000:8000 house-price-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
    environment:
      - API_HOST=0.0.0.0
```

## 📈 Model Features

### Input Features
- **Property Size**: Lot area, living area, basement area
- **Room Count**: Bedrooms, bathrooms, total rooms
- **Quality Ratings**: Overall quality, condition, exterior quality
- **Location**: Neighborhood, zoning, house style
- **Age**: Year built, year remodeled
- **Amenities**: Garage, fireplace, central air

### Engineered Features
- **House Age**: Current year - year built
- **Remodel Age**: Current year - year remodeled
- **Total Bathrooms**: Full + half bathrooms
- **Quality Score**: Overall quality × condition
- **Garage Efficiency**: Garage area per car
- **Room Density**: Rooms per square foot

### Target Variable
- **Sale Price**: Continuous house price in USD

## 🎨 Visualizations

The system generates various visualizations:

### Exploratory Analysis
- Price distribution histograms
- Feature correlation heatmaps
- Scatter plots of price vs features
- Box plots for categorical features
- Missing data patterns

### Model Analysis
- Performance comparison charts
- Feature importance plots
- Residual analysis
- Actual vs predicted scatter plots

### Market Insights
- Neighborhood price comparisons
- Market segment distribution
- Investment opportunity analysis
- Price per square foot trends

## 🔍 Model Interpretation

### Feature Importance
The most important features for price prediction typically include:

1. **Overall Quality**: Overall material and finish quality
2. **Living Area**: Above ground living area
3. **Neighborhood**: Location-based price multiplier
4. **Garage Cars**: Garage capacity
5. **Year Built**: Property age

### Model Selection
- **XGBoost**: Best overall performance, handles non-linear relationships
- **Random Forest**: Good balance of accuracy and interpretability
- **Ridge Regression**: Regularized linear model, prevents overfitting
- **Linear Regression**: Baseline model, highly interpretable

## 📚 Learning Outcomes

This project demonstrates expertise in:

### Technical Skills
- **Machine Learning**: Regression algorithms, model evaluation, feature engineering
- **Data Science**: Data cleaning, EDA, statistical analysis
- **Software Engineering**: API development, testing, deployment
- **DevOps**: Docker, CI/CD, monitoring

### Business Skills
- **Problem Solving**: Translating business requirements to technical solutions
- **Communication**: Explaining complex models to non-technical audiences
- **Project Management**: End-to-end project delivery

### Industry Applications
- **Real Estate**: Property valuation, market analysis
- **Finance**: Risk assessment, loan underwriting
- **Insurance**: Premium calculation, claims analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: For excellent ML algorithms and tools
- **FastAPI**: For modern, fast API framework
- **Next.js**: For powerful React framework
- **Kaggle**: For housing datasets and inspiration

## 📞 Contact

- **Project Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: linkedin.com/in/yourprofile
- **GitHub**: github.com/yourusername

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Real-time Data Integration**: Live MLS data feeds
- [ ] **Advanced Models**: Neural networks, ensemble methods
- [ ] **Geospatial Analysis**: Location-based features
- [ ] **Time Series Analysis**: Price trend forecasting
- [ ] **Mobile App**: React Native application
- [ ] **Microservices**: Distributed architecture

### Improvements
- [ ] **Hyperparameter Tuning**: Automated optimization with Optuna
- [ ] **Model Monitoring**: Drift detection and performance tracking
- [ ] **A/B Testing**: Model comparison in production
- [ ] **Explainable AI**: SHAP values for model interpretation
- [ ] **Security**: Authentication, rate limiting, input validation

### Research Opportunities
- [ ] **Transfer Learning**: Apply to different markets
- [ ] **Multi-task Learning**: Predict multiple property attributes
- [ ] **Graph Neural Networks**: Neighborhood relationships
- [ ] **Federated Learning**: Privacy-preserving model training

---

**⭐ If this project helped you, please give it a star!**

**🚀 Happy Coding & Happy Predicting!**
