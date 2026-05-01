# 🚀 Complete Setup Guide

This guide will walk you through setting up the House Price Prediction project from scratch.

## 📋 Prerequisites

### Required Software
- **Python 3.8+** (Recommended: 3.11)
- **Node.js 16+** (for Next.js dashboard)
- **Git** (for version control)
- **VS Code** (recommended IDE)

### Optional Software
- **Docker** (for containerization)
- **PostgreSQL** (for database storage)

## 🗂️ Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### 2. Python Environment Setup

#### Windows
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, xgboost, fastapi; print('✅ All packages installed successfully!')"
```

### 4. Node.js Environment Setup (Optional - for Next.js Dashboard)
```bash
# Navigate to dashboard directory
cd dashboard

# Install Node.js dependencies
npm install

# Return to root directory
cd ..
```

## 🏃‍♂️ Quick Start

### Option 1: Run Everything (Recommended)

#### Step 1: Train Models
```bash
cd src
python models.py
```
*This will:*
- Generate synthetic housing data
- Train all 4 regression models
- Save models to the `models/` directory
- Generate evaluation plots

#### Step 2: Start API Server
```bash
# Keep the first terminal running, open a new terminal
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Step 3: Start Dashboard (Choose one)

**Streamlit Dashboard (Easy Setup):**
```bash
# Open new terminal
cd src
streamlit run dashboard.py
```
Access at: http://localhost:8501

**Next.js Dashboard (Advanced):**
```bash
# Open new terminal
cd dashboard
npm run dev
```
Access at: http://localhost:3000

### Option 2: Individual Components

#### Data Processing Only
```bash
cd src
python data_processing.py
```

#### Exploratory Analysis Only
```bash
cd src
python eda.py
```

#### Prediction System Only
```bash
cd src
python predict.py
```

## 🔧 Detailed Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Model Configuration
MODEL_PATH=./models
BEST_MODEL=xgboost

# Database (optional)
DATABASE_URL=sqlite:///./housing.db

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Custom Dataset Setup

If you have your own housing dataset:

1. **Place your CSV file** in `data/raw/your_dataset.csv`

2. **Update data processor** in `src/data_processing.py`:
```python
def load_custom_data(self, filepath):
    """Load custom housing dataset"""
    df = pd.read_csv(filepath)
    
    # Map your columns to standard names
    column_mapping = {
        'your_price_column': 'SalePrice',
        'your_area_column': 'GrLivArea',
        # Add more mappings as needed
    }
    
    df = df.rename(columns=column_mapping)
    return df
```

3. **Run preprocessing**:
```bash
cd src
python -c "
from data_processing import HousePriceDataProcessor
processor = HousePriceDataProcessor()
df = processor.load_custom_data('../data/raw/your_dataset.csv')
X, y, df_processed = processor.preprocess_pipeline(df)
print('✅ Custom data processed successfully!')
"
```

## 🐳 Docker Setup

### Build and Run with Docker

#### Option 1: API Only
```bash
cd api
docker build -t house-price-api .
docker run -p 8000:8000 -v $(pwd)/../models:/app/models house-price-api
```

#### Option 2: Full Stack with Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - API_HOST=0.0.0.0
    depends_on:
      - db

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: housing_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  frontend:
    build: ./dashboard
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - api

volumes:
  postgres_data:
```

Run with:
```bash
docker-compose up --build
```

## 🧪 Testing the Setup

### 1. Test Python Components
```bash
# Test data processing
cd src
python -c "
from data_processing import HousePriceDataProcessor
processor = HousePriceDataProcessor()
df = processor.create_sample_dataset(100)
print(f'✅ Data processing test passed: {df.shape}')
"

# Test model training
python -c "
from models import HousePriceModels
models = HousePriceModels()
print('✅ Models module imported successfully')
"

# Test prediction system
python -c "
from predict import HousePricePredictor
predictor = HousePricePredictor()
print('✅ Prediction system initialized')
"
```

### 2. Test API Endpoints
```bash
# Start API server first (in another terminal)
cd api && uvicorn app:app --reload

# Test health endpoint
curl http://localhost:8000/health

# Test models endpoint
curl http://localhost:8000/models

# Test sample data endpoint
curl http://localhost:8000/sample

# Test prediction endpoint
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

### 3. Test Frontend
```bash
# Streamlit Dashboard
# Navigate to http://localhost:8501
# Should see the house price prediction interface

# Next.js Dashboard
# Navigate to http://localhost:3000
# Should see the modern web interface
```

## 🐛 Common Issues & Solutions

### Issue 1: Python Package Installation Errors
**Problem**: `pip install` fails with dependency conflicts

**Solution**:
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Use conda if pip fails (alternative)
conda create -n house-price python=3.11
conda activate house-price
conda install pandas numpy scikit-learn matplotlib seaborn xgboost fastapi uvicorn
```

### Issue 2: Model Loading Errors
**Problem**: `Models not loaded` error in API

**Solution**:
```bash
# Train models first
cd src
python models.py

# Check if models exist
ls ../models/
# Should see .joblib files
```

### Issue 3: Port Already in Use
**Problem**: `Port 8000 is already in use`

**Solution**:
```bash
# Find process using port
netstat -tulpn | grep :8000  # Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn app:app --port 8001
```

### Issue 4: Node.js Version Issues
**Problem**: Next.js dashboard fails to start

**Solution**:
```bash
# Check Node.js version
node --version  # Should be 16+

# Update Node.js with nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

### Issue 5: Memory Errors
**Problem**: Out of memory errors during training

**Solution**:
```bash
# Reduce dataset size in models.py
df = processor.create_sample_dataset(n_samples=1000)  # Instead of 2000

# Or use smaller models in models.py
rf = RandomForestRegressor(n_estimators=50)  # Instead of 100
```

## 📊 Expected Outputs

### After Model Training
- **4 trained models** saved in `models/` directory
- **Performance plots** in `outputs/plots/`
- **Evaluation metrics** displayed in terminal

### API Documentation
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc docs**: http://localhost:8000/redoc
- **OpenAPI spec**: http://localhost:8000/openapi.json

### Dashboard Features
- **Data Overview**: Dataset statistics and distributions
- **Model Performance**: Comparison charts and metrics
- **Price Predictor**: Interactive prediction form
- **Market Insights**: Neighborhood analysis and trends

## 🎯 Verification Checklist

Before proceeding to use the system, verify:

- [ ] ✅ Python environment activated
- [ ] ✅ All packages installed without errors
- [ ] ✅ Models trained successfully
- [ ] ✅ API server starts without errors
- [ ] ✅ Health endpoint returns healthy status
- [ ] ✅ Prediction endpoint works
- [ ] ✅ Dashboard loads and functions
- [ ] ✅ All visualizations generate correctly

## 🚀 Next Steps

Once setup is complete:

1. **Explore the API**: Try different prediction endpoints
2. **Customize Models**: Adjust hyperparameters in `models.py`
3. **Add Your Data**: Replace synthetic data with real housing data
4. **Extend Features**: Add new features to the prediction system
5. **Deploy**: Use Docker for production deployment

## 📞 Support

If you encounter issues:

1. **Check logs**: Look for error messages in terminal output
2. **Review documentation**: Check the main README.md
3. **Verify dependencies**: Ensure all requirements are met
4. **Test components**: Run individual component tests
5. **Community**: Open an issue on GitHub for help

---

**🎉 Setup Complete!** Your House Price Prediction system is ready to use!
