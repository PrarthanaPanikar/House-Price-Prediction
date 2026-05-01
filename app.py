"""
FastAPI Application for House Price Prediction
RESTful API for house price prediction with model serving
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import HousePriceDataProcessor
from models import HousePriceModels
from predict import HousePricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ML components
predictor = None
processor = None
models = None

# Pydantic models for request/response
class HouseFeatures(BaseModel):
    """House features for price prediction"""
    LotArea: float = Field(..., ge=1000, le=50000, description="Lot area in square feet")
    GrLivArea: float = Field(..., ge=500, le=10000, description="Living area in square feet")
    TotalBsmtSF: float = Field(default=0, ge=0, le=5000, description="Basement area in square feet")
    BedroomAbvGr: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    FullBath: int = Field(..., ge=1, le=10, description="Number of full bathrooms")
    HalfBath: int = Field(default=0, ge=0, le=5, description="Number of half bathrooms")
    TotRmsAbvGrd: int = Field(..., ge=2, le=20, description="Total rooms above ground")
    GarageCars: int = Field(default=0, ge=0, le=10, description="Number of garage cars")
    GarageArea: float = Field(default=0, ge=0, le=2000, description="Garage area in square feet")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Year built")
    YearRemodAdd: int = Field(..., ge=1800, le=2025, description="Year remodeled")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall quality rating")
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition rating")
    Fireplaces: int = Field(default=0, ge=0, le=5, description="Number of fireplaces")
    Neighborhood: str = Field(..., description="Neighborhood name")
    HouseStyle: str = Field(..., description="House style")
    ExterQual: str = Field(..., description="Exterior quality")
    KitchenQual: str = Field(..., description="Kitchen quality")
    CentralAir: str = Field(default="Y", description="Central air (Y/N)")
    PavedDrive: str = Field(default="Y", description="Paved drive (Y/P/N)")
    MSZoning: str = Field(..., description="Zoning classification")

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    house: HouseFeatures
    model_name: Optional[str] = Field(None, description="Model name to use (default: best model)")

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    houses: List[HouseFeatures]
    model_name: Optional[str] = Field(None, description="Model name to use (default: best model)")

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_price: float
    price_formatted: str
    confidence_interval: Dict[str, float]
    model_used: str
    model_performance: Dict[str, float]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]
    total_houses: int
    processing_time: float
    timestamp: str

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_type: str
    performance_metrics: Dict[str, float]
    feature_importance: Optional[List[Dict[str, Any]]] = None
    is_best_model: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    api_version: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize ML components on startup"""
    global predictor, processor, models
    
    try:
        logger.info("Loading ML components...")
        
        # Initialize components
        processor = HousePriceDataProcessor()
        models = HousePriceModels()
        predictor = HousePricePredictor()
        
        logger.info("ML components loaded successfully")
        logger.info(f"Best model: {predictor.models.best_model_name}")
        
    except Exception as e:
        logger.error(f"Error loading ML components: {e}")
        # Continue without models - endpoints will return appropriate errors

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = predictor is not None and predictor.models.best_model_name is not None
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        api_version="1.0.0"
    )

# Model information endpoint
@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about available models"""
    if not predictor or not predictor.models.evaluation_results:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    models_info = []
    for model_name, results in predictor.models.evaluation_results.items():
        model_info = ModelInfo(
            model_name=model_name,
            model_type=get_model_type(model_name),
            performance_metrics={
                "rmse": results['test_metrics']['rmse'],
                "mae": results['test_metrics']['mae'],
                "r2": results['test_metrics']['r2'],
                "cv_rmse": results['cv_rmse_mean']
            },
            is_best_model=model_name == predictor.models.best_model_name
        )
        models_info.append(model_info)
    
    return models_info

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Predict house price for a single house"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert Pydantic model to dict
        house_data = request.house.dict()
        
        # Validate input
        is_valid, message = predictor.validate_input(house_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Validation error: {message}")
        
        # Make prediction
        result = predictor.predict_single_house(house_data, request.model_name)
        
        if not result:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Format response
        response = PredictionResponse(
            predicted_price=result['predicted_price'],
            price_formatted=result['price_formatted'],
            confidence_interval=result['confidence_interval'],
            model_used=result['model_used'],
            model_performance=result['model_performance'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction made: {result['price_formatted']} using {result['model_used']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Predict house prices for multiple houses"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    start_time = datetime.now()
    
    try:
        # Convert Pydantic models to dicts
        houses_data = [house.dict() for house in request.houses]
        
        # Validate all inputs
        for i, house_data in enumerate(houses_data):
            is_valid, message = predictor.validate_input(house_data)
            if not is_valid:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Validation error for house {i+1}: {message}"
                )
        
        # Make batch predictions
        results = predictor.predict_batch(houses_data, request.model_name)
        
        if not results:
            raise HTTPException(status_code=500, detail="Batch prediction failed")
        
        # Format responses
        predictions = []
        for result in results:
            prediction = PredictionResponse(
                predicted_price=result['predicted_price'],
                price_formatted=result['price_formatted'],
                confidence_interval=result['confidence_interval'],
                model_used=result['model_used'],
                model_performance=result['model_performance'],
                timestamp=datetime.now().isoformat()
            )
            predictions.append(prediction)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_houses=len(request.houses),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} houses in {processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Feature importance endpoint
@app.get("/features/importance")
async def get_feature_importance(model_name: Optional[str] = None):
    """Get feature importance for a model"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        importance_df = predictor.get_feature_importance(model_name)
        
        if importance_df is None:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model")
        
        # Convert to list of dicts
        importance_list = importance_df.head(20).to_dict('records')
        
        return {
            "model_name": model_name or predictor.models.best_model_name,
            "feature_importance": importance_list
        }
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

# Sample input endpoint
@app.get("/sample")
async def get_sample_input():
    """Get sample house data for testing"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    sample_data = predictor.create_sample_input()
    return sample_data

# Model explanation endpoint
@app.post("/explain")
async def explain_prediction(request: PredictionRequest):
    """Get explanation for a prediction"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        house_data = request.house.dict()
        
        # Validate input
        is_valid, message = predictor.validate_input(house_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Validation error: {message}")
        
        # Get explanation
        explanation = predictor.explain_prediction(house_data, request.model_name)
        
        if not explanation:
            raise HTTPException(status_code=500, detail="Explanation generation failed")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

# Statistics endpoint
@app.get("/stats")
async def get_statistics():
    """Get API usage statistics"""
    return {
        "api_version": "1.0.0",
        "models_available": len(predictor.models.models) if predictor else 0,
        "best_model": predictor.models.best_model_name if predictor else None,
        "timestamp": datetime.now().isoformat()
    }

def get_model_type(model_name: str) -> str:
    """Get model type from model name"""
    if "Linear" in model_name:
        return "Linear Regression"
    elif "Ridge" in model_name:
        return "Ridge Regression"
    elif "Random" in model_name:
        return "Random Forest"
    elif "XGBoost" in model_name:
        return "XGBoost"
    else:
        return "Unknown"

# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "predict": "/predict"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
