"""
Main execution script for House Price Prediction Project
Run this script to execute the complete pipeline
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    print("=" * 80)
    print("🏠 HOUSE PRICE PREDICTION - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Data Processing
        print("📊 Step 1: Data Processing & Feature Engineering")
        print("-" * 50)
        from data_processing import HousePriceDataProcessor
        
        processor = HousePriceDataProcessor()
        df = processor.create_sample_dataset(n_samples=2000)
        print(f"✅ Dataset created: {df.shape}")
        
        X, y, df_processed = processor.preprocess_pipeline(df)
        print(f"✅ Data preprocessing completed: {X.shape}")
        print()
        
        # Step 2: Exploratory Data Analysis
        print("🔍 Step 2: Exploratory Data Analysis")
        print("-" * 50)
        from eda import HousePriceEDA
        
        eda = HousePriceEDA()
        eda.generate_eda_report(df)
        print("✅ EDA completed - plots saved to outputs/plots/")
        print()
        
        # Step 3: Model Training
        print("🤖 Step 3: Model Training & Evaluation")
        print("-" * 50)
        from models import HousePriceModels
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_trainer = HousePriceModels()
        comparison_df = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        print("✅ Model training completed")
        print(f"✅ Best model: {model_trainer.best_model_name}")
        print()
        
        # Step 4: Prediction System Demo
        print("🎯 Step 4: Prediction System Demo")
        print("-" * 50)
        from predict import HousePricePredictor
        
        predictor = HousePricePredictor()
        result = predictor.demo_prediction()
        print("✅ Prediction system tested successfully")
        print()
        
        # Step 5: Summary
        print("📈 Step 5: Pipeline Summary")
        print("-" * 50)
        print(f"📊 Dataset: {df.shape[0]} houses, {df.shape[1]} features")
        print(f"🤖 Models trained: {len(model_trainer.models)} algorithms")
        print(f"🏆 Best model: {model_trainer.best_model_name}")
        print(f"📊 Best RMSE: ${model_trainer.evaluation_results[model_trainer.best_model_name]['test_metrics']['rmse']:,.0f}")
        print(f"📊 Best R²: {model_trainer.evaluation_results[model_trainer.best_model_name]['test_metrics']['r2']:.4f}")
        print()
        
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Next steps:")
        print("1. Start API server: cd api && uvicorn app:app --reload")
        print("2. Start dashboard: streamlit run src/dashboard.py")
        print("3. View API docs: http://localhost:8000/docs")
        print("4. View dashboard: http://localhost:8501")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_training_only():
    """Run only model training"""
    print("🤖 Running Model Training Only...")
    try:
        from data_processing import HousePriceDataProcessor
        from models import HousePriceModels
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        processor = HousePriceDataProcessor()
        df = processor.create_sample_dataset(n_samples=2000)
        X, y, df_processed = processor.preprocess_pipeline(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        model_trainer = HousePriceModels()
        comparison_df = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        print("✅ Model training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_prediction_demo():
    """Run prediction demo only"""
    print("🎯 Running Prediction Demo...")
    try:
        from predict import HousePricePredictor
        
        predictor = HousePricePredictor()
        result = predictor.demo_prediction()
        
        print("✅ Prediction demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction demo failed: {e}")
        return False

def run_eda_only():
    """Run EDA only"""
    print("🔍 Running Exploratory Data Analysis...")
    try:
        from data_processing import HousePriceDataProcessor
        from eda import HousePriceEDA
        
        # Create data
        processor = HousePriceDataProcessor()
        df = processor.create_sample_dataset(n_samples=2000)
        
        # Run EDA
        eda = HousePriceEDA()
        eda.generate_eda_report(df)
        
        print("✅ EDA completed!")
        return True
        
    except Exception as e:
        print(f"❌ EDA failed: {e}")
        return False

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='House Price Prediction Pipeline')
    parser.add_argument('--mode', choices=['complete', 'train', 'predict', 'eda'], 
                       default='complete', help='Pipeline mode to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.mode == 'complete':
        success = run_complete_pipeline()
    elif args.mode == 'train':
        success = run_training_only()
    elif args.mode == 'predict':
        success = run_prediction_demo()
    elif args.mode == 'eda':
        success = run_eda_only()
    else:
        print("Invalid mode specified")
        success = False
    
    if success:
        print("\n🎉 Operation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
