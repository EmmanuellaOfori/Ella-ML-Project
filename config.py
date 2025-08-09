"""Configuration settings for the Ella ML Flask application."""

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ella-ml-super-secret-key-2025'
    
    # Database settings (if you want to add database later)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///ella_ml.db'
    
    # File paths
    PROFITABILITY_MODEL_PATH = 'profitability_model.pkl'
    PRICE_PREDICTION_MODEL_PATH = 'Forest_ML.pkl'
    
    # Primary dataset
    NUELLA_DATASET = 'Nuella_train.csv'
    
    # Legacy CSV file paths (for backward compatibility)
    CUSTOMERS_CSV = 'customers.csv'
    ORDERS_CSV = 'orders.csv'
    PRODUCTS_CSV = 'products.csv'
    
    # Business metrics defaults
    DEFAULT_COGS_PERCENTAGE = 0.4  # 40% Cost of Goods Sold
    DEFAULT_FIXED_COSTS = 5000.0  # Monthly estimate
    
    # Authentication settings
    SESSION_PERMANENT = False
    SESSION_TYPE = 'filesystem'
