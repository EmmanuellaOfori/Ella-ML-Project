"""Machine learning prediction and modeling module."""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from config import Config

def _to_native(obj):
    """Recursively convert numpy/pandas scalar types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.bool_, )):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_native(v) for v in obj ]
    return obj

class MLPredictions:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.profitability_model = None
        self.price_model = None
        self._load_models()
    
    def _load_models(self):
        """Load ML models from pickle files"""
        try:
            profitability_path = Path(Config.PROFITABILITY_MODEL_PATH)
            if profitability_path.exists():
                with open(profitability_path, 'rb') as f:
                    self.profitability_model = pickle.load(f)
                print("Profitability model loaded successfully")
            else:
                print(f"Profitability model not found at {profitability_path}")
        except Exception as e:
            print(f"Error loading profitability model: {e}")
        
        try:
            price_path = Path(Config.PRICE_PREDICTION_MODEL_PATH)
            if price_path.exists():
                with open(price_path, 'rb') as f:
                    self.price_model = pickle.load(f)
                print("Price prediction model loaded successfully")
            else:
                print(f"Price prediction model not found at {price_path}")
        except Exception as e:
            print(f"Error loading price prediction model: {e}")
    
    def get_profitability_features(self):
        """Extract features for profitability model from current data"""
        try:
            orders_df = self.data_manager.get_orders()
            products_df = self.data_manager.get_products()
            customers_df = self.data_manager.get_customers()
            
            if orders_df.empty or products_df.empty:
                return None
            
            # Merge data
            merged = orders_df.merge(products_df, on='product_id', how='left')
            merged = merged.merge(customers_df, on='customer_id', how='left')
            
            # Calculate basic features
            if 'quantity' not in merged.columns:
                merged['quantity'] = 0
            if 'unit_price' not in merged.columns:
                for alt in ['unit price', 'price']:
                    if alt in merged.columns:
                        merged['unit_price'] = merged[alt]
                        break
                if 'unit_price' not in merged.columns:
                    merged['unit_price'] = 0
            
            total_revenue = float((merged['quantity'].astype(float) * merged['unit_price'].astype(float)).sum())
            total_orders = int(len(orders_df))
            avg_order_value = float(total_revenue / total_orders) if total_orders > 0 else 0.0
            
            # Customer metrics
            unique_customers = int(merged['customer_id'].nunique())
            avg_orders_per_customer = float(total_orders / unique_customers) if unique_customers > 0 else 0.0
            
            # Product metrics
            unique_products = int(merged['product_id'].nunique())
            avg_price = float(products_df['unit_price'].mean()) if 'unit_price' in products_df.columns else 0.0
            
            return {
                'avg_revenue_per_order': avg_order_value,
                'total_orders': total_orders,
                'unique_customers': unique_customers,
                'avg_orders_per_customer': avg_orders_per_customer,
                'unique_products': unique_products,
                'avg_product_price': avg_price,
                'total_revenue': total_revenue
            }
        except Exception as e:
            print(f"Error extracting profitability features: {e}")
            return None
    
    def predict_profitability(self, target_profit, fixed_costs=5000, growth_factor=1.0):
        """Predict profitability using ML model or fallback calculation"""
        features = self.get_profitability_features()
        
        if self.profitability_model and features:
            try:
                # Prepare features for model (adjust based on your model's expected inputs)
                feature_array = np.array([[
                    features['avg_revenue_per_order'],
                    features['total_orders'],
                    features['unique_customers'],
                    features['avg_orders_per_customer'],
                    features['avg_product_price']
                ]])
                
                # Make prediction
                prediction = self.profitability_model.predict(feature_array)[0]
                
                # Calculate orders needed based on prediction
                predicted_profit_per_order = float(prediction * growth_factor)
                orders_needed = (target_profit + fixed_costs) / predicted_profit_per_order if predicted_profit_per_order > 0 else 0
                
                return {
                    'prediction_source': 'ml_model',
                    'target_profit': float(target_profit),
                    'fixed_costs': float(fixed_costs),
                    'growth_factor': float(growth_factor),
                    'predicted_profit_per_order': predicted_profit_per_order,
                    'orders_needed_monthly': int(np.ceil(orders_needed)),
                    'orders_needed_daily': float(round(orders_needed / 30, 1)),
                    'model_features': _to_native(features),
                    'confidence': 'high'
                }
            except Exception as e:
                print(f"Error with ML profitability prediction: {e}")
        
        # Fallback to rule-based calculation
        if features:
            estimated_cogs = features['total_revenue'] * Config.DEFAULT_COGS_PERCENTAGE
            estimated_profit = features['total_revenue'] - estimated_cogs - Config.DEFAULT_FIXED_COSTS
            avg_profit_per_order = estimated_profit / features['total_orders'] if features['total_orders'] > 0 else 0
            
            adjusted_profit = avg_profit_per_order * growth_factor
            orders_needed = (target_profit + fixed_costs) / adjusted_profit if adjusted_profit > 0 else 0
            
            return {
                'prediction_source': 'rule_based_fallback',
                'target_profit': float(target_profit),
                'fixed_costs': float(fixed_costs),
                'growth_factor': float(growth_factor),
                'predicted_profit_per_order': float(adjusted_profit),
                'orders_needed_monthly': int(np.ceil(orders_needed)),
                'orders_needed_daily': float(round(orders_needed / 30, 1)),
                'estimated_metrics': _to_native(features),
                'confidence': 'medium'
            }
        
        return {
            'error': 'No data available for profitability prediction',
            'prediction_source': 'none'
        }
    
    def predict_price(self, product_features):
        """Predict optimal price for a product using ML model"""
        if not self.price_model:
            return {
                'error': 'Price prediction model not available',
                'fallback_strategy': 'Use market research and competitor analysis'
            }
        
        try:
            # Prepare features for price model (adjust based on your model's expected inputs)
            if isinstance(product_features, dict):
                # Convert dict features to array based on model requirements
                feature_array = np.array([[
                    product_features.get('category_encoded', 0),
                    product_features.get('demand_score', 50),
                    product_features.get('stock_level', 100),
                    product_features.get('competitor_price', 0),
                    product_features.get('cost', 0)
                ]])
            else:
                feature_array = np.array([product_features])
            
            predicted_price = self.price_model.predict(feature_array)[0]
            
            return {
                'predicted_price': float(predicted_price),
                'prediction_source': 'ml_model',
                'confidence': 'high',
                'features_used': product_features
            }
        
        except Exception as e:
            print(f"Error with ML price prediction: {e}")
            return {
                'error': f'Price prediction failed: {str(e)}',
                'fallback_strategy': 'Use cost-plus pricing or market benchmarks'
            }
    
    def get_product_recommendations(self, limit=10):
        """Get ML-based product recommendations for inventory and marketing"""
        try:
            products_df = self.data_manager.get_products()
            orders_df = self.data_manager.get_orders()
            
            if products_df.empty:
                return {'error': 'No products available for recommendations'}
            
            # Calculate basic metrics for each product
            recommendations = []
            
            for _, product in products_df.iterrows():
                product_id = product['product_id']
                product_orders = orders_df[orders_df['product_id'] == product_id] if not orders_df.empty else pd.DataFrame()
                
                # Calculate metrics
                total_sold = int(product_orders['quantity'].sum()) if not product_orders.empty and 'quantity' in product_orders.columns else 0
                sales_frequency = int(len(product_orders))
                avg_order_size = float(product_orders['quantity'].mean()) if not product_orders.empty and 'quantity' in product_orders.columns else 0.0
                
                # Get current stock and price
                current_stock = float(product.get('quantity', 0))
                unit_price = float(product.get('unit_price', 0))
                
                # Calculate recommendation score
                velocity_score = total_sold * 0.4 + sales_frequency * 0.3
                inventory_score = max(0, 100 - current_stock) * 0.2
                price_score = max(0, 100 - unit_price) * 0.1
                
                total_score = velocity_score + inventory_score + price_score
                
                # Determine recommendation type
                if total_sold == 0:
                    rec_type = 'new_product_promotion'
                    priority = 'low'
                elif current_stock < 10 and total_sold > 5:
                    rec_type = 'urgent_restock'
                    priority = 'high'
                elif sales_frequency > 3:
                    rec_type = 'high_demand_monitor'
                    priority = 'medium'
                else:
                    rec_type = 'standard_inventory'
                    priority = 'low'
                
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product.get('product_name', product.get('name', 'Unknown')),
                    'recommendation_type': rec_type,
                    'priority': priority,
                    'score': round(total_score, 2),
                    'metrics': {
                        'total_sold': total_sold,
                        'sales_frequency': sales_frequency,
                        'avg_order_size': round(avg_order_size, 1),
                        'current_stock': current_stock,
                        'unit_price': unit_price
                    },
                    'action': self._get_recommendation_action(rec_type, current_stock, total_sold, unit_price)
                })
            
            # Sort by score and priority
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['score']), reverse=True)
            
            return {
                'recommendations': recommendations[:limit],
                'summary': {
                    'total_products': len(recommendations),
                    'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
                    'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
                    'low_priority': len([r for r in recommendations if r['priority'] == 'low'])
                }
            }
        
        except Exception as e:
            print(f"Error generating product recommendations: {e}")
            return {'error': str(e)}
    
    def _get_recommendation_action(self, rec_type, stock, sold, price):
        """Get specific action recommendation based on type"""
        actions = {
            'urgent_restock': f"🚨 Restock immediately! Only {stock} units left with {sold} sold",
            'high_demand_monitor': f"📈 Monitor closely - High sales activity ({sold} units sold)",
            'new_product_promotion': f"🆕 Launch promotion campaign - New product needs visibility",
            'standard_inventory': f"📦 Standard monitoring - Current stock: {stock} units"
        }
        
        return actions.get(rec_type, "📋 Review product performance and adjust strategy")
