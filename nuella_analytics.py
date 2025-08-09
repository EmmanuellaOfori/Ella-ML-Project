"""Simplified analytics engine for Nuella perfume business data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import warnings
warnings.filterwarnings('ignore')

def _to_native(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
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

class NuellaAnalytics:
    def __init__(self):
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the Nuella dataset."""
        try:
            self.data = pd.read_csv(Config.NUELLA_DATASET)
            # Convert Order Date to datetime
            self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
            # Clean column names
            self.data.columns = self.data.columns.str.strip()
            print(f"✅ Loaded {len(self.data)} orders from Nuella dataset")
        except Exception as e:
            print(f"❌ Error loading Nuella dataset: {e}")
            self.data = pd.DataFrame()
    
    def predict_30day_profit(self):
        """
        🎯 CORE FUNCTION: Predict profit to be obtained in the next 30 days
        Based on historical Total Price (sales) trends from Nuella_train.csv
        """
        if self.data.empty:
            return {'error': 'No data available for prediction'}
        
        try:
            # Prepare data for time series analysis
            df = self.data.copy()
            df = df.sort_values('Order Date')
            
            # Calculate daily sales aggregates
            daily_sales = df.groupby(df['Order Date'].dt.date).agg({
                'Total Price': 'sum',
                'Order ID': 'count'
            }).rename(columns={'Order ID': 'order_count'})
            
            # Calculate profit margin (estimate: 60% of sales is profit after COGS)
            profit_margin_rate = 0.60  # Conservative estimate for perfume business
            daily_sales['daily_profit'] = daily_sales['Total Price'] * profit_margin_rate
            
            # Get recent performance metrics (last 30 days)
            recent_data = daily_sales.tail(30)
            
            # Calculate trend indicators
            avg_daily_profit = recent_data['daily_profit'].mean()
            trend_slope = np.polyfit(range(len(recent_data)), recent_data['daily_profit'], 1)[0]
            
            # Revenue velocity (week-over-week growth)
            if len(recent_data) >= 14:
                week1_avg = recent_data['daily_profit'].head(7).mean()
                week2_avg = recent_data['daily_profit'].tail(7).mean()
                velocity = (week2_avg - week1_avg) / week1_avg * 100 if week1_avg > 0 else 0
            else:
                velocity = 0
            
            # Seasonal adjustment (perfume business tends to be higher in certain periods)
            current_month = datetime.now().month
            seasonal_multiplier = {
                1: 0.85, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,  # Winter/Spring
                7: 1.15, 8: 1.20, 9: 1.10, 10: 1.05, 11: 1.25, 12: 1.35  # Summer/Holiday
            }.get(current_month, 1.0)
            
            # 30-day profit prediction
            base_prediction = avg_daily_profit * 30
            trend_adjustment = trend_slope * 30 * 15  # Mid-point of 30 days
            seasonal_adjustment = base_prediction * (seasonal_multiplier - 1)
            
            predicted_30day_profit = base_prediction + trend_adjustment + seasonal_adjustment
            
            # Confidence calculation based on data consistency
            profit_std = recent_data['daily_profit'].std()
            confidence_score = max(0.5, min(0.95, 1 - (profit_std / avg_daily_profit)))
            
            # Business insights
            total_orders_recent = recent_data['order_count'].sum()
            avg_order_value = recent_data['Total Price'].sum() / total_orders_recent if total_orders_recent > 0 else 0
            
            return _to_native({
                'predicted_30day_profit': round(predicted_30day_profit, 2),
                'confidence_score': round(confidence_score * 100, 1),
                'analysis_period': f"{recent_data.index.min()} to {recent_data.index.max()}",
                'key_metrics': {
                    'avg_daily_profit': round(avg_daily_profit, 2),
                    'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'velocity_percent': round(velocity, 1),
                    'seasonal_factor': seasonal_multiplier,
                    'avg_order_value': round(avg_order_value, 2),
                    'recent_order_count': int(total_orders_recent)
                },
                'prediction_components': {
                    'base_prediction': round(base_prediction, 2),
                    'trend_adjustment': round(trend_adjustment, 2),
                    'seasonal_adjustment': round(seasonal_adjustment, 2)
                },
                'data_source': 'Nuella_train.csv',
                'algorithm': 'Time Series Trend Analysis with Seasonal Adjustment'
            })
            
        except Exception as e:
            print(f"Error in 30-day profit prediction: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def forecast_next_month_products(self):
        """
        🎯 CORE FUNCTION: Predict items most likely to be sold in the next month
        Based on product purchase patterns from Nuella_train.csv
        """
        if self.data.empty:
            return {'error': 'No data available for forecasting'}
        
        try:
            df = self.data.copy()
            
            # Product analysis across all three categories
            product_forecast = []
            
            # 1. Perfume Oils Analysis
            perfume_oils = df[df['Perfume Oils Quantity'].notna() & (df['Perfume Oils Quantity'] > 0)]
            if not perfume_oils.empty:
                oil_analysis = perfume_oils.groupby('Perfume Oils Bottle Sizes').agg({
                    'Perfume Oils Quantity': ['sum', 'count', 'mean'],
                    'Unit Price(Perfume Oil)': 'mean',
                    'Order Date': 'max'
                }).round(2)
                
                for bottle_size in oil_analysis.index:
                    if pd.notna(bottle_size):
                        total_sold = oil_analysis.loc[bottle_size, ('Perfume Oils Quantity', 'sum')]
                        frequency = oil_analysis.loc[bottle_size, ('Perfume Oils Quantity', 'count')]
                        avg_qty = oil_analysis.loc[bottle_size, ('Perfume Oils Quantity', 'mean')]
                        avg_price = oil_analysis.loc[bottle_size, ('Unit Price(Perfume Oil)', 'mean')]
                        last_order = oil_analysis.loc[bottle_size, ('Order Date', 'max')]
                        
                        # Calculate demand score
                        recency_score = self._calculate_recency_score(last_order)
                        demand_score = (total_sold * 0.4) + (frequency * 0.3) + (avg_qty * 0.2) + (recency_score * 0.1)
                        
                        product_forecast.append({
                            'product_category': 'Perfume Oil',
                            'product_detail': f"{bottle_size}",
                            'total_quantity_sold': int(total_sold),
                            'order_frequency': int(frequency),
                            'avg_quantity_per_order': round(avg_qty, 1),
                            'avg_unit_price': round(avg_price, 2),
                            'demand_score': round(demand_score, 2),
                            'last_order_date': str(last_order.date()),
                            'forecast_likelihood': self._get_likelihood_category(demand_score)
                        })
            
            # 2. Body Splashes Analysis
            body_splashes = df[df['Body Splashes Quantity'].notna() & (df['Body Splashes Quantity'] > 0)]
            if not body_splashes.empty:
                splash_stats = {
                    'total_quantity_sold': int(body_splashes['Body Splashes Quantity'].sum()),
                    'order_frequency': int(len(body_splashes)),
                    'avg_quantity_per_order': round(body_splashes['Body Splashes Quantity'].mean(), 1),
                    'avg_unit_price': round(body_splashes['Unit Price (Body Splashes)'].mean(), 2),
                    'last_order_date': str(body_splashes['Order Date'].max().date())
                }
                
                demand_score = (splash_stats['total_quantity_sold'] * 0.4) + (splash_stats['order_frequency'] * 0.3) + \
                              (splash_stats['avg_quantity_per_order'] * 0.2) + \
                              (self._calculate_recency_score(body_splashes['Order Date'].max()) * 0.1)
                
                product_forecast.append({
                    'product_category': 'Body Splash',
                    'product_detail': 'All Body Splashes',
                    **splash_stats,
                    'demand_score': round(demand_score, 2),
                    'forecast_likelihood': self._get_likelihood_category(demand_score)
                })
            
            # 3. Boxed Perfume Analysis
            boxed_perfumes = df[df['Boxed Perfume Quantity'].notna() & (df['Boxed Perfume Quantity'] > 0)]
            if not boxed_perfumes.empty:
                boxed_stats = {
                    'total_quantity_sold': int(boxed_perfumes['Boxed Perfume Quantity'].sum()),
                    'order_frequency': int(len(boxed_perfumes)),
                    'avg_quantity_per_order': round(boxed_perfumes['Boxed Perfume Quantity'].mean(), 1),
                    'avg_unit_price': round(boxed_perfumes['Boxed Perfume(Unit Price)'].mean(), 2),
                    'last_order_date': str(boxed_perfumes['Order Date'].max().date())
                }
                
                demand_score = (boxed_stats['total_quantity_sold'] * 0.4) + (boxed_stats['order_frequency'] * 0.3) + \
                              (boxed_stats['avg_quantity_per_order'] * 0.2) + \
                              (self._calculate_recency_score(boxed_perfumes['Order Date'].max()) * 0.1)
                
                product_forecast.append({
                    'product_category': 'Boxed Perfume',
                    'product_detail': 'All Boxed Perfumes',
                    **boxed_stats,
                    'demand_score': round(demand_score, 2),
                    'forecast_likelihood': self._get_likelihood_category(demand_score)
                })
            
            # Sort by demand score
            product_forecast.sort(key=lambda x: x['demand_score'], reverse=True)
            
            # Customer behavior analysis
            customer_patterns = df.groupby('Order Type').agg({
                'Total Price': ['sum', 'mean', 'count'],
                'Total Quantity': 'sum'
            }).round(2)
            
            return _to_native({
                'top_products_forecast': product_forecast,
                'forecast_summary': {
                    'total_products_analyzed': len(product_forecast),
                    'high_likelihood_products': len([p for p in product_forecast if p['forecast_likelihood'] == 'Very High']),
                    'medium_likelihood_products': len([p for p in product_forecast if p['forecast_likelihood'] == 'High']),
                    'analysis_period': f"{df['Order Date'].min().date()} to {df['Order Date'].max().date()}",
                    'total_orders_analyzed': len(df)
                },
                'customer_insights': {
                    'retail_vs_wholesale': dict(df['Order Type'].value_counts()),
                    'avg_order_value_retail': round(df[df['Order Type'] == 'Retail']['Total Price'].mean(), 2),
                    'avg_order_value_wholesale': round(df[df['Order Type'] == 'Wholesale']['Total Price'].mean(), 2)
                },
                'data_source': 'Nuella_train.csv',
                'algorithm': 'Product Frequency Analysis with Recency Weighting'
            })
            
        except Exception as e:
            print(f"Error in product forecasting: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _calculate_recency_score(self, last_order_date):
        """Calculate recency score based on how recent the last order was."""
        if pd.isna(last_order_date):
            return 0
        
        days_since = (datetime.now() - last_order_date).days
        if days_since <= 7:
            return 10
        elif days_since <= 30:
            return 8
        elif days_since <= 60:
            return 6
        elif days_since <= 90:
            return 4
        else:
            return 2
    
    def _get_likelihood_category(self, demand_score):
        """Convert demand score to likelihood category."""
        if demand_score >= 100:
            return 'Very High'
        elif demand_score >= 50:
            return 'High'
        elif demand_score >= 25:
            return 'Medium'
        else:
            return 'Low'
    
    def get_dashboard_metrics(self):
        """Get simplified metrics for dashboard display."""
        if self.data.empty:
            return None
        
        try:
            df = self.data.copy()
            
            # Basic business metrics
            total_revenue = df['Total Price'].sum()
            total_orders = len(df)
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            
            # Profit estimation (60% margin)
            estimated_profit = total_revenue * 0.60
            profit_margin = 60.0  # Fixed for perfume business
            
            return _to_native({
                'total_revenue': round(total_revenue, 2),
                'total_orders': int(total_orders),
                'avg_order_value': round(avg_order_value, 2),
                'estimated_profit': round(estimated_profit, 2),
                'total_profit': round(estimated_profit, 2),  # Template compatibility
                'profit_margin': profit_margin,
                'avg_profit_per_order': round(estimated_profit / total_orders, 2) if total_orders > 0 else 0,
                'data_period': f"{df['Order Date'].min().date()} to {df['Order Date'].max().date()}",
                'data_source': 'Nuella Dataset'
            })
            
        except Exception as e:
            print(f"Error calculating dashboard metrics: {e}")
            return None
