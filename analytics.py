"""Business analytics and metrics calculation module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

class BusinessAnalytics:
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def get_live_business_metrics(self):
        """Calculate real-time business metrics from current data"""
        try:
            orders_df = self.data_manager.get_orders()
            products_df = self.data_manager.get_products()
            
            merged = orders_df.merge(products_df, on='product_id', how='left')
            
            # Ensure required columns exist; else create defaults
            if 'quantity' not in merged.columns:
                merged['quantity'] = 0
            if 'unit_price' not in merged.columns:
                # Try alternative legacy column names
                for alt in ['unit price', 'price']:
                    if alt in merged.columns:
                        merged['unit_price'] = merged[alt]
                        break
                if 'unit_price' not in merged.columns:
                    merged['unit_price'] = 0
            
            total_revenue = float((merged['quantity'].astype(float) * merged['unit_price'].astype(float)).sum())
            total_orders = int(len(orders_df))
            avg_order_value = float(total_revenue / total_orders) if total_orders > 0 else 0.0
            estimated_cogs = float(total_revenue * Config.DEFAULT_COGS_PERCENTAGE)
            estimated_fixed_costs = Config.DEFAULT_FIXED_COSTS
            estimated_profit = float(total_revenue - estimated_cogs - estimated_fixed_costs)
            profit_margin = float((estimated_profit / total_revenue * 100)) if total_revenue > 0 else 0.0
            
            return {
                'total_revenue': total_revenue,
                'total_orders': total_orders,
                'avg_order_value': avg_order_value,
                'estimated_profit': estimated_profit,
                'total_profit': estimated_profit,  # Template compatibility
                'profit_margin': profit_margin,
                'avg_profit_per_order': float(estimated_profit / total_orders) if total_orders > 0 else 0.0
            }
        except Exception as e:
            print(f"Error calculating live metrics: {e}")
            return None
    
    def calculate_profitability_prediction(self, target_profit, fixed_costs=5000, growth_factor=1.0, profitability_model_data=None):
        """Calculate orders needed for target profitability using LIVE data"""
        live_metrics = self.get_live_business_metrics()
        
        if not live_metrics or live_metrics['avg_profit_per_order'] <= 0:
            if profitability_model_data:
                avg_profit_per_order = float(profitability_model_data['avg_profit_per_order'])
                avg_revenue_per_order = float(profitability_model_data['avg_revenue_per_order'])
            else:
                return {'error': 'No data available for prediction'}
        else:
            avg_profit_per_order = float(live_metrics['avg_profit_per_order'])
            avg_revenue_per_order = float(live_metrics['avg_order_value'])
        
        adjusted_profit = avg_profit_per_order * float(growth_factor)
        total_profit_needed = float(target_profit) + float(fixed_costs)
        
        if adjusted_profit <= 0:
            return {
                'error': 'Negative profit per order. Check your business model.',
                'orders_needed_monthly': 0,
                'orders_needed_daily': 0
            }
        
        orders_needed = total_profit_needed / adjusted_profit
        
        return {
            'target_profit': float(target_profit),
            'fixed_costs': float(fixed_costs),
            'growth_factor': float(growth_factor),
            'orders_needed_monthly': int(np.ceil(orders_needed)),
            'orders_needed_daily': float(round(orders_needed / 30, 1)),
            'adjusted_profit_per_order': float(round(adjusted_profit, 2)),
            'total_profit_needed': float(total_profit_needed),
            'estimated_revenue': float(round(orders_needed * avg_revenue_per_order, 2)),
            'break_even_point': float(round(fixed_costs / adjusted_profit, 1)),
            'current_performance': {
                'avg_profit_per_order': float(round(avg_profit_per_order, 2)),
                'avg_revenue_per_order': float(round(avg_revenue_per_order, 2))
            },
            'live_metrics': live_metrics,
            'data_source': 'live' if live_metrics else 'model'
        }
    
    def analyze_product_sales_forecast(self):
        """Analyze which products are most likely to be purchased in the next 30 days"""
        try:
            orders_df = self.data_manager.get_orders()
            products_df = self.data_manager.get_products()
            
            orders_clean = orders_df.copy()
            products_clean = products_df.copy()
            
            orders_clean.columns = orders_clean.columns.str.strip()
            products_clean.columns = products_clean.columns.str.strip()
            
            if 'order id' in orders_clean.columns: orders_clean = orders_clean.rename(columns={'order id': 'order_id'})
            if 'customer id' in orders_clean.columns: orders_clean = orders_clean.rename(columns={'customer id': 'customer_id'})
            if 'product id' in orders_clean.columns: orders_clean = orders_clean.rename(columns={'product id': 'product_id'})
            if 'product name' in products_clean.columns: products_clean = products_clean.rename(columns={'product name': 'product_name'})
            if 'unit price' in products_clean.columns: products_clean = products_clean.rename(columns={'unit price': 'unit_price'})
            
            if 'quantity' not in orders_clean.columns: orders_clean['quantity'] = 0
            if 'quantity' not in products_clean.columns: products_clean['quantity'] = 0
            if 'unit_price' not in products_clean.columns:
                for alt in ['unit price', 'price']:
                    if alt in products_clean.columns: products_clean['unit_price'] = products_clean[alt]; break
                if 'unit_price' not in products_clean.columns: products_clean['unit_price'] = 0
            
            orders_clean['date'] = pd.to_datetime(orders_clean['date'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
            orders_clean = orders_clean.dropna(subset=['date'])
            if orders_clean.empty: return {'error': 'No valid historical order dates available'}
            
            date_max = orders_clean['date'].max()
            date_min = orders_clean['date'].min()
            sales_data = orders_clean.merge(products_clean, on='product_id', suffixes=('_order', '_product'))
            if sales_data.empty: return {'error': 'No merged sales data available'}
            
            # Determine order quantity column from merged dataset
            order_qty_col = None
            for cand in ['quantity_order','quantity_x','quantity']:
                if cand in sales_data.columns: order_qty_col = cand; break
            if order_qty_col is None: return {'error': 'No quantity column in merged sales data'}
            
            unit_price_col = 'unit_price'
            if unit_price_col not in sales_data.columns:
                for cand in ['unit_price_product','unit_price_x']:
                    if cand in sales_data.columns: unit_price_col = cand; break
            
            product_analysis = []
            total_span_days = max(1, (date_max - date_min).days)
            
            for product_id in products_clean['product_id'].unique():
                product_rows = products_clean[products_clean['product_id'] == product_id]
                if product_rows.empty: continue
                product_info = product_rows.iloc[0]
                product_sales = sales_data[sales_data['product_id'] == product_id]
                
                if not product_sales.empty:
                    total_sold = float(product_sales[order_qty_col].sum())
                    total_revenue = float((product_sales[order_qty_col] * product_sales[unit_price_col]).sum())
                    avg_order_size = float(product_sales[order_qty_col].mean())
                    sales_frequency = int(len(product_sales))
                    recent_cutoff = date_max - pd.Timedelta(days=30)
                    recent_sales = product_sales[product_sales['date'] >= recent_cutoff]
                    recent_quantity = float(recent_sales[order_qty_col].sum()) if not recent_sales.empty else 0.0
                    stock_level = float(product_info.get('quantity', 0))
                    unit_price = float(product_info.get('unit_price', 0))
                    
                    purchase_score = ((recent_quantity * 2.0) + (total_sold * 0.5) + (sales_frequency * 1.5) + 
                                    (total_revenue / 200 * 1.0) + (max(0, 50 - stock_level) * 0.5) + 
                                    (max(0, 80 - unit_price) * 0.3))
                    
                    if purchase_score >= 40:
                        likelihood = 'Very High'; likelihood_percent = min(95, 75 + (purchase_score - 40) * 0.5)
                    elif purchase_score >= 25:
                        likelihood = 'High'; likelihood_percent = 55 + (purchase_score - 25) * 1.33
                    elif purchase_score >= 10:
                        likelihood = 'Medium'; likelihood_percent = 30 + (purchase_score - 10) * 1.67
                    else:
                        likelihood = 'Low'; likelihood_percent = max(5, purchase_score * 3)
                    
                    daily_rate = total_sold / total_span_days if total_span_days > 0 else 0
                    predicted_30day_sales = round(daily_rate * 30, 1)
                    
                    product_analysis.append({
                        'product_id': product_id,
                        'product_name': product_info.get('product_name', product_info.get('name', 'Unknown')),
                        'product_type': product_info.get('type', 'unknown'),
                        'current_stock': stock_level,
                        'unit_price': unit_price,
                        'total_sold': int(total_sold),
                        'recent_sales': int(recent_quantity),
                        'sales_frequency': sales_frequency,
                        'total_revenue': round(total_revenue, 2),
                        'avg_order_size': round(avg_order_size, 1),
                        'demand_score': round(purchase_score, 1),
                        'likelihood': likelihood,
                        'likelihood_percent': round(likelihood_percent, 1),
                        'predicted_30day_sales': predicted_30day_sales,
                        'recommended_action': self.get_30day_recommendation(purchase_score, stock_level, likelihood_percent, predicted_30day_sales)
                    })
                else:
                    stock_level = float(product_info.get('quantity', 0))
                    unit_price = float(product_info.get('unit_price', 0))
                    purchase_score = (max(0, 50 - stock_level) * 0.5) + (max(0, 80 - unit_price) * 0.3)
                    
                    product_analysis.append({
                        'product_id': product_id,
                        'product_name': product_info.get('product_name', product_info.get('name', 'Unknown')),
                        'product_type': product_info.get('type', 'unknown'),
                        'current_stock': stock_level,
                        'unit_price': unit_price,
                        'total_sold': 0,
                        'recent_sales': 0,
                        'sales_frequency': 0,
                        'total_revenue': 0.0,
                        'avg_order_size': 0.0,
                        'demand_score': round(purchase_score, 1),
                        'likelihood': 'Low',
                        'likelihood_percent': max(5, round(purchase_score * 2, 1)),
                        'predicted_30day_sales': 0.0,
                        'recommended_action': '📋 New product - Consider promotional launch to generate initial sales'
                    })
            
            if not product_analysis: return {'error': 'No products available for analysis'}
            
            product_analysis.sort(key=lambda x: x['likelihood_percent'], reverse=True)
            
            category_performance = {}
            for analysis in product_analysis:
                cat = analysis['product_type'] or 'unknown'
                if cat not in category_performance:
                    category_performance[cat] = {'total_products': 0, 'avg_likelihood': 0, 'total_predicted_sales': 0, 'high_likelihood_count': 0}
                category_performance[cat]['total_products'] += 1
                category_performance[cat]['avg_likelihood'] += analysis['likelihood_percent']
                category_performance[cat]['total_predicted_sales'] += analysis['predicted_30day_sales']
                if analysis['likelihood_percent'] >= 60:
                    category_performance[cat]['high_likelihood_count'] += 1
            
            for cat, vals in category_performance.items():
                if vals['total_products'] > 0:
                    vals['avg_likelihood'] = round(vals['avg_likelihood'] / vals['total_products'], 1)
            
            return {
                'products': product_analysis[:20],
                'categories': category_performance,
                'summary': {
                    'total_products_analyzed': len(product_analysis),
                    'high_demand_products': len([p for p in product_analysis if p['likelihood_percent'] >= 60]),
                    'medium_demand_products': len([p for p in product_analysis if 30 <= p['likelihood_percent'] < 60]),
                    'low_demand_products': len([p for p in product_analysis if p['likelihood_percent'] < 30]),
                    'total_predicted_30day_sales': sum(p['predicted_30day_sales'] for p in product_analysis)
                }
            }
        except Exception as e:
            print(f"Error in 30-day purchase prediction: {e}")
            import traceback; traceback.print_exc()
            return {'error': str(e)}
    
    def get_30day_recommendation(self, purchase_score, stock_level, likelihood_percent, predicted_sales):
        """Generate a 30‑day purchase recommendation using rule evaluation."""
        rules = [
            (lambda ctx: ctx['likelihood'] >= 75 and ctx['stock'] < 20,
             "🚀 URGENT: High purchase likelihood + Low stock - Restock immediately!"),
            (lambda ctx: ctx['likelihood'] >= 65 and ctx['predicted'] > ctx['stock'] * 0.5,
             "📈 PRIORITY: Very likely to sell - Monitor stock levels closely"),
            (lambda ctx: ctx['likelihood'] >= 60,
             "⭐ STRONG: High purchase probability - Top candidate for next 30 days"),
            (lambda ctx: ctx['likelihood'] >= 40,
             "📊 MODERATE: Good purchase potential - Consider promotion to boost likelihood"),
            (lambda ctx: ctx['likelihood'] >= 25,
             "⚠️ LOW: Limited purchase likelihood - May need marketing push"),
            (lambda ctx: ctx['likelihood'] < 25 and ctx['stock'] > 50,
             "📦 EXCESS: Low demand + High stock - Consider discount or bundle deals"),
        ]
        
        ctx = {
            'score': purchase_score,
            'stock': stock_level,
            'likelihood': likelihood_percent,
            'predicted': predicted_sales
        }
        
        for cond, msg in rules:
            if cond(ctx):
                return msg
        
        return "📋 MINIMAL: Very low purchase probability - Review product positioning"
