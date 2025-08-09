import pytest
import types
import pandas as pd
import numpy as np

# Import the Flask app module dynamically (file name contains space & parens)
import importlib.util, sys, pathlib
APP_PATH = pathlib.Path(__file__).resolve().parent.parent / 'app (1).py'
spec = importlib.util.spec_from_file_location('ella_app', APP_PATH)
ella_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ella_app)


def test_calculate_profitability_prediction_basic(monkeypatch):
    # Ensure live metrics return controlled values
    def fake_live_metrics():
        return {
            'total_revenue': 10000,
            'total_orders': 100,
            'avg_order_value': 100,
            'estimated_profit': 4000,
            'profit_margin': 40,
            'avg_profit_per_order': 40
        }
    monkeypatch.setattr(ella_app, 'get_live_business_metrics', fake_live_metrics)

    result = ella_app.calculate_profitability_prediction(target_profit=20000, fixed_costs=5000, growth_factor=1.0)
    assert 'orders_needed_monthly' in result
    # Orders needed = (target_profit + fixed_costs) / avg_profit_per_order = 25000/40 = 625
    assert result['orders_needed_monthly'] == 625
    assert result['orders_needed_daily'] == pytest.approx(20.8, rel=1e-2)
    assert result['data_source'] == 'live'


def test_calculate_profitability_prediction_growth_factor(monkeypatch):
    def fake_live_metrics():
        return {
            'total_revenue': 5000,
            'total_orders': 50,
            'avg_order_value': 100,
            'estimated_profit': 2000,
            'profit_margin': 40,
            'avg_profit_per_order': 40
        }
    monkeypatch.setattr(ella_app, 'get_live_business_metrics', fake_live_metrics)

    result = ella_app.calculate_profitability_prediction(target_profit=10000, fixed_costs=3000, growth_factor=1.5)
    # adjusted profit per order = 40 * 1.5 = 60
    # total profit needed = 13000
    # orders_needed = 216.66 -> ceil -> 217
    assert result['adjusted_profit_per_order'] == 60
    assert result['orders_needed_monthly'] == 217


def test_calculate_profitability_prediction_fallback(monkeypatch):
    # Force live metrics to return None
    def fake_live_metrics():
        return None
    monkeypatch.setattr(ella_app, 'get_live_business_metrics', fake_live_metrics)

    # Provide profitability_model_data fallback
    ella_app.profitability_model_data = {
        'avg_profit_per_order': 25,
        'avg_revenue_per_order': 120
    }
    result = ella_app.calculate_profitability_prediction(target_profit=5000, fixed_costs=5000, growth_factor=1.0)
    # orders needed = (5000+5000)/25 = 400
    assert result['orders_needed_monthly'] == 400
    assert result['data_source'] == 'model'


def test_analyze_product_sales_forecast_no_dates(monkeypatch):
    # Create empty / invalid dates orders dataframe
    ella_app.orders_df = pd.DataFrame({
        'order_id': [], 'date': [], 'order_type': [], 'payment_method': [], 'shipping_mode': [], 'order_status': [], 'customer_id': [], 'product_id': [], 'quantity': []
    })
    # Minimal products
    ella_app.products_df = pd.DataFrame({
        'product_id': ['P1'], 'name': ['Prod1'], 'brand': ['B'], 'size': ['S'], 'unit_price': [50], 'type': ['TypeA'], 'quantity': [20]
    })
    forecast = ella_app.analyze_product_sales_forecast()
    assert 'error' in forecast
    assert 'No valid historical order dates' in forecast['error']


def test_analyze_product_sales_forecast_basic(monkeypatch):
    # Build a simple dataset with two products and recent sales
    ella_app.orders_df = pd.DataFrame({
        'order_id': ['O1','O2','O3','O4'],
        'date': ['01/07/2025','10/07/2025','25/07/2025','05/08/2025'],
        'order_type': ['Retail']*4,
        'payment_method': ['Mobile Money']*4,
        'shipping_mode': ['Delivery']*4,
        'order_status': ['Completed']*4,
        'customer_id': ['C1','C2','C1','C3'],
        'product_id': ['P1','P1','P2','P2'],
        'quantity': [5,3,7,4]
    })
    ella_app.products_df = pd.DataFrame({
        'product_id': ['P1','P2','P3'],
        'name': ['Prod1','Prod2','Prod3'],
        'brand': ['B','B','B'],
        'size': ['S','S','S'],
        'unit_price': [40,60,90],
        'type': ['TypeA','TypeA','TypeB'],
        'quantity': [15,8,70]
    })
    forecast = ella_app.analyze_product_sales_forecast()
    assert 'products' in forecast
    # P3 has no sales -> should still appear (maybe low likelihood) among top slice if <=20 products
    product_ids = [p['product_id'] for p in forecast['products']]
    assert 'P1' in product_ids and 'P2' in product_ids and 'P3' in product_ids
    # Likelihood percent should be between 0 and 100
    for p in forecast['products']:
        assert 0 <= p['likelihood_percent'] <= 100


def test_get_30day_recommendation_rules():
    rec = ella_app.get_30day_recommendation(purchase_score=50, stock_level=10, likelihood_percent=90, predicted_sales=100)
    assert 'URGENT' in rec or 'PRIORITY' in rec
    rec2 = ella_app.get_30day_recommendation(purchase_score=5, stock_level=100, likelihood_percent=10, predicted_sales=2)
    # Expect EXCESS or MINIMAL depending on thresholds
    assert any(keyword in rec2 for keyword in ['EXCESS', 'MINIMAL', 'LOW'])

