import pytest
import pathlib, importlib.util
import pandas as pd

APP_PATH = pathlib.Path(__file__).resolve().parent.parent / 'app (1).py'
spec = importlib.util.spec_from_file_location('ella_app_api', APP_PATH)
ella_app_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ella_app_api)

@pytest.fixture
def client():
    app = ella_app_api.app
    app.config['TESTING'] = True
    with app.test_client() as client:
        # Default: logged in for most tests
        with client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'tester'
        yield client


def setup_sample_data():
    # Provide small deterministic dataset
    ella_app_api.products_df = pd.DataFrame({
        'product_id': ['P1','P2','P3'],
        'name': ['Prod1','Prod2','Prod3'],
        'brand': ['B','B','C'],
        'size': ['S','S','M'],
        'unit_price': [40, 60, 90],
        'type': ['TypeA','TypeA','TypeB'],
        'quantity': [15, 8, 70]
    })
    ella_app_api.orders_df = pd.DataFrame({
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
    ella_app_api.customers_df = pd.DataFrame({
        'customer_id': ['C1','C2','C3'],
        'name': ['Cust1','Cust2','Cust3'],
        'country': ['X','X','Y'],
        'region': ['R1','R1','R2'],
        'city': ['City1','City2','City3']
    })
    ella_app_api.profitability_model_data = {
        'avg_profit_per_order': 25,
        'avg_revenue_per_order': 120,
        'model_summary': {},
        'scenarios': {}
    }


def test_live_metrics_unauthorized():
    app = ella_app_api.app
    app.config['TESTING'] = True
    with app.test_client() as c:
        resp = c.get('/api/live-metrics')
        assert resp.status_code == 401
        data = resp.get_json()
        assert data['success'] is False


def test_live_metrics_authorized(client):
    setup_sample_data()
    resp = client.get('/api/live-metrics')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    for key in ['total_orders','total_revenue','total_profit']:
        assert key in data['metrics']


def test_profitability_endpoint(client):
    setup_sample_data()
    payload = {"target_profit": 5000, "fixed_costs": 4000, "growth_factor": 1.0}
    resp = client.post('/api/profitability', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'orders_needed_monthly' in data
    assert data['orders_needed_monthly'] > 0


def test_predict_price_heuristic(client):
    setup_sample_data()
    resp = client.post('/api/predict-price', json={"product_type": "TypeA", "customer_segment": "premium", "quantity": 3})
    data = resp.get_json()
    assert resp.status_code == 200
    assert data['success'] is True
    assert data['predicted_price'] > 0
    assert data['profit_margin'] in [25,30,35]


def test_product_forecast_endpoint(client):
    setup_sample_data()
    resp = client.get('/api/product-forecast')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    forecast = data['forecast']
    assert 'products' in forecast
    assert forecast['summary']['total_products_analyzed'] >= len(forecast['products'])


def test_purchase_forecast_endpoint(client):
    setup_sample_data()
    resp = client.post('/api/purchase-forecast', json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert 'summary' in data['forecast']


def test_trending_analysis_endpoint(client):
    setup_sample_data()
    resp = client.get('/api/trending-analysis')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert 'summary' in data['trends']
