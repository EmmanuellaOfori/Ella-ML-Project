from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import os
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load profitability model
try:
    with open('profitability_model.pkl', 'rb') as f:
        profitability_model_data = pickle.load(f)
    print("✅ Profitability model loaded successfully")
except FileNotFoundError:
    profitability_model_data = None
    print("⚠️ Profitability model not found. Run the ML notebook first.")

# Load the Random Forest model for price predictions
try:
    with open('Forest_ML.pkl', 'rb') as f:
        price_prediction_model = pickle.load(f)
    print("✅ Price prediction model loaded successfully")
except FileNotFoundError:
    price_prediction_model = None
    print("⚠️ Price prediction model not found.")

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

def get_live_business_metrics():
    """Calculate real-time business metrics from current data"""
    try:
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
        estimated_cogs = float(total_revenue * 0.4)
        estimated_fixed_costs = 5000.0
        estimated_profit = float(total_revenue - estimated_cogs - estimated_fixed_costs)
        profit_margin = float((estimated_profit / total_revenue * 100)) if total_revenue > 0 else 0.0
        return {
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'estimated_profit': estimated_profit,
            'profit_margin': profit_margin,
            'avg_profit_per_order': float(estimated_profit / total_orders) if total_orders > 0 else 0.0
        }
    except Exception as e:
        print(f"Error calculating live metrics: {e}")
        return None

def load_data():
    # Read customers data
    customers = pd.read_csv('customers.csv')
    # Orders raw read
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')
    # Strip whitespace in headers
    customers.columns = customers.columns.str.strip()
    orders.columns = orders.columns.str.strip()
    products.columns = products.columns.str.strip()
    # Rename to snake/lower
    rename_map_orders = {
        'order id': 'order_id', 'order type': 'order_type', 'payment method': 'payment_method',
        'shipping mode': 'shipping_mode', 'order status': 'order_status', 'customer id': 'customer_id',
        'product id': 'product_id'
    }
    orders = orders.rename(columns=rename_map_orders)
    rename_map_products = {'product id': 'product_id', 'unit price': 'unit_price'}
    products = products.rename(columns=rename_map_products)
    # Standardize to lowercase snake
    customers.columns = customers.columns.str.lower().str.replace(' ', '_')
    orders.columns = orders.columns.str.lower().str.replace(' ', '_')
    products.columns = products.columns.str.lower().str.replace(' ', '_')
    return customers, orders, products

# Save data to CSV files
def save_data(customers, orders, products):
    customers.to_csv('customers.csv', index=False)
    orders.to_csv('orders.csv', index=False)
    products.to_csv('products.csv', index=False)

# Initialize data
customers_df, orders_df, products_df = load_data()

# Landing page
@app.route('/')
def home():
    return render_template('index.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simple authentication (in a real app, use proper authentication)
        username = request.form['username']
        password = request.form['password']
        
        # For demo purposes, any non-empty password works
        if username and password:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

# Signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Simple signup (in a real app, add proper validation and password hashing)
        username = request.form['username']
        password = request.form['password']
        
        if username and password:
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Please fill all fields', 'error')
    
    return render_template('signup.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

#Dashboard
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Calculate dashboard metrics
    total_products = len(products_df)
    
    # Calculate total revenue - use correct column names
    try:
        merged = orders_df.merge(
            products_df, 
            left_on='product_id',  # Changed from 'product id'
            right_on='product_id',
            suffixes=('_order', '_product')
        )
        total_revenue = (merged['quantity'] * merged['unit_price']).sum()
    except KeyError as e:
        print(f"Merge error: {e}")
        print("Orders columns:", orders_df.columns.tolist())
        print("Products columns:", products_df.columns.tolist())
        total_revenue = 0  # Default value if merge fails
    
    # Low stock (quantity < 10)
    low_stock = products_df[products_df['quantity'] < 10]
    low_stock_count = len(low_stock)
    
    # Product types
    product_types = products_df['type'].nunique()
    
    # Recent activities (last 5 orders)
    recent_activities = orders_df.sort_values('date', ascending=False).head(5)
    
    # Get live business metrics for ML insights
    live_metrics = get_live_business_metrics()
    
    # ML-powered insights
    ml_insights = {}
    if live_metrics:
        ml_insights = {
            'profit_margin': round(live_metrics['profit_margin'], 1),
            'avg_profit_per_order': round(live_metrics['avg_profit_per_order'], 2),
            'break_even_orders': max(1, round(5000 / live_metrics['avg_profit_per_order'])) if live_metrics['avg_profit_per_order'] > 0 else 'N/A',
            'monthly_target_conservative': max(1, round(25000 / live_metrics['avg_profit_per_order'])) if live_metrics['avg_profit_per_order'] > 0 else 'N/A'
        }
    
    return render_template('dashboard.html', 
                         total_products=total_products,
                         total_revenue=total_revenue,
                         low_stock_count=low_stock_count,
                         product_types=product_types,
                         recent_activities=recent_activities.to_dict('records'),
                         live_metrics=live_metrics,
                         ml_insights=ml_insights)

# Products management
@app.route('/products', methods=['GET', 'POST'])
def products():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global products_df
    
    if request.method == 'POST':
        # Add new product
        new_product = {
            'product_id': request.form['id'],
            'name': request.form['name'],
            'brand': request.form['brand'],
            'size': request.form['size'],
            'unit_price': float(request.form['unit_price']),
            'type': request.form['type'],
            'quantity': int(request.form['quantity'])
        }
        
        # Check if product ID already exists
        if new_product['id'] in products_df['id'].values:
            flash('Product ID already exists!', 'error')
        else:
            products_df = pd.concat([products_df, pd.DataFrame([new_product])], ignore_index=True)
            save_data(customers_df, orders_df, products_df)
            flash('Product added successfully!', 'success')
    
    return render_template('products.html', products=products_df.to_dict('records'))

# Update product
@app.route('/update_product/<product_id>')
def update_product(product_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global products_df
    
    if request.method == 'POST':
        # Update product
        updated_product = {
            'name': request.form['name'],
            'brand': request.form['brand'],
            'size': request.form['size'],
            'unit_price': float(request.form['unit_price']),
            'type': request.form['type'],
            'quantity': int(request.form['quantity'])
        }
        
        products_df.loc[products_df['id'] == product_id, list(updated_product.keys())] = list(updated_product.values())
        save_data(customers_df, orders_df, products_df)
        flash('Product updated successfully!', 'success')
        return redirect(url_for('products'))
    
    product = products_df[products_df['product_id'] == product_id].iloc[0].to_dict()
    return render_template('update_product.html', product=product)

# Delete product
@app.route('/delete_product/<product_id>')
def delete_product(product_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global products_df
    
    products_df = products_df[products_df['product_id'] != product_id]
    save_data(customers_df, orders_df, products_df)
    flash('Product deleted successfully!', 'success')
    return redirect(url_for('products'))

# Orders management
@app.route('/orders')
def orders():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    global orders_df, products_df, customers_df
    
    try:
        # First merge orders with customers
        merged = orders_df.merge(
            customers_df,
            left_on='customer_id',
            right_on='customer_id',
            how='left',
            suffixes=('_order', '_customer')
        )
        
        # Then merge with products
        merged = merged.merge(
            products_df,
            left_on='product_id',
            right_on='product_id',
            how='left',
            suffixes=('', '_product')
        )
        
        # Select and rename columns
        display_orders = merged[[
            'order_id', 'date', 'name', 'customer_id',
            'name_product', 'product_id', 'quantity',
            'order_type', 'payment_method', 'shipping_mode', 'order_status'
        ]]
        
        return render_template('orders.html', 
                            orders=display_orders.to_dict('records'),
                            customers=customers_df.to_dict('records'),
                            products=products_df.to_dict('records'))
    
    except Exception as e:
        print(f"Error in orders route: {e}")
        print("Current columns:")
        print("Orders:", orders_df.columns.tolist())
        print("Customers:", customers_df.columns.tolist())
        print("Products:", products_df.columns.tolist())
        return "An error occurred while processing orders data", 500

# Customers management
@app.route('/customers', methods=['GET', 'POST'])
def customers():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global customers_df
    
    if request.method == 'POST':
        # Add new customer
        new_customer = {
            'customer_id': request.form['id'],
            'name': request.form['name'],
            'country': request.form['country'],
            'region': request.form['region'],
            'city': request.form['city']
        }
        
        # Check if customer ID already exists
        if new_customer['id'] in customers_df['id'].values:
            flash('Customer ID already exists!', 'error')
        else:
            customers_df = pd.concat([customers_df, pd.DataFrame([new_customer])], ignore_index=True)
            save_data(customers_df, orders_df, products_df)
            flash('Customer added successfully!', 'success')
    
    return render_template('customers.html', customers=customers_df.to_dict('records'))

# Reports
@app.route('/reports')
def reports():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global orders_df, products_df, customers_df
    
    # Merge data for analysis
    merged_data = orders_df.merge(products_df, left_on='product_id', right_on='product_id')
    merged_data = merged_data.merge(customers_df, left_on='customer_id', right_on='customer_id')
    
    # Top 10 most ordered products
    top_products = merged_data.groupby('name_x')['quantity_x'].sum().nlargest(10)
    
    # Shipping mode distribution
    shipping_dist = orders_df['shipping_mode'].value_counts()
    
    # Bottom 5 least ordered products
    bottom_products = merged_data.groupby('name_x')['quantity_x'].sum().nsmallest(5)
    
    # Top cities for orders
    top_cities = merged_data.groupby('city')['quantity_x'].sum().nlargest(5)
    
    # Order trends over time
    orders_df['date'] = pd.to_datetime(orders_df['date'], format='%d/%m/%Y')
    order_trend = orders_df.groupby(orders_df['date'].dt.to_period('M')).size()
    
    # Create plots
    def create_plot(data, title, plot_type='bar'):
        plt.figure(figsize=(10, 6))
        if plot_type == 'bar':
            data.plot(kind='bar')
        elif plot_type == 'pie':
            data.plot(kind='pie', autopct='%1.1f%%')
        plt.title(title)
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        return plot_url
    
    top_products_plot = create_plot(top_products, 'Top 10 Most Ordered Products')
    shipping_plot = create_plot(shipping_dist, 'Shipping Mode Distribution', 'pie')
    bottom_products_plot = create_plot(bottom_products, 'Bottom 5 Least Ordered Products')
    top_cities_plot = create_plot(top_cities, 'Top Cities by Order Volume')
    
    # Order trend plot (line chart)
    plt.figure(figsize=(10, 6))
    order_trend.plot(kind='line', marker='o')
    plt.title('Order Trend Over Time')
    plt.xlabel('Month')
    plt.ylabel('Number of Orders')
    plt.tight_layout()
    trend_img = BytesIO()
    plt.savefig(trend_img, format='png')
    trend_img.seek(0)
    trend_plot = base64.b64encode(trend_img.getvalue()).decode('utf8')
    plt.close()
    
    return render_template('reports.html',
                         top_products_plot=top_products_plot,
                         shipping_plot=shipping_plot,
                         bottom_products_plot=bottom_products_plot,
                         top_cities_plot=top_cities_plot,
                         trend_plot=trend_plot)

# Profitability Prediction Route
@app.route('/profitability', methods=['GET', 'POST'])
def profitability():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if not profitability_model_data:
        flash('Profitability model not available. Please run the ML analysis first.', 'error')
        return redirect(url_for('dashboard'))
    
    prediction_result = None
    
    if request.method == 'POST':
        try:
            target_profit = float(request.form['target_profit'])
            fixed_costs = float(request.form.get('fixed_costs', 5000))
            growth_factor = float(request.form.get('growth_factor', 1.0))
            
            # Calculate profitability prediction
            prediction_result = calculate_profitability_prediction(
                target_profit, fixed_costs, growth_factor
            )
            
        except (ValueError, KeyError) as e:
            flash('Invalid input. Please enter valid numbers.', 'error')
    
    # Get model summary and scenarios
    model_summary = profitability_model_data.get('model_summary', {})
    scenarios = profitability_model_data.get('scenarios', {})
    
    return render_template('profitability.html',
                         prediction=prediction_result,
                         model_summary=model_summary,
                         scenarios=scenarios)

# API endpoint for real-time profitability calculations
@app.route('/api/profitability', methods=['POST'])
def api_profitability():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401
    if not profitability_model_data:
        return jsonify({'error': 'Model not available'}), 500
    try:
        data = request.get_json()
        target_profit = float(data['target_profit'])
        fixed_costs = float(data.get('fixed_costs', 5000))
        growth_factor = float(data.get('growth_factor', 1.0))
        result = calculate_profitability_prediction(target_profit, fixed_costs, growth_factor)
        return jsonify(_to_native(result))
    except (ValueError, KeyError) as e:
        return jsonify({'error': 'Invalid input data'}), 400

def calculate_profitability_prediction(target_profit, fixed_costs=5000, growth_factor=1.0):
    """Calculate orders needed for target profitability using LIVE data"""
    live_metrics = get_live_business_metrics()
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

def predict_order_price(product_id, quantity, order_type='Retail', payment_method='Mobile Money', shipping_method='Delivery'):
    """
    Predict order price using the trained ML model with live product data
    """
    if not price_prediction_model:
        return None
    
    try:
        # Get product details from current inventory
        product = products_df[products_df['product_id'] == product_id]
        if product.empty:
            return {'error': 'Product not found'}
        
        product_data = product.iloc[0]
        
        # Create feature vector for prediction
        # This should match the features used in training
        features = {
            'Total Quantity': quantity,
            'Perfume Oils Quantity': quantity,
            'Body Splashes Quantity': 0,  # Default values
            'Boxed Perfume Quantity': 0,
            'Unit Price(Perfume Oil)': product_data['unit_price'],
            'Unit Price (Body Splashes)': 500,  # Average
            'Boxed Perfume(Unit Price)': 1000,  # Average
            'Shipping Cost': 25 if shipping_method == 'Delivery' else 0,
            'Order Type_Wholesale': 1 if order_type.lower() == 'wholesale' else 0,
            'Payment Method_Mobile Money': 1 if payment_method == 'Mobile Money' else 0,
            'Shipping Method_Delivery': 1 if shipping_method == 'Delivery' else 0
        }
        
        # Convert to format expected by model
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Make prediction
        predicted_price = price_prediction_model.predict(feature_array)[0]
        
        return {
            'predicted_price': round(predicted_price, 2),
            'product_name': product_data['name'],
            'unit_price': product_data['unit_price'],
            'quantity': quantity,
            'order_type': order_type,
            'confidence': 'Medium'  # Could be calculated from model metrics
        }
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

# Real-time dashboard metrics
@app.route('/api/live-metrics')
def api_live_metrics():
    """API endpoint for real-time business metrics (normalized for frontend)"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    metrics = get_live_business_metrics()
    if metrics:
        normalized = {
            'total_orders': int(metrics.get('total_orders', 0)),
            'total_revenue': float(round(metrics.get('total_revenue', 0), 2)),
            'total_profit': float(round(metrics.get('estimated_profit', 0), 2)),
            'profit_margin': float(round(metrics.get('profit_margin', 0), 2)),
            'avg_order_value': float(round(metrics.get('avg_order_value', 0), 2)),
            'avg_profit_per_order': float(round(metrics.get('avg_profit_per_order', 0), 2))
        }
        return jsonify({'success': True, 'metrics': _to_native(normalized)})
    else:
        return jsonify({'success': False, 'error': 'Unable to calculate metrics'}), 500

# Live price prediction API
@app.route('/api/predict-price', methods=['POST'])
def api_predict_price():
    """API endpoint for real-time price predictions (supports product_id or product_type heuristic)."""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    try:
        data = request.get_json() or {}
        quantity = int(data.get('quantity', 1))
        # Heuristic path if product_type provided (from profitability.html)
        product_type = data.get('product_type')
        customer_segment = data.get('customer_segment')
        product_id = data.get('product_id')
        if product_id and price_prediction_model:
            # Use ML model path
            prediction = predict_order_price(product_id, quantity,
                                             order_type=data.get('order_type', 'Retail'),
                                             payment_method=data.get('payment_method', 'Mobile Money'),
                                             shipping_method=data.get('shipping_method', 'Delivery'))
            if 'error' in prediction:
                return jsonify({'success': False, 'error': prediction['error']}), 400
            prediction['success'] = True
            return jsonify(prediction)
        elif product_type:
            # Simple heuristic: average unit_price of products in that type
            subset = products_df[products_df['type'].str.lower() == product_type.lower()]
            if subset.empty:
                # fallback to global avg
                avg_price = products_df['unit_price'].mean() if not products_df.empty else 0
            else:
                avg_price = subset['unit_price'].mean()
            # Adjust price by customer segment heuristic
            seg_multiplier = {
                'premium': 1.15,
                'regular': 1.0,
                'budget': 0.9
            }.get(str(customer_segment).lower(), 1.0)
            unit_price = round(avg_price * seg_multiplier, 2)
            predicted_price = round(unit_price * quantity, 2)
            profit_margin = 35 if customer_segment == 'premium' else 30 if customer_segment == 'regular' else 25
            return jsonify({
                'success': True,
                'predicted_price': predicted_price,
                'unit_price_recommended': unit_price,
                'quantity': quantity,
                'product_type': product_type,
                'customer_segment': customer_segment,
                'confidence': 'Heuristic',
                'profit_margin': profit_margin
            })
        else:
            return jsonify({'success': False, 'error': 'Provide product_id or product_type'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {e}'}), 500

# Sales Forecast Route
@app.route('/forecast')
def forecast():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('forecast.html')

# API endpoint for product sales forecast
@app.route('/api/product-forecast', methods=['GET'])
def product_forecast():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        # Analyze historical sales data
        forecast_data = analyze_product_sales_forecast()
        return jsonify({'success': True, 'forecast': forecast_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# API endpoint for 30-day purchase prediction
@app.route('/api/purchase-forecast', methods=['POST'])
def purchase_forecast():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        # Get 30-day purchase predictions for all products
        forecast_data = analyze_product_sales_forecast()
        return jsonify({'success': True, 'forecast': forecast_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# API endpoint for trending analysis
@app.route('/api/trending-analysis', methods=['GET'])
def trending_analysis():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        trending_data = analyze_product_trends()
        return jsonify({'success': True, 'trends': trending_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def analyze_product_trends():
    """Analyze product trends to identify rising and declining products"""
    try:
        orders_clean = orders_df.copy(); products_clean = products_df.copy()
        orders_clean.columns = orders_clean.columns.str.strip(); products_clean.columns = products_clean.columns.str.strip()
        if 'order id' in orders_clean.columns: orders_clean = orders_clean.rename(columns={'order id': 'order_id'})
        if 'customer id' in orders_clean.columns: orders_clean = orders_clean.rename(columns={'customer id': 'customer_id'})
        if 'product id' in orders_clean.columns: orders_clean = orders_clean.rename(columns={'product id': 'product_id'})
        if 'product name' in products_clean.columns: products_clean = products_clean.rename(columns={'product name': 'product_name'})
        if 'unit price' in products_clean.columns: products_clean = products_clean.rename(columns={'unit price': 'unit_price'})
        # Day-first parsing
        orders_clean['date'] = pd.to_datetime(orders_clean['date'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
        orders_clean = orders_clean.dropna(subset=['date'])
        if orders_clean.empty:
            return {'error': 'No valid dates for trend analysis'}
        orders_clean['month'] = orders_clean['date'].dt.to_period('M')
        sales_data = orders_clean.merge(products_clean, on='product_id')
        if sales_data.empty:
            return {'error': 'No sales data available for trends'}
        # Ensure product_name column exists (fallback to name)
        if 'product_name' not in sales_data.columns:
            if 'name' in sales_data.columns:
                sales_data['product_name'] = sales_data['name']
            else:
                sales_data['product_name'] = 'Unknown'
        # Identify order quantity column
        qty_candidates = [c for c in sales_data.columns if c.startswith('quantity')]
        order_qty_col = None
        # Prefer exact match to orders quantity before product inventory quantity
        for cand in ['quantity_x','quantity_order','quantity']:  # common variants
            if cand in sales_data.columns:
                order_qty_col = cand; break
        if order_qty_col is None and qty_candidates:
            order_qty_col = qty_candidates[0]
        if order_qty_col is None:
            return {'error': 'No quantity column available for trends'}
        sales_data['order_quantity'] = sales_data[order_qty_col]
        monthly_sales = sales_data.groupby(['product_id', 'product_name', 'month']).agg({
            'order_quantity': 'sum', 'order_id': 'count'
        }).reset_index()
        trending_products = []
        for product_id in monthly_sales['product_id'].unique():
            product_data = monthly_sales[monthly_sales['product_id'] == product_id]
            product_name = product_data['product_name'].iloc[0]
            if len(product_data) >= 2:
                months_numeric = list(range(len(product_data)))
                quantities = product_data['order_quantity'].tolist()
                trend_slope = (quantities[-1] - quantities[0]) / (len(months_numeric) - 1)
                avg_quantity = sum(quantities) / len(quantities)
                if avg_quantity == 0:
                    continue
                if trend_slope > avg_quantity * 0.2:
                    trend = 'Rising Strong'; trend_score = min(100, 70 + (trend_slope / avg_quantity) * 30)
                elif trend_slope > avg_quantity * 0.1:
                    trend = 'Rising'; trend_score = 60 + (trend_slope / avg_quantity) * 10
                elif trend_slope > -avg_quantity * 0.1:
                    trend = 'Stable'; trend_score = 50
                elif trend_slope > -avg_quantity * 0.2:
                    trend = 'Declining'; trend_score = 40 + (trend_slope / avg_quantity) * 10
                else:
                    trend = 'Declining Strong'; trend_score = max(10, 30 + (trend_slope / avg_quantity) * 20)
                trending_products.append({
                    'product_id': product_id,
                    'product_name': product_name,
                    'trend': trend,
                    'trend_score': round(float(trend_score), 1),
                    'trend_slope': round(float(trend_slope), 2),
                    'avg_monthly_sales': round(float(avg_quantity), 1),
                    'recent_sales': quantities[-1],
                    'months_analyzed': len(months_numeric)
                })
        trending_products.sort(key=lambda x: x['trend_score'], reverse=True)
        rising_products = [p for p in trending_products if p['trend'] in ['Rising Strong', 'Rising']]
        declining_products = [p for p in trending_products if p['trend'] in ['Declining Strong', 'Declining']]
        stable_products = [p for p in trending_products if p['trend'] == 'Stable']
        return {
            'all_trends': trending_products[:15],
            'rising_products': rising_products[:5],
            'declining_products': declining_products[:5],
            'stable_products': stable_products[:3],
            'summary': {
                'total_analyzed': len(trending_products),
                'rising_count': len(rising_products),
                'declining_count': len(declining_products),
                'stable_count': len(stable_products)
            }
        }
    except Exception as e:
        print(f"Error in trend analysis: {e}")
        import traceback; traceback.print_exc()
        return {'error': str(e)}

def analyze_product_sales_forecast():
    """Analyze which products are most likely to be purchased in the next 30 days"""
    try:
        orders_clean = orders_df.copy(); products_clean = products_df.copy()
        orders_clean.columns = orders_clean.columns.str.strip(); products_clean.columns = products_clean.columns.str.strip()
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
        date_max = orders_clean['date'].max(); date_min = orders_clean['date'].min()
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
                purchase_score = ((recent_quantity * 2.0) + (total_sold * 0.5) + (sales_frequency * 1.5) + (total_revenue / 200 * 1.0) + (max(0, 50 - stock_level) * 0.5) + (max(0, 80 - unit_price) * 0.3))
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
                    'recommended_action': get_30day_recommendation(purchase_score, stock_level, likelihood_percent, predicted_30day_sales)
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

def get_30day_recommendation(purchase_score, stock_level, likelihood_percent, predicted_sales):
    """
    Generate a 30‑day purchase recommendation using rule evaluation instead of if/elif chain.

    Args:
        purchase_score (float)
        stock_level (int|float)
        likelihood_percent (float)
        predicted_sales (float)
    Returns:
        str: Recommendation message.
    """
    rules = [
        # Each rule: (condition_lambda, message)
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



if __name__ == '__main__':
    app.run(debug=True)