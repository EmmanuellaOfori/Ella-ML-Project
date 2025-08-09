"""Data management module for loading and saving CSV data."""

import pandas as pd
from config import Config

class DataManager:
    def __init__(self):
        self.customers_df = None
        self.orders_df = None
        self.products_df = None
        self.load_all_data()
    
    def load_data(self):
        """Load and normalize CSV data."""
        try:
            # Read customers data
            customers = pd.read_csv(Config.CUSTOMERS_CSV)
            # Orders raw read
            orders = pd.read_csv(Config.ORDERS_CSV)
            products = pd.read_csv(Config.PRODUCTS_CSV)
            
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
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return empty DataFrames if files don't exist
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def load_all_data(self):
        """Load all data and store in instance variables."""
        self.customers_df, self.orders_df, self.products_df = self.load_data()
    
    def save_data(self):
        """Save all data to CSV files."""
        try:
            self.customers_df.to_csv(Config.CUSTOMERS_CSV, index=False)
            self.orders_df.to_csv(Config.ORDERS_CSV, index=False)
            self.products_df.to_csv(Config.PRODUCTS_CSV, index=False)
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def get_customers(self):
        return self.customers_df
    
    def get_orders(self):
        return self.orders_df
    
    def get_products(self):
        return self.products_df
    
    def add_customer(self, customer_data):
        """Add a new customer to the dataframe."""
        if customer_data['customer_id'] in self.customers_df['customer_id'].values:
            return False, "Customer ID already exists"
        
        new_customer_df = pd.DataFrame([customer_data])
        self.customers_df = pd.concat([self.customers_df, new_customer_df], ignore_index=True)
        return True, "Customer added successfully"
    
    def add_product(self, product_data):
        """Add a new product to the dataframe."""
        if product_data['product_id'] in self.products_df['product_id'].values:
            return False, "Product ID already exists"
        
        new_product_df = pd.DataFrame([product_data])
        self.products_df = pd.concat([self.products_df, new_product_df], ignore_index=True)
        return True, "Product added successfully"
    
    def update_product(self, product_id, updated_data):
        """Update an existing product."""
        if product_id not in self.products_df['product_id'].values:
            return False, "Product not found"
        
        for key, value in updated_data.items():
            self.products_df.loc[self.products_df['product_id'] == product_id, key] = value
        return True, "Product updated successfully"
    
    def delete_product(self, product_id):
        """Delete a product."""
        if product_id not in self.products_df['product_id'].values:
            return False, "Product not found"
        
        self.products_df = self.products_df[self.products_df['product_id'] != product_id]
        return True, "Product deleted successfully"
