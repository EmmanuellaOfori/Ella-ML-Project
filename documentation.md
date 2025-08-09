# Group 5 Stock Management System - User Documentation

## Overview

The Group 5 Stock Management System is a simplified business intelligence platform that provides real-time analytics and predictions for perfume business operations. The system uses the Nuella dataset containing 2,000 order records to deliver accurate business insights.

## System Requirements

- Python 3.7 or higher
- Flask web framework
- Web browser for accessing the interface
- Port 5000 available for local development

## Getting Started

### 1. Starting the Application

1. Navigate to the project directory
2. Run the command: `python app.py`
3. Open your web browser and go to: `http://127.0.0.1:5000`

### 2. Login Credentials

Use the following credentials to access the system:
- Username: `admin`
- Password: `admin123`

## System Features

### Dashboard

The main dashboard provides an overview of your business performance including:

- Total Products count
- Total Revenue amount
- Low Stock Items alert
- Product Types summary

#### Nuella Analytics Section

The dashboard includes real-time business intelligence showing:
- Current Profit Margin percentage
- Average Profit per Order
- Total Orders processed
- Average Order Value
- Business overview with total revenue, profit, and data period

### Core Business Functions

#### 1. 30-Day Profit Prediction

**Location**: Profitability page (accessible from dashboard or navigation menu)

**How to use**:
1. Click on "Profitability" in the navigation menu
2. Click the "Predict 30-Day Profit" button
3. View the prediction results including:
   - Predicted profit amount for the next 30 days
   - Confidence score of the prediction
   - Key metrics used in the calculation
   - Trend analysis and seasonal adjustments

#### 2. Product Forecasting

**Location**: Forecast page (accessible from dashboard or navigation menu)

**How to use**:
1. Click on "Forecast" in the navigation menu
2. Click the "Forecast Next Month Products" button
3. Review the forecasting results including:
   - Top products likely to be sold next month
   - Demand scores for each product
   - Product categories and details
   - Order frequency and quantity predictions

### Navigation Menu

The system includes the following pages accessible from the main navigation:

- **Dashboard**: Main overview page with live metrics
- **Products**: View product catalog and inventory
- **Orders**: Browse order history and details
- **Customers**: Access customer information
- **Reports**: Generate business reports
- **Profitability**: Access 30-day profit prediction tool
- **Forecast**: Access product forecasting tool
- **Logout**: Exit the system securely

### Data Management

#### Primary Dataset

The system uses the Nuella training dataset which contains:
- 2,000 perfume business orders
- Date range: January 2025 to June 2025
- 17 data columns including order details, pricing, and customer information
- Three main product categories: Perfume Oils, Body Splashes, and Boxed Perfumes

#### Real-time Updates

The dashboard automatically refreshes business metrics every 30 seconds to provide current information without manual intervention.

## Analytics Engine

### Profit Prediction Algorithm

The 30-day profit prediction uses:
- Time series trend analysis
- Seasonal adjustment factors
- Historical order patterns
- Recent business velocity calculations

### Product Forecasting Algorithm

The product forecasting system employs:
- Product frequency analysis
- Recency weighting for recent orders
- Demand score calculations
- Customer type analysis (retail vs wholesale)

## System Architecture

### Core Components

- **Flask Application**: Main web server handling requests
- **Nuella Analytics Engine**: Primary analytics and prediction system
- **Authentication Manager**: User login and session management
- **Data Manager**: Legacy data handling for backward compatibility
- **Configuration System**: Settings and dataset management

### File Structure

- `app.py`: Main application file
- `nuella_analytics.py`: Core analytics engine
- `auth_manager.py`: User authentication system
- `config.py`: System configuration
- `templates/`: HTML templates for web interface
- `Nuella_train.csv`: Primary dataset
- `users.json`: User account information

## Troubleshooting

### Common Issues

**Login Problems**:
- Verify username and password are correct
- Ensure the server is running on port 5000
- Check that users.json file exists

**Prediction Errors**:
- Confirm Nuella dataset is present and accessible
- Verify all required Python packages are installed
- Check server console for error messages

**Performance Issues**:
- Close unnecessary browser tabs
- Restart the Flask application
- Ensure adequate system memory is available

### Technical Support

For technical issues:
1. Check the server console output for error messages
2. Verify all system files are present
3. Ensure Python dependencies are properly installed
4. Restart the application if problems persist

## Security Considerations

- Change default admin credentials in production environments
- Use HTTPS for production deployments
- Regularly backup user data and business information
- Monitor system access logs for security purposes

## Updates and Maintenance

- Keep Python and Flask frameworks updated
- Regularly backup the Nuella dataset
- Monitor system performance and optimize as needed
- Update user credentials periodically for security

---

**System Version**: Nuella Analytics v1.0  
**Last Updated**: August 2025  
**Developed by**: Group 5 Team
