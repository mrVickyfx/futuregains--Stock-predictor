from flask import Flask, render_template, request, jsonify, send_file
import yfinance as yf
from model import predict_stock_price, get_historical_performance, get_news, DataValidationError, get_available_stocks
import pandas as pd
from datetime import datetime, timedelta
import io
import csv
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_symbol(symbol):
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Invalid stock symbol")
    # Remove any whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    # Check if symbol contains only valid characters
    if not all(c.isalnum() or c in '.-' for c in symbol):
        raise ValueError("Invalid characters in stock symbol")
    return symbol

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate stock symbol
        symbol = validate_symbol(request.form.get('symbol', ''))
        
        # Get historical data
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({'error': f'No data found for symbol: {symbol}'})
        
        # Get prediction and metrics
        result = predict_stock_price(df)
        
        # Get historical performance
        performance = get_historical_performance(symbol)
        
        # Get news
        news = get_news(symbol)
        
        return jsonify({
            'prediction': round(result['prediction'], 2),
            'lower_bound': round(result['lower_bound'], 2),
            'upper_bound': round(result['upper_bound'], 2),
            'current_price': round(float(df['Close'].iloc[-1]), 2),
            'symbol': symbol,
            'metrics': {
                'mae': round(result['metrics']['mae'], 2),
                'rmse': round(result['metrics']['rmse'], 2),
                'r2': round(result['metrics']['r2'] * 100, 1)
            },
            'confidence': {
                'level': round(result['confidence']['level'], 1),
                'spread': round(result['confidence']['spread'], 1),
                'trend': result['confidence']['trend']
            },
            'performance': performance,
            'news': news
        })
        
    except DataValidationError as e:
        logger.warning(f"Data validation error for symbol {request.form.get('symbol', '')}: {str(e)}")
        return jsonify({'error': str(e)})
    except ValueError as e:
        logger.warning(f"Value error: {str(e)}")
        return jsonify({'error': str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'})

@app.route('/export', methods=['POST'])
def export_data():
    try:
        symbol = validate_symbol(request.form.get('symbol', ''))
        stock = yf.Ticker(symbol)
        df = stock.history(period='1y')
        
        if df.empty:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Write data
        for index, row in df.iterrows():
            writer.writerow([
                index.strftime('%Y-%m-%d'),
                round(row['Open'], 2),
                round(row['High'], 2),
                round(row['Low'], 2),
                round(row['Close'], 2),
                int(row['Volume'])
            ])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_historical_data.csv'
        )
        
    except Exception as e:
        logger.error(f"Error in export_data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

@app.route('/stocks')
def get_stocks():
    try:
        stocks = get_available_stocks()
        return jsonify({'stocks': stocks})
    except Exception as e:
        logger.error(f"Error fetching stocks list: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 