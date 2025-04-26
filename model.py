import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

class DataValidationError(Exception):
    pass

def create_features(df):
    try:
        # Reduce minimum data requirement from 30 to 20 days
        if len(df) < 20:
            raise DataValidationError(f"Insufficient historical data (need at least 20 days, got {len(df)} days)")
            
        df = df.copy()
        
        # Handle missing values before calculations
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Technical indicators with error handling and shorter windows
        try:
            # Shorter SMA windows
            df['SMA_7'] = df['Close'].rolling(window=7, min_periods=1).mean()
            df['SMA_20'] = df['Close'].rolling(window=14, min_periods=1).mean()
            
            # RSI with shorter period
            df['RSI'] = calculate_rsi(df['Close'], period=7)
            
            # Shorter EMA windows
            df['EMA_12'] = df['Close'].ewm(span=8, min_periods=1).mean()
            df['EMA_26'] = df['Close'].ewm(span=17, min_periods=1).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            
            # Volume and volatility with shorter windows
            df['Volume_MA'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)
            df['Volatility'] = df['Daily_Return'].rolling(window=10, min_periods=1).std()
            
            # Create target variable (next day's closing price)
            df['Target'] = df['Close'].shift(-1)
            
        except Exception as e:
            raise DataValidationError(f"Error calculating technical indicators: {str(e)}")
            
        return df
    except Exception as e:
        raise DataValidationError(f"Error in create_features: {str(e)}")

def calculate_rsi(prices, period=7):
    try:
        delta = prices.diff()
        delta = delta.fillna(0)
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, np.inf)
        rs = gain / loss
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill any remaining NaN with neutral RSI
    except Exception as e:
        raise DataValidationError(f"Error calculating RSI: {str(e)}")

def get_news(symbol):
    try:
        all_news = []
        
        # Try Yahoo Finance first
        try:
            stock = yf.Ticker(symbol)
            yahoo_news = stock.news
            if yahoo_news:
                for item in yahoo_news[:10]:  # Get latest 10 news items
                    # Skip items without title or summary
                    if not item.get('title') or not item.get('summary'):
                        continue
                        
                    cleaned_item = {
                        'title': item.get('title'),
                        'summary': item.get('summary', item.get('description', '')).strip(),
                        'time_published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                        'url': item.get('link', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'sentiment': analyze_sentiment(item.get('title', '') + ' ' + item.get('summary', ''))
                    }
                    all_news.append(cleaned_item)
        except Exception as e:
            print(f"Error fetching Yahoo news: {str(e)}")

        # Try Finnhub as second source
        if len(all_news) < 5 and FINNHUB_API_KEY:
            try:
                headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}"
                response = requests.get(url, headers=headers)
                finnhub_news = response.json()
                
                for item in finnhub_news[:10]:
                    if not item.get('headline') or not item.get('summary'):
                        continue
                        
                    cleaned_item = {
                        'title': item.get('headline'),
                        'summary': item.get('summary').strip(),
                        'time_published': datetime.fromtimestamp(item.get('datetime', 0)),
                        'url': item.get('url', ''),
                        'source': item.get('source', 'Finnhub'),
                        'sentiment': analyze_sentiment(item.get('headline', '') + ' ' + item.get('summary', ''))
                    }
                    all_news.append(cleaned_item)
            except Exception as e:
                print(f"Error fetching Finnhub news: {str(e)}")

        # Try Alpha Vantage as last resort
        if len(all_news) < 5 and ALPHA_VANTAGE_API_KEY:
            try:
                url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
                
                response = requests.get(url, timeout=10)
                data = response.json()
                av_news = data.get('feed', [])
                
                for item in av_news[:10]:
                    if not item.get('title') or not item.get('summary'):
                        continue
                        
                    cleaned_item = {
                        'title': item.get('title'),
                        'summary': item.get('summary').strip(),
                        'time_published': datetime.strptime(
                            item.get('time_published', datetime.now().strftime('%Y%m%dT%H%M%S')),
                            '%Y%m%dT%H%M%S'
                        ),
                        'url': item.get('url', ''),
                        'source': item.get('source', 'Alpha Vantage'),
                        'sentiment': item.get('overall_sentiment_score', 0)
                    }
                    all_news.append(cleaned_item)
            except Exception as e:
                print(f"Error fetching Alpha Vantage news: {str(e)}")

        # Filter out duplicates based on title
        seen_titles = set()
        filtered_news = []
        for item in all_news:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                filtered_news.append(item)

        # Sort news by date and get most recent
        filtered_news.sort(key=lambda x: x['time_published'], reverse=True)
        
        # Return at least one placeholder if no news found
        if not filtered_news:
            return [{
                'title': 'No recent news available',
                'summary': 'Try checking financial news websites for the latest updates.',
                'time_published': datetime.now(),
                'url': '',
                'source': 'System',
                'sentiment': 0
            }]
            
        return filtered_news[:5]  # Return 5 most recent news items
        
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return [{
            'title': 'Error fetching news',
            'summary': 'Temporarily unable to fetch news. Please try again later.',
            'time_published': datetime.now(),
            'url': '',
            'source': 'System',
            'sentiment': 0
        }]

def analyze_sentiment(text):
    """Simple sentiment analysis based on keywords"""
    positive_words = set(['up', 'rise', 'gain', 'positive', 'growth', 'profit', 'success', 'higher'])
    negative_words = set(['down', 'fall', 'loss', 'negative', 'decline', 'drop', 'lower', 'risk'])
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total = positive_count + negative_count
    if total == 0:
        return 0
    return (positive_count - negative_count) / total

def predict_stock_price(df):
    try:
        if len(df) < 30:
            raise DataValidationError(f"Insufficient data for reliable prediction (need at least 30 days, got {len(df)} days)")
            
        # Create features with error handling
        try:
            df = create_features(df)
            df = df.dropna()
        except Exception as e:
            raise DataValidationError(f"Error preparing data: {str(e)}")
        
        if len(df) == 0:
            raise DataValidationError("No valid data after preprocessing")
            
        features = ['Close', 'SMA_7', 'SMA_20', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'Volume_MA', 'Volatility']
        
        # Verify all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise DataValidationError(f"Missing features: {', '.join(missing_features)}")
        
        X = df[features]
        y = df['Target']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with more estimators and better parameters
        model = RandomForestRegressor(
            n_estimators=200,  # Increased from 100
            max_depth=15,      # Added parameter
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Calculate base metrics
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Make prediction for next day
        last_data = scaler.transform(X.iloc[-1:])
        
        # Get prediction and confidence interval
        predictions = []
        for estimator in model.estimators_:
            pred = estimator.predict(last_data)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        prediction = float(np.mean(predictions))
        conf_interval = np.percentile(predictions, [5, 95])
        
        # Calculate confidence metrics
        pred_std = np.std(predictions)
        pred_spread = (conf_interval[1] - conf_interval[0]) / prediction if prediction != 0 else 1
        
        # Calculate weighted confidence level
        volatility_factor = df['Volatility'].iloc[-1] / df['Volatility'].mean()
        volume_factor = df['Volume'].iloc[-1] / df['Volume'].mean()
        
        confidence_level = max(min(
            (1 - pred_spread) * 0.3 +           # Prediction spread weight
            (r2 * 0.3) +                        # Model accuracy weight
            (1 - volatility_factor) * 0.2 +     # Volatility weight
            (min(volume_factor, 2) / 2) * 0.2,  # Volume weight
            1.0
        ), 0.0) * 100
        
        # Calculate trend and momentum
        current_price = float(df['Close'].iloc[-1])
        price_change = ((prediction - current_price) / current_price * 100) if current_price != 0 else 0
        momentum = calculate_momentum(df)
        
        return {
            'prediction': float(prediction),
            'lower_bound': float(conf_interval[0]),
            'upper_bound': float(conf_interval[1]),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            },
            'confidence': {
                'level': float(confidence_level),
                'spread': float(pred_spread * 100),
                'trend': {
                    'direction': 'up' if prediction > current_price else 'down',
                    'strength': float(abs(price_change)),
                    'percentage': float(price_change),
                    'momentum': momentum
                }
            }
        }
        
    except DataValidationError as e:
        raise DataValidationError(str(e))
    except Exception as e:
        raise DataValidationError(f"Error in prediction process: {str(e)}")

def calculate_momentum(df):
    """Calculate price momentum indicator"""
    try:
        # Calculate short and long term momentum
        short_term = df['Close'].pct_change(5).iloc[-1]  # 5-day momentum
        medium_term = df['Close'].pct_change(10).iloc[-1] # 10-day momentum
        long_term = df['Close'].pct_change(20).iloc[-1]   # 20-day momentum
        
        # Weight the momentum periods
        weighted_momentum = (
            short_term * 0.5 +    # 50% weight to short term
            medium_term * 0.3 +   # 30% weight to medium term
            long_term * 0.2       # 20% weight to long term
        )
        
        # Classify momentum strength
        if abs(weighted_momentum) < 0.02:
            return 'neutral'
        elif weighted_momentum > 0:
            return 'strong' if weighted_momentum > 0.05 else 'moderate'
        else:
            return 'weak' if weighted_momentum > -0.05 else 'very weak'
            
    except Exception:
        return 'neutral'

def get_historical_performance(symbol, period='1y'):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if len(hist) == 0:
            raise DataValidationError("No historical data available")
            
        performance = {
            'start_price': float(hist['Close'].iloc[0]),
            'end_price': float(hist['Close'].iloc[-1]),
            'return': float(((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100),
            'highest_price': float(hist['High'].max()),
            'lowest_price': float(hist['Low'].min()),
            'average_volume': float(hist['Volume'].mean()),
            'price_history': [float(x) for x in hist['Close'].tolist()],
            'dates': [x.strftime('%Y-%m-%d') for x in hist.index.tolist()]
        }
        
        return performance
    except Exception as e:
        raise DataValidationError(f"Error fetching historical performance: {str(e)}")

def get_available_stocks():
    try:
        # List of popular stocks (expanded list)
        available_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'AMD', 'INTC',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'V', 'MA',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
            # Consumer
            'PG', 'KO', 'PEP', 'WMT', 'MCD', 'DIS', 'NFLX', 'NKE', 'SBUX', 'HD',
            # Industrial
            'XOM', 'CVX', 'GE', 'BA', 'CAT', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
            # Others
            'VZ', 'T', 'IBM', 'QCOM', 'TXN', 'PYPL', 'BABA', 'TSM', 'ASML', 'TMUS'
        ]
        
        # Get basic info for each stock
        stocks_info = []
        for symbol in available_stocks:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Get current price
                current_price = info.get('currentPrice', 0)
                if not current_price:
                    current_price = info.get('regularMarketPrice', 0)
                
                stocks_info.append({
                    'symbol': symbol,
                    'name': info.get('shortName', info.get('longName', symbol)),
                    'sector': info.get('sector', info.get('industry', 'N/A')),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': current_price,
                    'currency': info.get('currency', 'USD'),
                    'exchange': info.get('exchange', 'N/A')
                })
            except Exception as e:
                print(f"Error fetching info for {symbol}: {str(e)}")
                continue
        
        # Sort by market cap
        stocks_info = [s for s in stocks_info if s['market_cap'] > 0]  # Filter out invalid entries
        stocks_info.sort(key=lambda x: x['market_cap'], reverse=True)
        return stocks_info
        
    except Exception as e:
        print(f"Error getting available stocks: {str(e)}")
        return [] 