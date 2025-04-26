# futuregains--Stock-predictor
A stock market predictor that forecasts next-day prices using machine learning and real-time data. 
## Setup

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` and add your API keys:
   - Get Alpha Vantage API key from: https://www.alphavantage.co/support/#api-key
   - Get Finnhub API key from: https://finnhub.io/register
6. Run the application:
   ```bash
   python app.py
   ``` 

## Features
- Real-time data fetching
- Predicts 1-day ahead stock prices
- Beautiful charts using D3.js/Chart.js

## Tools Used
JavaScript | Python | Flask | TensorFlow.js | D3.js | HTML/CSS

## Future Work
- Integrate with Power BI
- Add risk analysis
