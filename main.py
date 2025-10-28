from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

CORS(app)

def calculate_seasonality(ticker, period='10y'):
    """
    Calculate seasonality trends for a given ticker
    Returns monthly average returns based on historical data
    """
    try:
        # Download historical data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change() * 100
        
        # Extract month and year
        df['Month'] = pd.DatetimeIndex(df.index).month
        df['Year'] = pd.DatetimeIndex(df.index).year
        df['MonthName'] = pd.DatetimeIndex(df.index).strftime('%B')
        
        # Calculate monthly statistics
        monthly_stats = df.groupby('Month').agg({
            'Returns': ['mean', 'std', 'count'],
            'MonthName': 'first'
        }).round(4)
        
        monthly_stats.columns = ['avg_return', 'std_dev', 'count', 'month_name']
        monthly_stats = monthly_stats.reset_index()
        
        # Calculate win rate (percentage of positive return days)
        win_rate = df.groupby('Month')['Returns'].apply(
            lambda x: (x > 0).sum() / len(x) * 100
        ).round(2)
        
        monthly_stats['win_rate'] = win_rate.values
        
        # Sort by month
        monthly_stats = monthly_stats.sort_values('Month')
        
        return monthly_stats.to_dict('records')
    
    except Exception as e:
        return {'error': str(e)}

def calculate_quarterly_seasonality(ticker, period='10y'):
    """Calculate seasonality by quarter"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        df['Returns'] = df['Close'].pct_change() * 100
        df['Quarter'] = pd.DatetimeIndex(df.index).quarter
        df['Year'] = pd.DatetimeIndex(df.index).year
        
        quarterly_stats = df.groupby('Quarter').agg({
            'Returns': ['mean', 'std', 'count']
        }).round(4)
        
        quarterly_stats.columns = ['avg_return', 'std_dev', 'count']
        quarterly_stats = quarterly_stats.reset_index()
        
        quarter_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarterly_stats['quarter_name'] = quarterly_stats['Quarter'].map(quarter_names)
        
        return quarterly_stats.to_dict('records')
    
    except Exception as e:
        return {'error': str(e)}
    
def calculate_ytd_trend(ticker):
    """Calculate Year-to-Date (YTD) cumulative performance trend."""
    try:
        stock = yf.Ticker(ticker)
        start = datetime(datetime.now().year, 1, 1)
        end = datetime.now()
        df = stock.history(start=start, end=end)

        if df.empty:
            return None

        # Compute cumulative performance (%)
        df['Cumulative'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        df['Date'] = df.index
        df['Month'] = df['Date'].dt.strftime('%B')
        df['Week'] = df['Date'].dt.isocalendar().week

        # Group by week for smoother line (optional)
        ytd = df.groupby('Week').agg({
            'Cumulative': 'last',
            'Month': 'first'
        }).reset_index()

        return ytd.to_dict('records')

    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def home():
    return jsonify({
        'message': 'yfinance Seasonality API',
        'endpoints': {
            '/seasonality/monthly': 'Get monthly seasonality trends',
            '/seasonality/quarterly': 'Get quarterly seasonality trends',
            '/seasonality/compare': 'Compare multiple indices'
        },
        'example': '/seasonality/monthly?ticker=^GSPC&period=10y'
    })

@app.route('/seasonality/weekly', methods=['GET'])
def weekly_seasonality():
    """
    Calculate average weekly performance and compare with YTD trend.
    Query params:
    - ticker: Stock/Index ticker (default: ^GSPC)
    - period: Data period (default: 10y)
    """
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '10y')

    try:
        # Get 10-year history
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return jsonify({'error': 'No data found for ticker'}), 404

        # Calculate weekly returns
        df = df.resample('W').last()  # take weekly closing prices
        df['Returns'] = df['Close'].pct_change() * 100
        df['Week'] = df.index.isocalendar().week
        df['Year'] = df.index.year

        # Calculate average return per week across all years
        weekly_avg = (
            df.groupby('Week')['Returns']
            .mean()
            .fillna(0)
            .reset_index()
            .rename(columns={'Returns': 'avg_return'})
        )

        # Calculate current year-to-date (YTD) cumulative trend
        start = datetime(datetime.now().year, 1, 1)
        end = datetime.now()
        ytd = stock.history(start=start, end=end)
        ytd['Cumulative'] = (ytd['Close'] / ytd['Close'].iloc[0] - 1) * 100
        ytd['Week'] = ytd.index.isocalendar().week

        ytd_weekly = (
            ytd.groupby('Week')['Cumulative']
            .last()
            .reset_index()
            .rename(columns={'Cumulative': 'ytd_return'})
        )

        # Merge both datasets for easy plotting
        merged = pd.merge(weekly_avg, ytd_weekly, on='Week', how='outer').fillna(0)

        return jsonify({
            'ticker': ticker,
            'period': period,
            'analysis_type': 'weekly',
            'weekly_avg': merged.to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/seasonality/monthly', methods=['GET'])
def monthly_seasonality():
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '10y')

    avg_result = calculate_seasonality(ticker, period)
    ytd_result = calculate_ytd_trend(ticker)

    if avg_result is None or isinstance(avg_result, dict) and 'error' in avg_result:
        return jsonify({'error': 'No historical data found'}), 404
    if ytd_result is None or isinstance(ytd_result, dict) and 'error' in ytd_result:
        ytd_result = []  # Fallback empty

    return jsonify({
        'ticker': ticker,
        'period': period,
        'analysis_type': 'monthly',
        'avg_10y': avg_result,
        'ytd_trend': ytd_result
    })

@app.route('/seasonality/quarterly', methods=['GET'])
def quarterly_seasonality():
    """
    Get quarterly seasonality trends
    Query params:
    - ticker: Stock/Index ticker
    - period: Data period (default: 10y)
    """
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '10y')
    
    result = calculate_quarterly_seasonality(ticker, period)
    
    if result is None:
        return jsonify({'error': 'No data found for ticker'}), 404
    
    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 400
    
    return jsonify({
        'ticker': ticker,
        'period': period,
        'analysis_type': 'quarterly',
        'data': result
    })

@app.route('/seasonality/compare', methods=['GET'])
def compare_seasonality():
    """
    Compare seasonality across multiple indices
    Query params:
    - tickers: Comma-separated list of tickers (e.g., ^GSPC,^DJI,^IXIC)
    - period: Data period (default: 10y)
    """
    tickers_param = request.args.get('tickers', '^GSPC,^DJI,^IXIC')
    period = request.args.get('period', '10y')
    
    tickers = [t.strip() for t in tickers_param.split(',')]
    
    results = {}
    for ticker in tickers:
        monthly_data = calculate_seasonality(ticker, period)
        if monthly_data and not (isinstance(monthly_data, dict) and 'error' in monthly_data):
            results[ticker] = monthly_data
    
    return jsonify({
        'tickers': tickers,
        'period': period,
        'comparison': results
    })

@app.route('/info/<string:ticker>', methods=['GET'])
def ticker_info(ticker):
    """Get basic information about a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return jsonify({
            'ticker': ticker,
            'name': info.get('longName', info.get('shortName', 'N/A')),
            'currency': info.get('currency', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'market': info.get('market', 'N/A')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)