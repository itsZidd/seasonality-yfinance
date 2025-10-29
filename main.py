from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================================
# 🔹 Seasonality Calculations
# ============================================================

def calculate_seasonality(ticker, period='10y'):
    """Monthly seasonality trends."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return None

        df['Returns'] = df['Close'].pct_change() * 100
        df['Month'] = df.index.month
        df['MonthName'] = df.index.strftime('%B')

        monthly_stats = (
            df.groupby('Month')
            .agg(avg_return=('Returns', 'mean'), std_dev=('Returns', 'std'), count=('Returns', 'count'))
            .reset_index()
        )

        win_rate = (
            df.groupby('Month')['Returns']
            .apply(lambda x: (x > 0).sum() / len(x) * 100)
            .reset_index(name='win_rate')
        )

        monthly_stats = monthly_stats.merge(win_rate, on='Month')
        monthly_stats['MonthName'] = monthly_stats['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
        monthly_stats = monthly_stats.sort_values('Month')

        return monthly_stats.to_dict('records')
    except Exception as e:
        return {'error': str(e)}


def calculate_quarterly_seasonality(ticker, period='10y'):
    """Quarterly seasonality trends."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None

        df['Returns'] = df['Close'].pct_change() * 100
        df['Quarter'] = df.index.quarter

        quarterly_stats = (
            df.groupby('Quarter')
            .agg(avg_return=('Returns', 'mean'), std_dev=('Returns', 'std'), count=('Returns', 'count'))
            .reset_index()
        )

        quarterly_stats['QuarterName'] = quarterly_stats['Quarter'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
        return quarterly_stats.to_dict('records')
    except Exception as e:
        return {'error': str(e)}


def calculate_ytd_trend(ticker):
    """Year-to-date cumulative trend."""
    try:
        stock = yf.Ticker(ticker)
        start = datetime(datetime.now().year, 1, 1)
        end = datetime.now()
        df = stock.history(start=start, end=end)

        if df.empty:
            return None

        df['Cumulative'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        df['Week'] = df.index.isocalendar().week
        return df[['Week', 'Cumulative']].to_dict('records')
    except Exception as e:
        return {'error': str(e)}


def calculate_weekly_seasonality(ticker, period='10y'):
    """Cumulative weekly seasonality (%)."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return None

        df['Return'] = df['Close'].pct_change()
        df['Week'] = df.index.isocalendar().week
        df['Year'] = df.index.isocalendar().year

        # Weekly aggregated returns
        weekly = df.groupby(['Year', 'Week'])['Return'].sum().reset_index()

        # Calculate cumulative per year
        weekly['Cumulative'] = (1 + weekly['Return']).groupby(weekly['Year']).cumprod() - 1

        # Average across years
        avg = (
            weekly.groupby('Week')['Cumulative']
            .mean()
            .reset_index()
            .rename(columns={'Cumulative': 'avg_10y'})
        )

        # Get YTD
        current_year = datetime.now().year
        ytd = weekly[weekly['Year'] == current_year][['Week', 'Cumulative']].rename(columns={'Cumulative': 'ytd'})

        merged = pd.merge(avg, ytd, on='Week', how='outer').fillna(method='ffill')
        merged = merged[merged['Week'] <= 53]
        merged['avg_10y'] = merged['avg_10y'] * 100
        merged['ytd'] = merged['ytd'] * 100

        return merged.to_dict('records')

    except Exception as e:
        return {'error': str(e)}


# ============================================================
# 🔹 API Endpoints
# ============================================================

@app.route('/')
def home():
    return jsonify({
        'message': 'YFinance Seasonality API',
        'endpoints': {
            '/seasonality/monthly': 'Monthly average trends',
            '/seasonality/quarterly': 'Quarterly average trends',
            '/seasonality/weekly': 'Weekly cumulative trends',
            '/seasonality/compare': 'Compare multiple indices'
        },
        'example': '/seasonality/weekly?ticker=^GSPC&period=10y'
    })


@app.route('/seasonality/weekly', methods=['GET'])
def weekly_seasonality():
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '10y')

    data = calculate_weekly_seasonality(ticker, period)
    if data is None or (isinstance(data, dict) and 'error' in data):
        return jsonify({'error': 'No data found'}), 404

    return jsonify({
        'ticker': ticker,
        'period': period,
        'analysis_type': 'weekly',
        'data': data
    })


@app.route('/seasonality/monthly', methods=['GET'])
def monthly_seasonality():
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '10y')

    avg_result = calculate_seasonality(ticker, period)
    ytd_result = calculate_ytd_trend(ticker)

    if avg_result is None or isinstance(avg_result, dict) and 'error' in avg_result:
        return jsonify({'error': 'No historical data found'}), 404

    return jsonify({
        'ticker': ticker,
        'period': period,
        'analysis_type': 'monthly',
        'avg_10y': avg_result,
        'ytd_trend': ytd_result or []
    })


@app.route('/seasonality/quarterly', methods=['GET'])
def quarterly_seasonality():
    ticker = request.args.get('ticker', '^GSPC')
    period = request.args.get('period', '10y')

    result = calculate_quarterly_seasonality(ticker, period)
    if result is None:
        return jsonify({'error': 'No data found for ticker'}), 404

    return jsonify({
        'ticker': ticker,
        'period': period,
        'analysis_type': 'quarterly',
        'data': result
    })


@app.route('/seasonality/compare', methods=['GET'])
def compare_seasonality():
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


# ============================================================
# 🔹 Run Flask (local only)
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
