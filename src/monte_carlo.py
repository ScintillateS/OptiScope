import yfinance as yf
import numpy as np
import pandas as pd

def fetch_price_data(tickers, start="2018-01-01"):
    # Download multi-ticker data with multi-index columns
    data = yf.download(tickers.split(','), start=start, group_by='ticker', auto_adjust=True)

    # Prepare Adj Close DataFrame
    adj_close = pd.DataFrame()
    for ticker in tickers.split(','):
        ticker = ticker.strip()
        try:
            adj_close[ticker] = data[ticker]['Adj Close']
        except (KeyError, TypeError):
            raise KeyError(f"'Adj Close' not found for ticker '{ticker}'.")

    return adj_close.dropna()

def monte_carlo_simulation(price_data, num_simulations=5000, risk_free_rate=0.01):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(price_data.columns)
    results = np.zeros((4, num_simulations))
    weights_record = []

    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio
        results[3, i] = i

    return results, weights_record, list(price_data.columns)
