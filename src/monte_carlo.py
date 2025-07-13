import yfinance as yf
import numpy as np
import pandas as pd

def fetch_price_data(tickers, start="2018-01-01"):
    data = yf.download(tickers, start=start, group_by='ticker', auto_adjust=False)
    
    if isinstance(tickers, str) or len(tickers.split(',')) == 1:
        # Single ticker
        return data['Adj Close']
    else:
        adj_close = pd.DataFrame()
        for ticker in tickers.split(','):
            adj_close[ticker.strip()] = data[ticker.strip()]['Adj Close']
        return adj_close

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
        results[3, i] = i  # Store index for tracking

    return results, weights_record, list(price_data.columns)
