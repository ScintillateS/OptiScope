import numpy as np
import pandas as pd
import yfinance as yf

def fetch_price_data(tickers, start="2018-01-01"):
    data = yf.download(tickers, start=start)['Adj Close']
    return data.dropna()

def monte_carlo_simulation(data, num_portfolios=5000, risk_free_rate=0.01):
    log_returns = np.log(data / data.shift(1)).dropna()
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    results = np.zeros((4, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(data.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio
        results[3, i] = i

    return results, weights_record
