import numpy as np
import pandas as pd
import yfinance as yf

def fetch_price_data(tickers, start="2020-01-01"):
    data = yf.download(tickers, start=start)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            return data['Adj Close']
        else:
            raise KeyError("'Adj Close' not found in multi-index columns.")
    elif 'Adj Close' in data.columns:
        return data[['Adj Close']]
    else:
        raise KeyError(f"'Adj Close' not found in downloaded data columns: {data.columns}")

def run_monte_carlo_simulation(data, num_portfolios=5000, risk_free_rate=0.01):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    results = np.zeros((4, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(data.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio
        results[3, i] = i

    results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe", "Index"])
    return results_df, weights_record
