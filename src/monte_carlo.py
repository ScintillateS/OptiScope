import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_price_data(tickers, start="2015-01-01"):
    df = yf.download(tickers, start=start, group_by="ticker", auto_adjust=True)

    if len(tickers) == 1:
        df = df["Close"]
        df = df.to_frame(name=tickers[0])
    else:
        prices = pd.DataFrame()
        for ticker in tickers:
            if (ticker, 'Close') in df.columns:
                prices[ticker] = df[ticker]['Close']
            else:
                raise ValueError(f"'Close' not found for {ticker}")
        df = prices

    return df

def simulate_portfolios(price_data, num_portfolios=5000, risk_free_rate=0.01):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    results = np.zeros((4, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(price_data.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio
        results[3, i] = i  # ID for later reference

    results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe Ratio", "ID"])
    return results_df, weights_record

def plot_portfolios(results_df):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(results_df["Volatility"], results_df["Return"],
                          c=results_df["Sharpe Ratio"], cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Sharpe Ratio")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Monte Carlo Portfolio Optimization")
    plt.grid(True)
    return plt
