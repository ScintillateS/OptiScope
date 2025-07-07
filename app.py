import streamlit as st
import matplotlib.pyplot as plt
from src.monte_carlo import fetch_price_data, monte_carlo_simulation

st.title("OptiScope – Portfolio Optimizer")

# Ticker input
tickers = st.text_input("Enter tickers separated by commas", "AAPL, MSFT, AMZN, TSLA")

# Number of portfolios
num_portfolios = st.slider("Number of portfolios to simulate", 1000, 10000, 5000)

risk_free_rate = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=5.0, value=0.01) / 100

if st.button("Run Optimization"):
    tickers = [ticker.strip().upper() for ticker in tickers.split(",")]
    data = fetch_price_data(tickers)

    results, weights_record = monte_carlo_simulation(data, num_portfolios=num_portfolios, risk_free_rate=risk_free_rate)

    max_sharpe_idx = results[2].argmax()
    max_sharpe_return = results[0, max_sharpe_idx]
    max_sharpe_volatility = results[1, max_sharpe_idx]
    optimal_weights = weights_record[max_sharpe_idx]

    st.subheader("Optimal Portfolio Weights")
    for ticker, weight in zip(tickers, optimal_weights):
        st.write(f"{ticker}: {weight:.2%}")

    # Plot Efficient Frontier
    st.subheader("Efficient Frontier")
    fig, ax = plt.subplots(figsize=(12, 6))

    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10)
    ax.scatter(max_sharpe_volatility, max_sharpe_return, c='red', marker='*', s=300, label='Optimal Portfolio')
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    fig.colorbar(scatter, label='Sharpe Ratio')

    st.pyplot(fig)
