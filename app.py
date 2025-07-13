import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.monte_carlo import fetch_price_data, run_monte_carlo_simulation

st.set_page_config(page_title="OptiScope – Portfolio Optimizer", layout="wide")

st.title("📈 OptiScope – Portfolio Optimizer")
st.markdown("Run Monte Carlo simulations on your stock portfolio to visualize risk, return, and the efficient frontier.")

# Ticker input and processing
tickers_input = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Parameters
num_portfolios = st.slider("Number of Simulations", 1000, 20000, 5000, step=1000)
risk_free_rate = st.slider("Risk-Free Rate (as decimal)", 0.0, 0.10, 0.01, step=0.005)

# Run simulation
if st.button("Run Optimization"):
    try:
        price_data = fetch_price_data(tickers, start="2020-01-01")
        results_df, weights_record = run_monte_carlo_simulation(price_data, num_portfolios, risk_free_rate)

        # Display results
        max_sharpe_idx = results_df["Sharpe"].idxmax()
        max_sharpe_port = results_df.loc[max_sharpe_idx]
        max_sharpe_weights = weights_record[int(max_sharpe_port["Index"])]

        st.subheader("Optimal Portfolio (Max Sharpe Ratio)")
        st.write(f"**Sharpe Ratio:** {max_sharpe_port['Sharpe']:.2f}")
        st.write(f"**Expected Return:** {max_sharpe_port['Return']:.2%}")
        st.write(f"**Volatility:** {max_sharpe_port['Volatility']:.2%}")

        weights_df = (
            {ticker: f"{weight:.2%}" for ticker, weight in zip(price_data.columns, max_sharpe_weights)}
        )
        st.write("**Weights:**")
        st.json(weights_df)

        # Efficient frontier plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x="Volatility", y="Return", hue="Sharpe", palette="viridis", edgecolor=None)
        plt.scatter(max_sharpe_port["Volatility"], max_sharpe_port["Return"], marker="*", color="red", s=200, label="Max Sharpe")
        plt.title("Efficient Frontier")
        plt.xlabel("Volatility (Std Dev)")
        plt.ylabel("Expected Return")
        plt.legend()
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"An error occurred: {e}")
