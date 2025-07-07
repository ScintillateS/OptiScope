OptiScope – Portfolio Optimizer
OptiScope is a Python-based portfolio optimization tool using Monte Carlo simulations to visualize the efficient frontier and calculate optimal portfolio allocations.

🌐 Live App: Launch OptiScope
🚀 Features
✅ Multi-ticker portfolio optimization

✅ Monte Carlo simulation engine with customizable portfolio size

✅ Real-time risk-free rate tuning

✅ Live weight constraint sliders (min/max per asset)

✅ Portfolio mode switching: Max Sharpe Ratio or Minimum Variance

✅ Transaction cost simulation per rebalance

✅ Efficient frontier visualization with optimal portfolio marker

📸 Preview
(Insert a screenshot of your deployed app here)

📂 Project Structure
plaintext
Copy
Edit
OptiScope/
├── app.py                 # Streamlit web app
├── src/
│   ├── monte_carlo.py     # Monte Carlo simulation engine
├── logs/                  # Portfolio logs (optional for expansion)
├── requirements.txt       # Project dependencies
├── README.md
└── .gitignore
💻 Setup
Clone the repo:
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/OptiScope.git
cd OptiScope
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the app locally:
bash
Copy
Edit
python -m streamlit run app.py
✅ Future Improvements
Portfolio performance logging and daily summaries

Email performance reports

Additional optimization strategies (risk parity, sector constraints)

Public log viewing in Streamlit

