from app.services.fx_wrapper import run_fx_framework_allocation

rows, kpi = run_fx_framework_allocation(
    pairs=["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"],
    initial_capital=100000,
    risk_aversion="medium",
    rebalance_mode="weekly",
)

print("SUCCESS")
print(rows[:2])
print(kpi)