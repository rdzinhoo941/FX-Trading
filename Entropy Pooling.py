import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Airbus=yf.Ticker("Air.PA")


hist=Airbus.history(period="5d")
plt.plot(hist["Close"])
plt.show()


