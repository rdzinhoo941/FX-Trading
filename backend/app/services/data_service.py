"""
Data service – FX universe definitions & helpers.
>>> WHERE TO PLUG REAL MODELS: replace FX_PAIRS lookups with live market data feeds.
"""

FX_MAJORS = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
FX_MINORS = ["EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/NZD", "EUR/CHF", "CAD/JPY", "AUD/JPY"]
FX_EXOTICS = ["USD/TRY", "USD/ZAR", "USD/MXN", "USD/SGD", "EUR/NOK", "USD/HKD", "USD/THB"]

def get_pairs(universe: str) -> list[str]:
    if universe == "majors":
        return FX_MAJORS
    if universe == "minors":
        return FX_MINORS
    if universe == "exotics":
        return FX_EXOTICS
    # mix
    return FX_MAJORS[:4] + FX_MINORS[:2] + FX_EXOTICS[:1]
