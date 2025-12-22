import pandas as pd
import numpy as np
from scipy.optimize import minimize

def getdata(client, symbol:str, interval:str, start_time:str = None, end_time:str = None) -> pd.DataFrame:
    historical_data = pd.DataFrame(client.get_historical_klines(symbol, interval, start_time, end_time))
    historical_data = historical_data.iloc[:, :6] # datetime\date, OHLC and volume in tokens (not USD)
    historical_data.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    historical_data = historical_data.set_index('date')
    historical_data.index = pd.to_datetime(historical_data.index, unit='ms')
    historical_data = historical_data.astype(float)
    return historical_data

def max_sharpe_weights(returns: pd.DaraFrame, max_w: float = 1.0):
    mu = returns.mean()
    cov = returns.cov()

    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret / port_vol)
    
    cons = [{'type':'eq', 'fun': lambda w: np.sum(w)-1.0}]
    bounds = [(0.0, max_w) for _ in range(n)]
    w0 = np.ones(n)/n

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    return res.x

def equiry_curve(returns: pd.DataFrame, w: np.ndarray):
    port_rets = returns.values @ w
    eq = pd.Series((1.0 + port_rets/100).cumprod(), index=returns.index)
    return eq, pd.Series(port_rets, index=returns.index)

def emama(data: pd.DataFrame) -> pd.DataFrame:
    long_window = 25
    short_window = 5
    
    mean_long = data['close_price'].ewm(span = long_window, adjust=False, min_periods=long_window).mean().rename('ema_25')
    mean_short = data['close_price'].ewm(span = short_window, adjust=False, min_periods=short_window).mean().rename('ema_5')
    
    delta = (mean_short - mean_long).dropna().rename('ema_diff')
    
    strategies = (np.where(np.sign(delta).diff()[1:].ne(0), 1, 0) * np.sign(delta)[1:]).astype(int).rename('ema_strategies')
    
    return pd.concat([mean_long, mean_short, delta, strategies], axis = 1)