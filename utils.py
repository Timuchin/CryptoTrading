import pandas as pd

def getdata(client, symbol:str, interval:str, start_time:str = None, end_time:str = None) -> pd.DataFrame:
    historical_data = pd.DataFrame(client.get_historical_klines(symbol, interval, start_time, end_time))
    historical_data = historical_data.iloc[:, :6] # Время открытия свечи, цена открытия, закрытия, объем
    historical_data.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    historical_data = historical_data.set_index('date')
    historical_data.index = pd.to_datetime(historical_data.index, unit='ms')
    historical_data = historical_data.astype(float)
    return historical_data