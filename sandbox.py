import datetime
import pandas as pd


import pandas as pd






if __name__ == '__main__':
    data = {'Garmin_load': [443, 108, 0, 235, 0, 132, 231, 100, 0, 100]}
    df = pd.DataFrame(data)

    import pandas as pd

    def calculate_rolling_ATL(df):
        df['rolling_ATL'] = df['Garmin_load'].rolling(window=7).apply(lambda x: x.ewm(alpha=0.9).mean())
        return df

    data = {'Garmin_load': [443, 108, 0, 235, 0, 132, 231]}
    df = pd.DataFrame(data)

    df = calculate_rolling_ATL(df)

    print(df)



