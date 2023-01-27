import datetime
import pandas as pd


def calculate_rolling_CTL(dataframe, window=3, alpha=2.0):
    dataframe = dataframe.sort_values(by='date')
    dataframe['weight'] = (alpha ** pd.Series(range(len(dataframe)))).tolist()
    dataframe['weighted_trimp'] = dataframe['Runalyze_trimp'] * dataframe['weight']
    CTL = dataframe['weighted_trimp'].rolling(window=window,center=False).sum() / dataframe['weight'].rolling(window=window,center=False).sum()
    print(dataframe)
    return CTL



if __name__ == '__main__':

    dataframe = pd.DataFrame(data=[100, 200, 0, 0, 100, 150, 50, 0, 100])
    dataframe.set_index(pd.date_range(datetime.date(2022, 1, 22), periods=len(dataframe), freq="D"))
    dataframe.reset_index(inplace=True)
    dataframe.columns = ["date", "Runalyze_trimp"]
    CTL = calculate_rolling_CTL(dataframe)
    print(CTL)
