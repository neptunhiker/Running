import datetime
import pandas as pd
import matplotlib.pyplot as plt


def ewm():

    load = [100 for i in range(80)]
    load[0] = 0
    load = [0 for i in range(80)]
    load[0] = 100

    df = pd.DataFrame(data=load)
    atl = df.ewm(alpha=2 / (1 + 7), adjust=False).mean()
    ctl = df.ewm(alpha=2 / (1 + 42), adjust=False).mean()
    tsb = ctl - atl

    fig, ax = plt.subplots()
    ax.plot(atl, color="red", label="ATL")
    ax.plot(ctl, color="blue", label="CTL")
    ax.plot(tsb, color="green", label="TSB")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ewm()



