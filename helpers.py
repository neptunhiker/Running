import matplotlib.pyplot as plt
import pandas as pd
from typing import List


def weighted_sum(lst: List[int], b: float, n: int) -> float:
    """
    This function takes a list of integers, a value for b and an integer n, and returns the weighted sum of the first
    n elements of the list based on the specified weighting.
    The first element shall be weighted with 100 %, the second element with b**1, the third element with b**2 and so on.
    :param lst: a list of integers
    :param b : weigthing factor
    :param n: number of periods for which the weighted sum shall be calculated
    :return: float
    """
    w_sum = 0
    for i, num in enumerate(lst[:n]):
        weight = b ** i
        w_sum += num * weight
    return w_sum


def ewm(lst: List[int], alpha: float) -> pd.Series:
    """
    Calculate an exponentially weighted moving average for training loads
    :param lst: list of training loads
    :param alpha: smoothing factor
    :return: list of exponentially weighted moving averages
    """

    ewm_load = []
    for i, load in enumerate(lst):
        if i == 0:
            ewm_load.append(alpha * load)
        else:
            ewm_load.append(alpha * load + (1-alpha) * ewm_load[i-1])

    # ewm_load.reverse()

    return pd.Series(ewm_load)


def atl(loads: pd.Series) -> pd.Series:
    """
    Calculate the Acute Training Load (ATL) as exponentially weighted moving average
    :param loads: a pd.Series of training loads
    :return: pd.Series of ATLs
    """
    return ewm(lst=loads.tolist(), alpha=2 / (1 + 7))


def ctl(loads: pd.Series) -> pd.Series:
    """
    Calculate the Chronic Training Load (CTL) as exponentially weighted moving average
    :param loads: a pd.Series of training loads
    :return: pd.Series of CTLs
    """
    return ewm(lst=loads.tolist(), alpha=2 / (1 + 42))


if __name__ == '__main__':
    # loads = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 152, 0, 139, 0, 101, 159, 173]
    loads = [100 for i in range(100)]
    loads[0] = 0
    # loads = [0 for i in range(100)]
    # loads[1] = 100
    atls = pd.Series(ewm(lst=loads, alpha=0.25))
    ctls = pd.Series(ewm(lst=loads, alpha=2 / (1 + 42)))
    tsb = ctls - atls

    fig, ax = plt.subplots()
    ax.plot(atls, color="red", label="ATL")
    ax.plot(ctls, color="blue", label="CTL")
    ax.plot(tsb, color="green", label="TSB")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(atls)
    print(ctls)