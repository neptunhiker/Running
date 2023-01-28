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
