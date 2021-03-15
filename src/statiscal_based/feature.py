import numpy as np
import pandas as pd
import itertools

#util function
def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

## feature extraction function
def basic_info(x):
    return [min(x), max(x), np.mean(x), np.median(x), np.std(x)]

def startend(x):
    # 是否发生concept drift
    return [x[-1]-x[0]]

def absolute_sum_of_changes(x):
    # 波动情况，返回序列x的连续变化的绝对值之和
    diffsum = np.sum(np.abs(np.diff(x)))
    return [np.mean(diffsum), np.min(diffsum), np.max(diffsum)]

def distribution(x):
    # 分布情况,峰度，偏度
    freq, _ = np.histogram(x, 10)
    return freq.tolist()+[pd.Series(x).skew(), pd.Series(x).kurt()]

def count_with_mean(x):
    # 比均值大/小的点的个数
    m = np.mean(x)
    return [np.where(x > m)[0].size, np.where(x < m)[0].size]

def location(x):
    # 最大值最小值的相对位置
    first_location_of_maximum = np.argmax(x) / len(x) if len(x) > 0 else np.NaN
    last_location_of_maximum = 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN
    first_location_of_minimum = np.argmin(x) / len(x) if len(x) > 0 else np.NaN
    last_location_of_minimum = 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN
    return [first_location_of_maximum,last_location_of_maximum,first_location_of_minimum,last_location_of_minimum]
# def longest_strike_below_mean(x):
    
#     # Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x
#     if not isinstance(x, (np.ndarray, pd.Series)):
#         x = np.asarray(x)
#     return [np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0]


def cid_ce(x, normalize=False):
    #This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    #valleys etc.). 时间序列复杂度刻画，来自tsfresh
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s != 0:
            x = (x - np.mean(x)) / s
        else:
            return 0.0

    x = np.diff(x)
    return [np.sqrt(np.dot(x, x))]