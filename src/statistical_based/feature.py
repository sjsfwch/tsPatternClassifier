import numpy as np
import pandas as pd
import itertools
from tsfresh import extract_features, select_features

# util function


def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group))
               for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


# feature extraction function
def basic_info(x):
    # 基本的统计特征  5个
    return [min(x), max(x), np.mean(x), np.median(x), np.std(x), np.quantile(x, 0.90),
            np.quantile(x, 0.95)]


def startend(x):
    # 是否发生concept drift  1个
    return [x[-1]-x[0]]


def absolute_sum_of_changes(x):
    # 波动情况，返回序列x的连续变化的绝对值之和 5个
    diffsum = np.sum(np.abs(np.diff(x)))
    # number_crossing_m
    positive = x > np.mean(x)
    num_cross = np.where(np.diff(positive))[0].size
    return [diffsum, np.mean(diffsum), np.min(diffsum), np.max(diffsum), num_cross]


def distribution(x):
    # 分布情况,峰度，偏度 12个
    freq, _ = np.histogram(x, 10)
    return freq.tolist()+[pd.Series(x).skew(), pd.Series(x).kurt()]


def count_with_mean(x):
    # 比均值大/小的点的个数 2个
    m = np.mean(x)
    return [np.where(x > m)[0].size, np.where(x < m)[0].size]


def location(x):
    # 最大值最小值的相对位置  4个
    first_location_of_maximum = np.argmax(x) / len(x) if len(x) > 0 else np.NaN
    last_location_of_maximum = 1.0 - \
        np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN
    first_location_of_minimum = np.argmin(x) / len(x) if len(x) > 0 else np.NaN
    last_location_of_minimum = 1.0 - \
        np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN
    return [first_location_of_maximum, last_location_of_maximum, first_location_of_minimum, last_location_of_minimum]


def transform_min(x):
    index_min, index_max = np.argmin(x), np.argmax(x)

    def impute_zero(value, flag):
        if len(value) < 3 and flag == 'start':
            value = [0]*(3-len(value))+value
        if len(value) < 3 and flag == 'end':
            value = value + [0]*(3-len(value))
        return value
    before_min = np.diff(x[index_min-3:index_min]).tolist()
    before_max = np.diff(x[index_max-3:index_max]).tolist()
    after_min = np.diff(x[index_min:index_min+3]).tolist()
    after_max = np.diff(x[index_max:index_max+3]).tolist()

    return impute_zero(before_min, 'start') + impute_zero(after_min, 'end') +\
        impute_zero(before_max, 'start') + impute_zero(after_max, 'end')

# def longest_strike_below_mean(x):

#     # Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x
#     if not isinstance(x, (np.ndarray, pd.Series)):
#         x = np.asarray(x)
#     return [np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0]


def cid_ce(x, normalize=False):
    # This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    # valleys etc.). 时间序列复杂度刻画，来自tsfresh 1个
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


def use_tsfresh(x):
    x_df = pd.DataFrame(
        data={'value': x, 'time': range(len(x)), 'id': range(len(x))})
    tsfresh_feature = extract_features(
        x_df, column_id='id', column_sort='time')
    return tsfresh_feature
