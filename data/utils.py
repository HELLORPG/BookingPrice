import numpy as np

from torch.utils.data import Dataset, DataLoader


def strs_to_tokens(strs: list[str], token2str=None, str2token=None) -> (list[int], dict, dict):
    """
    将输入的字符串列表，转换成为tokens。
    :param token2str: 之前使用的token2str，或者不传入为空。
    :param str2token: similar to "token2str".
    :param strs: string list.
    :return: tokens, token2str, str2token
    """
    tokens = list()
    if token2str is None:
        token2str = dict()
        str2token = dict()

    for s in strs:
        if s not in str2token:  # 如果当前的str并没有被分配一个token数值
            new_token = len(str2token)
            str2token[s] = new_token
            token2str[new_token] = s
        # 一般情况下查询dict进行token分配
        tokens.append(str2token[s])

    return tokens, token2str, str2token


def get_statistic(values, mode: str):
    if isinstance(values, list):
        values = np.array(values, dtype=np.float32)
    if mode == "mean":
        return np.nanmean(values)
    elif mode == "std":
        return np.nanstd(values)
    elif mode == "min":
        return np.nanmin(values)
    elif mode == "max":
        return np.nanmax(values)
    elif mode == "median":
        return np.nanmean
    else:
        print("Mode %s is not supported." % mode)
        exit(-1)


def z_score(value, mean: float, std: float):
    return (value - mean) / std


def min_max(value, value_min: float, value_max: float):
    return (value - value_min) / (value_max - value_min)


if __name__ == '__main__':
    ss = ["hello", "where", "are", "you", "hello"]
    _, t2s, s2t = strs_to_tokens(ss)
    ss = ["I", "hello", "you"]
    print(strs_to_tokens(ss, t2s, s2t))

