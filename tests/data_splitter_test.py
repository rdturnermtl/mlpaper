# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import division, print_function

from builtins import range
from collections import OrderedDict
from string import ascii_letters

import numpy as np
import pandas as pd

import benchmark_tools.data_splitter as ds
from benchmark_tools.test_constants import MC_REPEATS_LARGE


def unif2():
    x = np.random.choice([0.0, np.random.rand(), 1.0], p=[0.05, 0.9, 0.05])
    return x


def unif_subset(L, size=None):
    if len(L) == 0:
        assert size is None or size == 0
        return list(L)
    size = np.random.randint(len(L) + 1) if size is None else size
    S = list(np.random.choice(L, size, replace=False))
    return S


def vec_rnd(N, req_sorted=False, req_unique=False):
    x = np.concatenate((np.random.randn(N), [-np.inf, np.inf]))
    x = np.random.choice(x, N, replace=(not req_unique))

    if req_sorted:
        x.sort()
    return x[:, None]


def test_df(high_M=6, high_N=6):
    M = np.random.randint(low=1, high=high_M)
    N = np.random.randint(low=1, high=high_N)
    col = unif_subset(list(ascii_letters), size=N)
    idx = range(M)  # Will over-write anyway

    s_list, u_list = unif_subset(col), unif_subset(col)
    dat = [vec_rnd(M, req_sorted=(c in s_list), req_unique=(c in u_list)) for c in col]
    dat = np.concatenate(dat, axis=1)

    idx_name = col[0]
    df = pd.DataFrame(data=dat, index=idx, columns=col, dtype=np.float)
    df.set_index(idx_name, drop=True, inplace=True)

    if idx_name in s_list:
        s_list.remove(idx_name)
        s_list.append(ds.INDEX)

    if idx_name in u_list:
        u_list.remove(idx_name)
        u_list.append(ds.INDEX)

    return df, s_list, u_list


def test_splitter(seed0=10, seed1=100):
    np.random.seed(seed0)
    df, s_list, u_list = test_df()

    col = list(df.columns)
    kk = unif_subset(col)
    splits = {k: (np.random.choice(list(ds.SPLITTER_LIB.keys())), unif2()) for k in kk}

    for k in splits:
        split_type, _ = splits[k]
        # Need to get rid of inf's in linear
        if split_type == ds.LINEAR:
            df[k] = np.nan_to_num(df[k].values)

    # To ensure reproducability across 2 runs
    splits2 = OrderedDict(splits)
    splits = OrderedDict(splits)

    if len(splits) == 0 or np.random.rand() <= 0.2:
        # No LINEAR since then we need to fix the infs
        splits[ds.INDEX] = (np.random.choice([ds.RANDOM, ds.ORDRED]), unif2())
    if ds.INDEX in splits:
        splits2[df.index.name] = splits[ds.INDEX]

    np.random.seed(seed1)
    df_train, df_test, df_unused = ds.split_df(df, splits, assume_unique=u_list)

    # Make index a normal column and see if split gives same result
    df.reset_index(drop=False, inplace=True)
    np.random.seed(seed1)
    df_train2, df_test2, df_unused2 = ds.split_df(df, splits2)

    df_train.reset_index(drop=False, inplace=True)
    df_test.reset_index(drop=False, inplace=True)
    df_unused.reset_index(drop=False, inplace=True)

    df_train2.reset_index(drop=True, inplace=True)
    df_test2.reset_index(drop=True, inplace=True)
    df_unused2.reset_index(drop=True, inplace=True)

    assert df_train.equals(df_train2)
    assert df_test.equals(df_test2)
    assert df_unused.equals(df_unused2)


if __name__ == "__main__":
    np.random.seed(635463)

    runs = MC_REPEATS_LARGE
    seeds = np.random.randint(low=0, high=10 ** 6, size=(runs, 2))
    for rr in range(runs):
        test_splitter(seeds[rr, 0], seeds[rr, 1])
    print("passed")
