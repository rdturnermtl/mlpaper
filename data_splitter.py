# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pandas as pd

RANDOM = 'random'
ORDRED = 'ordered'
LINEAR = 'linear'

SFT_FMT = 'L%d'
INDEX = None  # Dummy variable to represent index of dataframe
DEFAULT_SPLIT = {INDEX: (RANDOM, 0.8)}  # The ML standard for some reason


def build_lag_df(df, n_lags, stride=1, features=None):
    df_sub = df if features is None else df[features]  # Take all if None
    D = {(SFT_FMT % nn): df_sub.shift(stride * nn) for nn in xrange(1, n_lags)}
    D[SFT_FMT % 0] = df

    df = pd.concat(D, axis=1, names=['lag'])
    # Re-order the levels so there are the same as before but lag at end
    df = df.reorder_levels(range(1, len(df.columns.names)) + [0], axis=1)
    return df


def index_to_series(index):
    return pd.Series(index=index, data=index)


def rand_subset(x, frac):
    assert(0.0 <= frac and frac <= 1.0)

    N = int(np.ceil(frac * len(x)))
    assert(0 <= N and N <= len(x))
    L = np.random.choice(x, N, replace=False)
    assert(len(L) >= len(x) * frac)
    assert(len(L) - 1 < len(x) * frac)
    return L


def rand_mask(N, frac):
    # rand_subset() checks that frac in range
    pos = rand_subset(xrange(N), frac)
    mask = np.zeros(N, dtype=bool)
    mask[pos] = True
    assert(np.sum(mask) >= N * frac)
    assert(np.sum(mask) - 1 < N * frac)
    return mask


def random_split_series(S, frac, assume_sorted=False, assume_unique=False):
    assert(not S.isnull().any())  # Ordering/comparing NaNs ambiguous
    # Frac range checking taken care of by sub-routines

    if assume_unique:
        train_curr = pd.Series(index=S.index, data=rand_mask(len(S), frac))
    else:
        # Note: pd.unique() does not sort, this is required to maintain
        # identical result to assume_unique case (w/ same random seed).
        train_cases = rand_subset(S.unique(), frac)
        train_curr = S.isin(train_cases)
    return train_curr


def ordered_split_series(S, frac, assume_sorted=False, assume_unique=False):
    assert(not S.isnull().any())  # Ordering/comparing NaNs ambiguous
    assert(0.0 <= frac and frac <= 1.0)

    # Get all cases in sorted order
    if assume_sorted and assume_unique:
        all_cases = S.values
    elif assume_unique:  # but not sorted
        all_cases = np.sort(S.values)
    else:
        all_cases = np.unique(S.values)

    idx = min(int(frac * len(all_cases)), len(all_cases) - 1)
    assert(0 <= idx)  # Should never happen due to frac check earlier
    pivotal_case = all_cases[idx]
    # Check we rounded to err just on side of putting more data in train
    assert(np.mean(all_cases <= pivotal_case) >= frac)
    assert(idx == 0 or np.mean(all_cases <= all_cases[idx - 1]) < frac)
    train_curr = (S <= pivotal_case)
    return train_curr


def linear_split_series(S, frac, assume_sorted=False, assume_unique=False):
    assert(not S.isnull().any())  # Ordering/comparing NaNs ambiguous
    assert(0.0 <= frac and frac <= 1.0)

    if assume_sorted:
        start, end = S.values[0], S.values[-1]
    else:
        start, end = np.min(S.values), np.max(S.values)
    assert(np.isfinite(start) and np.isfinite(end))

    pivotal_point = (1.0 - frac) * start + frac * end
    # For numerics:
    pivotal_point = np.maximum(start, np.minimum(pivotal_point, end))
    assert(start <= pivotal_point and pivotal_point <= end)

    train_curr = (S <= pivotal_point)
    return train_curr

SPLITTER_LIB = {RANDOM: random_split_series,
                ORDRED: ordered_split_series,
                LINEAR: linear_split_series}


def split_df(df, splits=DEFAULT_SPLIT, assume_unique=(), assume_sorted=()):
    assert(len(splits) > 0)
    assert(len(df) > 0)  # It is not hard to get working with len 0, but why.
    assert(INDEX not in df) # None repr for INDEX as col name is reserved here.

    train_series = pd.Series(index=df.index, data=True)
    test_series = pd.Series(index=df.index, data=True)
    for feature, how in splits.iteritems():
        split_type, frac = how
        # Could throw exception for unknown splitter type
        splitter_f = SPLITTER_LIB[split_type]

        S = index_to_series(df.index) if feature is INDEX else df[feature]
        train_curr = splitter_f(S, frac,
                                assume_sorted=(feature in assume_sorted),
                                assume_unique=(feature in assume_unique))

        train_series &= train_curr
        test_series &= ~train_curr
    assert(not (train_series & test_series).any())

    df_train, df_test = df[train_series], df[test_series]
    df_unused = df[~(train_series | test_series)]
    assert(len(df_train) + len(df_test) + len(df_unused) == len(df))
    return df_train, df_test, df_unused
