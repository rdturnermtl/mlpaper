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
    """Build a lad dataframe from dataframe where the rows are ordered time
    indices for a time series data set. This is useful for autoregressive
    models.

    Parameters
    ----------
    df : pandas dataframe
        Orginal dataset we want to build lag data set from.
    n_lags : int
        Number of lags. ``n_lags=1`` means only the original data set. Must be
        >= 1.
    stride : int
        Stride of the lags. For instance, ``stride=2`` means only even lags.
    features : array_like
        List of columns to include in the lags data. All columns are retained
        for lag 0. For data frames containing features and targets, the
        features (inputs)  can be placed in `features` so the targets (outputs)
        are only present for lag 0. If None, use all columns.

    Returns
    -------
    df : pandas dataframe
        New data frame where lags data frames have been concat'ed tegether.
        The columns are a new hierarchical index with the lag at the lowest
        level.

    Examples
    --------
    >>> data=np.random.choice(10,size=(4,3))
    >>> df=pd.DataFrame(data=data,columns=['a','b','c'])
    >>> ds.build_lag_df(df,3,features=['a','b'])
              a  b  c   a   b   a   b
         lag L0 L0 L0  L1  L1  L2  L2
         0    2  2  2 NaN NaN NaN NaN
         1    2  9  4   2   2 NaN NaN
         2    8  4  0   2   9   2   2
         3    3  5  6   8   4   2   9
    """
    df_sub = df if features is None else df[list(features)]  # Take all if None
    D = {(SFT_FMT % nn): df_sub.shift(stride * nn) for nn in xrange(1, n_lags)}
    D[SFT_FMT % 0] = df

    df = pd.concat(D, axis=1, names=['lag'])
    # Re-order the levels so there are the same as before but lag at end
    df = df.reorder_levels(range(1, len(df.columns.names)) + [0], axis=1)
    return df


def index_to_series(index):
    """Make a pandas series from a pandas index with the value equal to index.

    Parameters
    ----------
    index : pandas index
        Index to make series from.

    Returns
    -------
    s : pandas series
        Series where ``s[idx] = idx``.

    Examples
    --------
    >>> index_to_series(pd.Index([1,5,7]))
    1    1
    5    5
    7    7
    dtype: int64
    """
    return pd.Series(index=index, data=index)


def rand_subset(x, frac):
    """Take random subset of array `x` with a certain fraction. Rounds number
    of elements up to next integer when exact fraction is not possible.

    Parameters
    ----------
    x : array_like
        List that we want a subset of.
    frac : float
        Fraction of `x` elements we want to keep in subset. Must be in [0,1].

    Returns
    -------
    L : 1d np array
        Array that is subset.
    """
    assert(0.0 <= frac and frac <= 1.0)

    N = int(np.ceil(frac * len(x)))
    assert(0 <= N and N <= len(x))
    L = np.random.choice(x, N, replace=False)
    assert(len(L) >= len(x) * frac)
    assert(len(L) - 1 < len(x) * frac)
    return L


def rand_mask(N, frac):
    """Make a random binary mask with a certain fraction. Rounds number of
    elements up to next integer when exact fraction is not possible.

    Parameters
    ----------
    N : int
        Length of mask.
    frac : float
        Fraction of elements we want to be True. Must be in [0,1].

    Returns
    -------
    L : 1d np array of type bool
        Random binary mask.
    """
    # rand_subset() checks that frac in range
    pos = rand_subset(xrange(N), frac)
    mask = np.zeros(N, dtype=bool)
    mask[pos] = True
    assert(np.sum(mask) >= N * frac)
    assert(np.sum(mask) - 1 < N * frac)
    return mask


def random_split_series(S, frac, assume_sorted=False, assume_unique=False):
    """Create a binary mask to split a series into training/test based on a
    random split based on values of series. That is, elements with the same
    value in the series always get grouped into both train or both test.

    Parameters
    ----------
    S : pandas series
        Series whose index will be used for binary mask. Random splitting is
        based on a random parititioning of the series *values*.
    frac : float
        Fraction of elements we want to be True. Must be in [0,1].
    assume_sorted : bool
        If True, assume series is already sorted based on values. This can be
        used for computational speedups.
    assume_unique : bool
        If True, assume all values in series are unique. This can be
        used for computational speedups.

    Returns
    -------
    train_curr : pandas series with values of type bool
        Random binary mask with index matching `S`.
    """
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
    """Create a binary mask to split a series into training/test based on a
    ordered split based on values of series. That is, indices with a lower
    value get put in train and the rest go in test.

    Parameters
    ----------
    S : pandas series
        Series whose index will be used for binary mask. The ordered split is
        based on the series *values*.
    frac : float
        Fraction of elements we want to be True. Must be in [0,1].
    assume_sorted : bool
        If True, assume series is already sorted based on values. This can be
        used for computational speedups.
    assume_unique : bool
        If True, assume all values in series are unique. This can be
        used for computational speedups.

    Returns
    -------
    train_curr : pandas series with values of type bool
        Binary mask with index matching `S`.
    """
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
    """Create a binary mask to split a series into training/test based on a
    linear split based on values of series. That is, the train/test divide is
    based on a point that is a linear interpolation between lowest value and
    highest value in the series.

    Parameters
    ----------
    S : pandas series
        Series whose index will be used for binary mask. The linear split is
        based on the series *values*.
    frac : float
        Fraction of region be between series min and series max we want to be
        True. Must be in [0,1].
    assume_sorted : bool
        If True, assume series is already sorted based on values. This can be
        used for computational speedups.
    assume_unique : bool
        If True, assume all values in series are unique. This can be
        used for computational speedups.

    Returns
    -------
    train_curr : pandas series with values of type bool
        Binary mask with index matching `S`.
    """
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
    """Split a pandas data frame based on criteria across multiple columns.

    A seperate train test split is done for each column specified as a split
    column in `splits`. A row is added to the final training set, only if it
    is placed in training by every column splits. Likewise, A row is added to
    the final test set, only if it is placed in test by every column splits.
    All other rows are placed in the unused data points DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame we wish to split into training and test chunks
    splits : dict of object to ({RANDOM, ORDRED, LINEAR}, float)
        Dictionary explaining how to do the split. The keys of the `splits` are
        the columns in `df` we will base the split on. The constant INDEX can
        be used to symbolize that the index is the desired column.
        Each value is a tuple with (split type, fraction for training). The 
        split type can be either: random, ordered, or linear. The fraction for
        training must be in [0,1]. Fraction of region be between series min and
        series max we want to be True. The Fraction must be in [0,1]. If
        `splits` is omitted, the default is to perform a 80-20 random split
        based on the index.
    assume_sorted : array_like of str
        Columns that we can assume are alreay sorted by value. This can be
        used for computational speedups.
    assume_unique : array_like of str
        Columns that we can assume have unique values. This can be used for
        computational speedups.

    Returns
    -------
    df_train : pandas DataFrame
        Subset of `df` placed in training set.
    df_test : pandas DataFrame
        Subset of `df` placed in test set.
    df_unused : pandas DataFrame
        Subset of `df` not in training or test. This will be empty if only a
        single column is ued in `splits`.
    """
    assert(len(splits) > 0)
    assert(len(df) > 0)  # It is not hard to get working with len 0, but why.
    assert(INDEX not in df)  # None repr for INDEX, col name is reserved here.

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

        # TODO assert dtype bool before ~ operation
        train_series &= train_curr
        test_series &= ~train_curr
    assert(not (train_series & test_series).any())

    df_train, df_test = df[train_series], df[test_series]
    df_unused = df[~(train_series | test_series)]
    assert(len(df_train) + len(df_test) + len(df_unused) == len(df))
    return df_train, df_test, df_unused
