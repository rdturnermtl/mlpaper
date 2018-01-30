# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, absolute_import, division
from builtins import range
import decimal
from sys import version_info
import warnings
import numpy as np
import pandas as pd
from benchmark_tools.constants import (METHOD, METRIC, STAT,
                                       STD_STATS, FMT_STATS)
from benchmark_tools.constants import MEAN_COL, ERR_COL, PVAL_COL, EST_COL
from benchmark_tools.constants import (GEN_FMT, ABOVE_FMT, BELOW_FMT,
                                       _PREFIX, _PREFIX_TEX)

NAN_STR = str(np.nan)  # Our string rep of NaN
# Constants of Decimal type
D_INF = decimal.Decimal('Infinity')
D_NINF = decimal.Decimal('-Infinity')


def remove_chars_py2(x_str, del_chars):
    x_str = x_str.translate(None, del_chars)
    return x_str


def remove_chars_py3(x_str, del_chars):
    translator = str.maketrans('', '', del_chars)
    x_str = x_str.translate(translator)
    return x_str

# TODO figure out how to make some routine work in py2 and 3, move to util:
# The py3 versions seems to work in Py2 after using
# from builtins import str
# if x_str is unicode => need to make sure we use unicode consistently
remove_chars = remove_chars_py3 if version_info[0] >= 3 else remove_chars_py2

# ============================================================================
# General utils
# ============================================================================


def all_same(L):
    '''Check if all elements in list are equal.

    Parameters
    ----------
    L : array-like, shape (n,)
        List of objects of any type.

    Returns
    -------
    y : bool
        True if all elements are equal.
    '''
    y = len(L) == 0 or all(x == L[0] for x in L)
    return y


def floor_mod(x, mod):
    '''Do floor in base mod instead of to nearest integer.

    Parameters
    ----------
    x : int
        Number to floor.
    mod : int
        Positive number (`x` >= 1) to use as modulus.

    Returns
    -------
    y : int
        Largest number ``y <= x`` such that ``y % mod = 0``.
    '''
    y = (x // mod) * mod
    return y


def ceil_mod(x, mod):
    '''Do ceil in base mod instead of to nearest integer.

    Parameters
    ----------
    x : int
        Number to ceil.
    mod : int
        Positive number (`x` >= 1) to use as modulus.

    Returns
    -------
    y : int
        Smallest number ``y >= x`` such that ``y % mod = 0``.
    '''
    y = floor_mod(x, -mod)
    return y


def str_print_len(x_str):
    '''Estimated width of formatted number of string when *not* displayed using
    a fixed width font. This is the number of characters not including ``.``
    and ``,`` because they are assumed to be of negligible width.

    Parameters
    ----------
    x_str : str
        Already formatted number string.

    Returns
    -------
    str_len : int
        Length of string without negligible width characters ``.`` and ``,``.
    '''
    str_len = len(remove_chars(x_str, ',.'))
    return str_len


def ensure_tuple_of_ints(L):
    '''This could possibly be done more efficiently with `tolist` if L is
    np or pd array, but will stick with this simple solution for now.
    '''
    T = tuple([int(mm) for mm in L])
    return T

# ============================================================================
# Decimal utils
# ============================================================================


def decimal_all_finite(x_dec_list):
    '''Check if all elements in list of decimals are finite.

    Parameters
    ----------
    x_dec_list : iterable of Decimal
        List of decimal objects.

    Returns
    -------
    y : bool
        True if all elements are finite.
    '''
    y = all(x.is_finite() for x in x_dec_list)
    return y


def decimal_from_tuple(signed, digits, expo):
    '''Build `Decimal` objects from components of decimal tuple.

    Parameters
    ----------
    signed : bool
        True for negative values.
    digits : iterable of ints
        digits of value each in [0,10).
    expo : int or {'F', 'n', 'N'}
        exponent of decimal.

    Returns
    -------
    y : Decimal
        corresponding decimal object.
    '''
    # Get everything in correct type because the Py3 decimal package is anal
    signed = int(signed)
    digits = ensure_tuple_of_ints(digits)
    expo = expo if expo in ('F', 'n', 'N') else int(expo)

    y = decimal.Decimal(decimal.DecimalTuple(signed, digits, expo))
    return y


def as_tuple_chk(x_dec):
    '''Convert `Decimal` to `DecimalTuple` and check finite.

    Parameters
    ----------
    x_dec : Decimal
        Input value in decimal.

    Returns
    -------
    x_tup : DecimalTuple
        Input converted to `DecimalTuple`.
    '''
    if not x_dec.is_finite():
        raise ValueError('only accepts finite input')
    x_tup = x_dec.as_tuple()
    return x_tup


def decimal_1ek(k, signed=False):
    '''Returns ``10 ** k`` or ``-1 * 10 ** k`` in `Decimal`.

    Parameters
    ----------
    k : int
        exponent for value.
    signed : bool
        If True, return negative.

    Returns
    -------
    y : Decimal
        ``10 ** k`` or ``-1 * 10 ** k`` in `Decimal`.
    '''
    y = decimal_from_tuple(signed, (1,), k)
    return y


def decimal_eps(x_dec):
    '''Analog of eps (`np.spacing`) for `Decimal` objects.

    Parameters
    ----------
    x_dec : Decimal
        Input value in decimal.

    Returns
    -------
    y : Decimal
        Smallest value that can be added to `x_dec`.
    '''
    y = decimal_1ek(x_dec.as_tuple().exponent)
    return y


def decimal_to_dot(x_dec):
    '''Test if `Decimal` value has enough precision that it is defined to dot,
    i.e., its eps is <= 1.

    Parameters
    ----------
    x_dec : Decimal
        Input value in decimal.

    Returns
    -------
    y : bool
        True if `x_dec` defined to dot.

    Examples
    --------
    >>> decimal_to_dot(Decimal('1.23E+1'))
    True
    >>> decimal_to_dot(Decimal('1.23E+2'))
    True
    >>> decimal_to_dot(Decimal('1.23E+3'))
    False
    '''
    y = x_dec.is_finite() and (x_dec.as_tuple().exponent <= 0)
    return y


def create_decimal(x, digits, rounding=decimal.ROUND_HALF_UP):
    '''Create `Decimal` object from `float` with desired significant figures.

    Parameters
    ----------
    x : float
        Value to convert to decimal.
    digits : int
        Number of signficant figures to keep in `x`, must be >= 1.
    rounding : str
        Rounding mode, must be one of the rounding modes accepted as in
        `decimal.Context.rounding`.

    Returns
    -------
    y : Decimal
        Conversion of `x` to `Decimal`.
    '''
    assert(digits >= 1)  # Makes not sense otherwise
    with decimal.localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = rounding
        y = +decimal.Decimal(x)
    return y


def digit_str(x_dec):
    '''Decimal to string with only digits (no decimal point, exponent, sign).

    Parameters
    ----------
    x_dec : Decimal
        Input value in `Decimal`.

    Returns
    -------
    y : str
        String of digits in `x_dec`.
    '''
    x_tup = as_tuple_chk(x_dec)
    y = ''.join(str(digit) for digit in x_tup.digits)
    return y

# ============================================================================
# Convert into decimal
# ============================================================================


def decimalize(perf_tbl, err_digits=2, pval_digits=4, default_digits=5,
               EB_limit={}):
    '''Convert a performance table from `float` entries to `Decimal`.

    Parameters
    ----------
    perf_tbl : DataFrame, shape (n_methods, n_metrics * 3)
        DataFrame with curve/loss summary of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (summary, error bar, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo, corresponding error bar, statistical sig).
    err_digits : int
        Number of digits of error to keep for rounding in `Decimal` conversion:
        1.2345 +/- 0.0671 is rounded to 1.235 +/- 0.068 when ``err_digits=2``.
        The error is always rounded up, and the summary is rounded up on half.
        Must be >= 1.
    pval_digits : int
        Precision to keep in p-value when rounding to decimal:
        0.001234 is rounded to 0.0013 when ``pval_digits=4``. The p-value is
        always rounded up. Must be >= 1
    default_digits : int
        Number of digits to keep in estimate when error bar is 0, inf, nan, or
        beyond the error bar limit. Must be >= 1.
    EB_limit : dict of str to int
        Error bar limit in log10 scale for each column. If the
        ``error > 10 ** EB_limit`` then the error is treated as if
        ``error = inf`` since it is too large to be useful. This dictionary is
        optional. Can be positive or negative integer since in log10 scale.

    Returns
    -------
    perf_tbl_dec : DataFrame, shape (n_methods, n_metrics * 3)
        DataFrame with same rows and columns as `perf_tbl`, however the entires
        are now Decimal objects that have been rounded in accordance with the
        input options.
    '''
    assert(pval_digits >= 1)
    assert(perf_tbl.columns.names == (METRIC, STAT))
    metrics, stats = perf_tbl.columns.levels
    assert(sorted(stats) == sorted(STD_STATS))

    assert(perf_tbl.index.name == METHOD)
    methods = perf_tbl.index

    perf_tbl_dec = pd.DataFrame(index=perf_tbl.index, columns=perf_tbl.columns,
                                dtype=object)
    # Check all in same order as original
    assert(list(perf_tbl_dec.columns) == list(perf_tbl.columns))
    for metric in metrics:
        # Handle error bar clipping
        # If error bars are huge, just treat them as inf, print_estimate() will
        # behave accordingly. Also, don't use to quantize mean estimate.
        EB_clip = decimal_1ek(EB_limit.get(metric, 'F'), signed=False)
        for method in methods:
            EB = create_decimal(perf_tbl.loc[method, (metric, ERR_COL)],
                                err_digits, decimal.ROUND_CEILING)
            assert(EB.is_nan() or EB >= 0.0)
            # Going with <= for now, possible < makes more sense.
            EB = EB if EB.is_nan() or EB <= EB_clip else D_INF

            if EB.is_finite() and (not EB.is_zero()):
                mu = decimal.Decimal(perf_tbl.loc[method, (metric, MEAN_COL)])
                mu = mu.quantize(EB, rounding=decimal.ROUND_HALF_UP)
            else:
                # If EB is nan, inf, or 0 just round to default # of digits:
                mu = create_decimal(perf_tbl.loc[method, (metric, MEAN_COL)],
                                    default_digits, decimal.ROUND_HALF_UP)

            # Could use create_decimal to ensure full 17 digits prec, but
            # default is probably good enough.
            pval = decimal.Decimal(perf_tbl.loc[method, (metric, PVAL_COL)])
            pval = pval.quantize(decimal_1ek(-pval_digits),
                                 rounding=decimal.ROUND_CEILING)
            assert(pval.is_nan() or (0 <= pval and pval <= 1))

            perf_tbl_dec.loc[method, metric] = (mu, EB, pval)
    return perf_tbl_dec

# ============================================================================
# Decimal to string
# ============================================================================


def print_estimate(mu, EB, shift=0, min_clip=D_NINF, max_clip=D_INF,
                   below_fmt=BELOW_FMT, above_fmt=ABOVE_FMT,
                   non_finite_fmt={}):
    '''Convert a mean and error bar pair in `Decimal` to a string.

    Parameters
    ----------
    mu : Decimal
        Value of estimate in `Decimal`. Mu must have enough precision to be
        defined to dot after shifting. Can be inf or nan.
    EB : Decimal
        Error bar on estimate in `Decimal`. Must be non-negative. It must be
        defined to same precision (quantum) as `mu` if `EB` is finite positive
        and `mu` is positive.
    shift : int
        How many decimal points to shift `mu` for display purposes. If `mu`
        is in meters and shift=3 than we display the result in mm, i.e., x1e3.
    min_clip : Decimal
        Lower limit clip value on estimate. If ``mu < min_clip`` then simply
        return ``< min_clip`` for string. This is used for score metric where a
        lower metric is simply on another order of magnitude to other methods.
    max_clip : Decimal
        Upper limit clip value on estimate. If ``mu > max_clip`` then simply
        return ``> max_clip`` for string. This is used for loss metric where a
        high metric is simply on another order of magnitude to other methods.
    below_fmt : str (format string)
        Format string to display when estimate is lower limit clipped, often:
        '<{0:,f}'.
    above_fmt : str (format string)
        Format string to display when estimate is upper limit clipped, often:
        '>{0:,f}'.
    non_finite_fmt : dict of str to str
        Display format when estimate is non-finite. For example, for latex
        looking output, one could use:
        ``{'inf': r'\infty', '-inf': r'-\infty', 'nan': '--'}``.

    Returns
    -------
    std_str : str
        String representation of `mu` and `EB`. This is in format 1.234(56)
        for ``mu=1.234`` and ``EB=0.056`` unless there are non-finite values
        or a value has been clipped.
    '''
    assert(min_clip == D_NINF or min_clip.is_finite())
    assert(max_clip == D_INF or max_clip.is_finite())
    assert(min_clip < max_clip)

    shift = int(shift)  # scaleb doesn't like np ints in Py3 => cast to int

    # First check the clipped case
    if (not mu.is_nan()) and max_clip < mu:  # above max
        assert(max_clip.is_finite())
        return above_fmt.format(max_clip.scaleb(shift))
    if (not mu.is_nan()) and mu < min_clip:  # below min
        assert(min_clip.is_finite())
        return below_fmt.format(min_clip.scaleb(shift))

    # Now let's process the non-finite estimate case
    if not mu.is_finite():
        mu_str = NAN_STR if mu.is_nan() else str(float(mu))
        # Default to float string rep if no instructions
        return non_finite_fmt.get(mu_str, mu_str)

    mu_shifted = mu.scaleb(shift)
    if not decimal_to_dot(mu_shifted):
        raise ValueError('Shifting mu too far left for its precision.')

    std_str = GEN_FMT.format(mu_shifted)
    if EB.is_finite():
        # At this point everything should be finite and match quantums
        assert(EB.is_zero() or
               as_tuple_chk(mu).exponent == as_tuple_chk(EB).exponent)
        assert(EB >= 0)
        EB_str = digit_str(EB)
        std_str = '%s(%s)' % (std_str, EB_str)
    assert('E' not in std_str)
    return std_str


def print_pval(pval, below_fmt=BELOW_FMT, non_finite_fmt={}):
    '''Convert decimal p-value into string representation.

    Parameters
    ----------
    pval : Decimal
        Decimal p-value to represent as string. Must be in [0,1] or nan.
    below_fmt : str (format string)
        Format string to display when p-value is lower limit clipped, often:
        ``'<{0:,f}'``.
    non_finite_fmt : dict of str to str
        Display format when estimate is non-finite. For example, for latex
        looking output, one could use: ``{'nan': '--'}``.

    Returns
    -------
    pval_str : str
        String representation of p-value. If p-value is zero or minimum
        Decimal value allowable in precision of pval. We simply return clipped
        string, e.g. ``'<0.0001'``, as value.
    '''
    pval_eps = decimal_eps(pval)
    if pval.is_nan():
        pval_str = non_finite_fmt.get(NAN_STR, NAN_STR)
    elif pval <= pval_eps:
        assert(0 <= pval and pval <= pval_eps)
        # Note this assumes that if pvalue was rounded up to 0.0001
        # then the actual value must be stricly <0.0001 and not equal
        # to 0.0001. This sounds shaky but 1ek is not representable
        # exactly in binary fp anyway, so it is true.
        pval_str = below_fmt.format(pval_eps)
    else:
        assert(pval_eps < pval and pval <= 1)
        # Some style guides suggest we should remove the leading zero
        # here, but format strings give no easy to do that. we could
        # still add that option later.
        pval_str = GEN_FMT.format(pval)
    return pval_str


def get_shift_range(x_dec_list, shift_mod=1):
    '''Helper function to `find_shift` that find upper and lower limits to
    shift the estimates based on the constraints. This bounds the search space
    for the optimal shift.

    Attempts to fulful three constraints:
    1) All estimates displayed to dot after shifting
    2) At least one estimate is >= 1 after shift to avoid space waste with 0s.
    3) ``shift % shift_mod == 0``
    If not all 3 are possible then requirement 2 is violated.

    Parameters
    ----------
    x_dec_list : array-like of Decimal
        List of `Decimal` estimates to format. Assumes all non-finite and
        clipped values are already removed.
    shift_mod : int
        Required modulus for output. This is usually 1 or 3. When an SI prefix
        is desired on the shift then a modulus of 3 is used. Must be >= 1.

    Returns
    -------
    min_shift : int
        Minimum shift (inclusive) to consider to satisfy contraints.
    max_shift : int
        Maximum shift (inclusive) to consider to satisfy contraints.
    all_small : bool
        If True, it means constraint 2 needed to be violated. This could be
        used to flag warning.
    '''
    assert(len(x_dec_list) >= 1)
    assert(shift_mod >= 1)
    assert(all(x.is_finite() for x in x_dec_list))

    # Maximum allowed and keep everything decimal to dot. Arguably this is only
    # relevant for mean estimates with finite errorbars, but we ignore that for
    # the moment for simplicity.
    max_shift_0 = min(-mu.as_tuple().exponent for mu in x_dec_list)
    # Round down to make sure it obeys shift_mod
    max_shift = floor_mod(max_shift_0, shift_mod)
    assert(max_shift % shift_mod == 0 and max_shift <= max_shift_0)

    # Try to keep at least one number >= 1 to avoid wasting space with 0s
    min_shift_0 = min(-mu.adjusted() for mu in x_dec_list)
    # Round up to make sure it obeys shift_mod
    min_shift = ceil_mod(min_shift_0, shift_mod)
    assert(min_shift % shift_mod == 0 and min_shift >= min_shift_0)

    # Might not be possible, in which case, sacrifice >= 1 requirement
    all_small = min_shift > max_shift
    if all_small:
        min_shift = max_shift

    assert(min_shift <= max_shift)
    assert(any(k % shift_mod == 0 for k in range(min_shift, max_shift + 1)))
    return min_shift, max_shift, all_small


def find_shift(mean_list, err_list, shift_mod=1):
    '''Find optimal decimal point shift to display the numbers in `mean_list`
    for display compactness.

    Finds optimal shift of Decimal numbers with potentially varying significant
    figures and varying magnitudes to limit the length of the longest resulting
    string of all the numbers. This is to limit the length of the resulting
    column which is determined by the longest number. This function assumes the
    number will *not* be displayed in a fixed width font and hence the decimal
    point only adds a neglible width. Assumes all clipped and non-finite values
    have been removed from list.

    Attempts to fulful three constraints:
    1) All estimates displayed to dot after shifting
    2) At least one estimate is >= 1 after shift to avoid space waste with 0s.
    3) ``shift % shift_mod == 0``
    If not all 3 are possible then requirement 2 is violated.

    Parameters
    ----------
    mean_list : array-like of Decimal, shape (n,)
        List of `Decimal` estimates to format. Assumes all non-finite and
        clipped values are already removed.
    err_list : array-like of Decimal, shape (n,)
        List of `Decimal` error bars. Must be of same length as `mean_list`.
    shift_mod : int
        Required modulus for output. This is usually 1 or 3. When an SI prefix
        is desired on the shift then a modulus of 3 is used. Must be >= 1.

    Returns
    -------
    best_shift : int
        Best shift of mean_list for compactness. This is number of digits
        to move point to right, e.g. ``shift=3`` => change 1.2345 to 1234.5

    Notes
    -----
    This function is fairly inefficient and could be done implicitly, but it
    shouldn't be the bottleneck anyway for most usages.
    '''
    assert(len(mean_list) == len(err_list))
    # Check all non-finite values for mean removed, but allow non-finite EB
    assert(all(x.is_finite() for x in mean_list))
    assert(shift_mod >= 1)

    if len(mean_list) == 0:
        return 0  # Just return 0 to keep it simple (if all is clipped)

    min_shift, max_shift, _ = get_shift_range(mean_list, shift_mod)

    # Build an order that prefers small magnitude shifts as tie breaker
    L = np.array(range(min_shift, max_shift + 1))
    idx = np.argsort(np.abs(L))
    L = L[idx]

    best_shift = None
    best_len = np.inf
    # Must cast to list for Py3 compatibility
    zip_list = list(zip(mean_list, err_list))
    for shift in L:
        if shift % shift_mod != 0:
            continue

        max_len = max(str_print_len(print_estimate(mu, EB, shift))
                      for mu, EB in zip_list)

        if max_len < best_len:
            best_shift = shift
            best_len = max_len
    assert(best_shift is not None)
    return best_shift


def find_last_dig(num_str):
    '''Find index in string of number (possibly) with error bars immediately
    before the decimal point.

    Parameters
    ----------
    num_str : str
        String representation of a float, possibly with error bars in parens.

    Returns
    -------
    pos : int
        String index of digit before decimal point.

    Examples
    --------
    >>> find_last_dig('5.555')
    0
    >>> find_last_dig('-5.555')
    1
    >>> find_last_dig('-567.555')
    3
    >>> find_last_dig('-567.555(45)')
    3
    >>> find_last_dig('-567(45)')
    3
    '''
    pos = num_str.find('.')
    assert(pos != 0)

    if pos < 0:
        pos = num_str.find('(')
        assert(pos != 0)

    if pos < 0:
        pos = len(num_str)
        assert(pos != 0)

    pos = pos - 1  # Indexing adjust
    return pos


def pad_num_str(num_str_list, pad=' '):
    '''Pad strings of formatted numbers so they are aligned at the decimal
    point when displayed in a right aligned manner (which is typical for
    numeric data).

    Parameters
    ----------
    num_str_list : array-like of str, shape (n,)
        List of numbers already formatted as strings.
    pad : str
        Padding character, typically space. Must be length 1.

    Returns
    -------
    L : list of str, shape (n,)
        List of padded strings.

    Examples
    --------
    >>> sp.pad_num_str(['-55.5', '1.12(34)', '0'], pad='~')
    ['-55.5~~~~~', '1.12(34)', '0~~~~~~~']
    '''
    max_right = max(len(ss) - find_last_dig(ss) for ss in num_str_list)
    L = [ss + pad * (max_right - (len(ss) - find_last_dig(ss)))
         for ss in num_str_list]
    return L


def format_table(perf_tbl_dec, shift_mod=None, pad=True,
                 crap_limit_max={}, crap_limit_min={}, non_finite_fmt={}):
    '''Format a performance table that is already in decimal form to one that
    is formatted with entries in string type.

    Parameters
    ----------
    perf_tbl_dec : DataFrame, shape (n_methods, n_metrics * 3)
        DataFrame with curve/loss summary of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (summary, error bar, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo, corresponding error bar, statistical sig).
        All entries *must* be of type `Decimal`.
    shift_mod : int
        Required modulus for output. This is usually 1 or 3. When an SI prefix
        is desired on the shift then a modulus of 3 is used. Must be >= 1.
        Use None for no shifting at all.
    pad : bool
        If True, pad resulting strings with spaces to make the decimal points
        align. If the resulting strings are TeX source, this will make the
        source more readable but not effect the appearence of the compiled TeX.
    crap_limit_max : dict of str to int
        Dictionary with the log10 max_clip for each column. This is optional.
    crap_limit_min : dict of str to int
        Dictionary with the log10 min_clip for each column. This is optional.
    non_finite_fmt : dict of str to str
        Display format when estimate is non-finite. For example, for latex
        looking output, one could use:
        ``{'inf': r'\infty', '-inf': r'-\infty', 'nan': '--'}``.

    Returns
    -------
    perf_tbl_str : DataFrame, shape (n_methods, n_metrics * 2)
        DataFrame with summary string of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (estimate with error, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo with error bar, statistical sig).
        All entries are of type string.
    shifts : dict of str to int
        The used shift in log10 scale for each metric.
    '''
    # For now, require perf_tbl to be all finite, might relax later.
    assert(perf_tbl_dec.columns.names == (METRIC, STAT))
    metrics, stats = perf_tbl_dec.columns.levels
    assert(sorted(stats) == sorted(STD_STATS))

    assert(perf_tbl_dec.index.name == METHOD)
    methods = perf_tbl_dec.index

    # This system might work with an empty table, but I assert to be safe.
    assert(len(perf_tbl_dec.index) > 0)
    assert(len(perf_tbl_dec.columns) > 0)

    # Note: metrics will be in sorted order, not original order from perf_tbl
    cols = pd.MultiIndex.from_product([metrics, FMT_STATS],
                                      names=[METRIC, STAT])
    perf_tbl_str = pd.DataFrame(index=methods, columns=cols, dtype=object)
    shifts = {}
    for metric in metrics:
        mean_series = perf_tbl_dec[(metric, MEAN_COL)]
        # We might end up modifying this, so simpler to make a copy to be safe
        err_series = pd.Series(perf_tbl_dec[(metric, ERR_COL)], copy=True)
        # Code works with p-values having diff # of digits but looks weird
        assert(all_same([p.as_tuple().exponent
                         for p in perf_tbl_dec[(metric, PVAL_COL)]
                         if p.is_finite()]))

        # Handle mean estimate clipping
        # Use -inf and inf limits as default
        min_clip = decimal_1ek(crap_limit_min.get(metric, 'F'), signed=True)
        max_clip = decimal_1ek(crap_limit_max.get(metric, 'F'), signed=False)
        # Will ignore clipped values for shifting purposes, also check finite
        idx = (min_clip <= mean_series) & (mean_series <= max_clip)
        # Subroutines can handle inf mean, but keep non-clipped finite for now
        assert(decimal_all_finite(mean_series[idx]))
        if len(idx) == 0:
            warnings.warn('no non-clipped values for metric %s' % str(metric))

        # Find the best shift
        if shift_mod is None:  # => no shifting at all
            best_shift = 0
            # Check all to dot, otherwise will get error. We could do this
            # check in an except block only to optimize computation.
            if not all(decimal_to_dot(x) for x in mean_series[idx]):
                ValueError('shift_mod=None not possible for %s due to '
                           'insufficient precision' % metric)
        else:
            # The .tolist() might not be needed, but doing anyway to be safe.
            best_shift = find_shift(mean_series[idx].tolist(),
                                    err_series[idx].tolist(),
                                    shift_mod=shift_mod)
        shifts[metric] = best_shift

        # Now actually do it
        for method in methods:
            mu, EB = mean_series[method], err_series[method]
            estimate_str = print_estimate(mu, EB, best_shift,
                                          min_clip, max_clip,
                                          non_finite_fmt=non_finite_fmt)
            perf_tbl_str.loc[method, (metric, EST_COL)] = estimate_str

            pval = perf_tbl_dec.loc[method, (metric, PVAL_COL)]
            pval_str = print_pval(pval, non_finite_fmt=non_finite_fmt)
            perf_tbl_str.loc[method, (metric, PVAL_COL)] = pval_str

        if pad:  # Overwrite with padded values if requested
            perf_tbl_str.loc[:, (metric, EST_COL)] = \
                pad_num_str(perf_tbl_str[(metric, EST_COL)])
    return perf_tbl_str, shifts


def adjust_headers(headers, shifts, unit_dict, use_prefix=True, use_tex=False):
    '''Adjust the headers of a table generated by format_table to reflect the
    shift.

    Parameters
    ----------
    headers : array-like of str, shape (n_metrics,)
        List of metrics to adjust
    shifts : dict of str to int
        The used shift in log10 scale for each metric.
    unit_dict : dict or str to str
        Dictionary from metric name to associated unit symbol. Treat as
        unitless if entry is missing for a metric.
    use_prefix : bool
        If True, attempt to apply SI prefix to unit symbol for shift.
    use_tex : bool
        If True, adjust headers with TeX based formatting.

    Returns
    -------
    headers : list of str, shape (n_metrics,)
        New header strings in same order as headers.

    Notes
    -----
    Requiring list `headers` is not redundant with dictionary `shifts` which
    contains the same entries as keys because we care about the order. Standard
    dictionaries in Python do not guarantee order.
    '''
    prefix_dict = _PREFIX_TEX if use_tex else _PREFIX
    fmt_shift = '%s $\times 10^{%d}$' if use_tex else '%s x 1e%d'
    fmt_unit = '%s (%s)'
    fmt_shift_unit = fmt_shift + ' (%s)'

    new_headers = [None] * len(headers)
    for nn, el in enumerate(headers):
        metric, stat = el
        assert(metric in shifts)
        assert(stat in (EST_COL, PVAL_COL))

        shift = shifts[metric]
        rev_shift = -shift
        # Removes _ for spaces for display, Note: assuming metric already str!
        metric_str = metric.replace('_', ' ')

        unit = unit_dict.get(metric, None)
        if stat == PVAL_COL:
            new_headers[nn] = PVAL_COL
        elif unit is None:  # ==> stat == EST_COL, unitless
            if shift != 0:
                new_headers[nn] = fmt_shift % (metric_str, shift)
            else:
                new_headers[nn] = metric_str
        else:  # Need to display units
            prefix = prefix_dict.get(rev_shift, None)
            if use_prefix and (prefix is not None):
                new_headers[nn] = '%s (%s%s)' % (metric_str, prefix, unit)
            elif shift != 0:
                new_headers[nn] = fmt_shift_unit % (metric_str, shift, unit)
            else:
                new_headers[nn] = fmt_unit % (metric_str, unit)

        if use_tex:
            # siunitx package confused w/o {}, maybe better way than if
            new_headers[nn] = '{%s}' % new_headers[nn]
    return new_headers


def table_to_latex(perf_tbl_str, shifts, unit_dict, use_prefix=True):
    r'''Export performance table already converted to string entries to a
    single string of LaTeX source.

    This function includes adjustement of headers to reflect shift and display
    units.

    Parameters
    ----------
    perf_tbl_str : DataFrame, shape (n_methods, n_metrics * 2)
        DataFrame with summary string of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (estimate with error, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo with error bar, statistical sig).
        All entries must be of type string.
    shifts : dict of str to int
        The used shift in log10 scale for each metric.
    unit_dict : dict or str to str
        Dictionary from metric name to associated unit symbol. Treat as
        unitless if entry is missing for a metric.
    use_prefix : bool
        If True, attempt to apply SI prefix to unit symbol for shift.

    Returns
    -------
    latex_str : str
        String containing LaTeX export of perf_tbl_str.

    Notes
    -----
    Pandas LaTeX export requires ``\usepackage{booktabs}`` and proper aligning
    of the decimal point requires ``\usepackage{siunitx}``.
    '''
    assert(perf_tbl_str.columns.names == [METRIC, STAT])
    n_metrics, rem = divmod(len(perf_tbl_str.columns), 2)
    assert(rem == 0)
    align = '|l' + '|Sr' * n_metrics + '|'

    new_headers = adjust_headers(perf_tbl_str.columns, shifts, unit_dict,
                                 use_prefix=use_prefix, use_tex=True)
    # Avoid doing inplace changes to perf_tbl_str, need index name to be none
    # anyways to avoid a bug in pandas (0.19.2) that puts the midrule in the
    # wrong place. Maybe in a future version of pandas this will not be needed.
    perf_tbl_str = pd.DataFrame(data=perf_tbl_str.values, columns=new_headers,
                                index=perf_tbl_str.index.values)
    latex_str = perf_tbl_str.to_latex(escape=False, column_format=align,
                                      index_names=False)
    return latex_str


def table_to_string(perf_tbl_str, shifts, unit_dict, use_prefix=True):
    '''Export performance table already converted to string entries to a single
    string of nicely formatted output in human readable form.

    This function includes adjustement of headers to reflect shift and display
    units.

    Parameters
    ----------
    perf_tbl_str : DataFrame, shape (n_methods, n_metrics * 2)
        DataFrame with summary string of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (estimate with error, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo with error bar, statistical sig).
        All entries must be of type string.
    shifts : dict of str to int
        The used shift in log10 scale for each metric.
    unit_dict : dict or str to str
        Dictionary from metric name to associated unit symbol. Treat as
        unitless if entry is missing for a metric.
    use_prefix : bool
        If True, attempt to apply SI prefix to unit symbol for shift.

    Returns
    -------
    latex_str : str
        String containing nicely formatted output in human readable form.
    '''
    assert(perf_tbl_str.columns.names == [METRIC, STAT])

    new_headers = adjust_headers(perf_tbl_str.columns, shifts, unit_dict,
                                 use_prefix=use_prefix, use_tex=False)
    # Avoid doing inplace changes to perf_tbl_str
    perf_tbl_str = pd.DataFrame(data=perf_tbl_str.values, columns=new_headers,
                                index=perf_tbl_str.index.values)
    tbl_str = perf_tbl_str.to_string(index=True, index_names=False)
    return tbl_str


def just_format_it(perf_tbl_fp, unit_dict={}, shift_mod=None,
                   crap_limit_max={}, crap_limit_min={}, EB_limit={},
                   non_finite_fmt={}, use_tex=False, use_prefix=True):
    r'''One stop function call to format a results table and get the output as
    a string in readable human plain text or as LaTeX source.

    Parameters
    ----------
    perf_tbl_fp : DataFrame, shape (n_methods, n_metrics * 3)
        DataFrame with curve/loss summary of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (summary, error bar, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo, corresponding error bar, statistical sig).
        The entries should all be `float`.
    unit_dict : dict or str to str
        Dictionary from metric name to associated unit symbol. Treat as
        unitless if entry is missing for a metric.
    shift_mod : int
        Required modulus for output. This is usually 1 or 3. When an SI prefix
        is desired on the shift then a modulus of 3 is used. Must be >= 1.
        Use None for no shifting at all.
    crap_limit_max : dict of str to int
        Dictionary with the log10 max_clip for each column. This is optional.
    crap_limit_min : dict of str to int
        Dictionary with the log10 min_clip for each column. This is optional.
    EB_limit : dict of str to int
        Error bar limit in log10 scale for each column. If the
        ``error > 10 ** EB_limit`` then the error is treated as if
        ``error = inf`` since it is too large to be useful. This dictionary is
        optional. Can be positive or negative integer since in log10 scale.
    non_finite_fmt : dict of str to str
        Display format when estimate is non-finite. For example, for latex
        looking output, one could use:
        ``{'inf': r'\infty', '-inf': r'-\infty', 'nan': '--'}``.
    use_tex : bool
        If True, adjust headers with TeX based formatting.
    use_prefix : bool
        If True, attempt to apply SI prefix to unit symbol for shift.

    Returns
    -------
    str_out : str
        String containing formatted table in plain text or LaTeX.

    Notes
    -----
    For Pandas ``use_tex=True``, LaTeX export requires
    ``\usepackage{booktabs}`` and proper aligning of the decimal point requires
    ``\usepackage{siunitx}``.
    '''
    to_str = table_to_latex if use_tex else table_to_string

    perf_tbl_dec = decimalize(perf_tbl_fp, EB_limit=EB_limit)
    perf_tbl_str, shifts = format_table(perf_tbl_dec, shift_mod=shift_mod,
                                        crap_limit_max=crap_limit_max,
                                        crap_limit_min=crap_limit_min,
                                        non_finite_fmt=non_finite_fmt)
    str_out = to_str(perf_tbl_str, shifts, unit_dict, use_prefix=use_prefix)
    return str_out
