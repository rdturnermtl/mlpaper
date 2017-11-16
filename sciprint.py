# Ryan Turner (turnerry@iro.umontreal.ca)
import warnings
import numpy as np
import pandas as pd
import decimal
from constants import METHOD, METRIC, STAT, STD_STATS, FMT_STATS
from constants import MEAN_COL, ERR_COL, PVAL_COL, EST_COL
from constants import GEN_FMT, ABOVE_FMT, BELOW_FMT, _PREFIX, _PREFIX_TEX

# Some numeric constants
NAN_STR = str(np.nan)
D_INF = decimal.Decimal('Infinity')
D_NINF = decimal.Decimal('-Infinity')

# ============================================================================
# General utils
# ============================================================================


def all_same(L):
    return len(L) == 0 or all(x == L[0] for x in L)


def floor_mod(x, mod):
    return (x // mod) * mod


def ceil_mod(x, mod):
    return floor_mod(x, -mod)


def str_print_len(x_str):
    return len(x_str.translate(None, ',.'))

# ============================================================================
# Decimal utils
# ============================================================================


def decimal_all_finite(x_dec_list):
    return all(x.is_finite() for x in x_dec_list)


def decimal_from_tuple(signed, digits, expo):
    return decimal.Decimal(decimal.DecimalTuple(int(signed), digits, expo))


def as_tuple_chk(x_dec):
    if not x_dec.is_finite():
        raise ValueError('only accepts finite input')
    x_tup = x_dec.as_tuple()
    return x_tup


def decimal_1ek(k, signed=False):
    return decimal_from_tuple(signed, (1,), k)


def decimal_eps(x_dec):
    return decimal_1ek(x_dec.as_tuple().exponent)


def decimal_left_digits(x_dec):
    assert(decimal_to_dot(x_dec))  # Should make exception
    y = 1 + max(0, x_dec.adjusted())
    return y


def decimal_right_digits(x_dec):
    assert(decimal_to_dot(x_dec))  # Should make exception
    x_tup = as_tuple_chk(x_dec)
    y = max(0, -x_tup.exponent)
    return y


def decimal_digits(x_dec):
    assert(decimal_to_dot(x_dec))  # Should make exception
    x_tup = as_tuple_chk(x_dec)
    y = len(x_tup.digits) - min(0, x_dec.adjusted())
    return y


def decimal_floor_log10_abs(x_dec):
    assert(x_dec.is_finite() and x_dec != 0)  # Should make exception
    return x_dec.adjusted()


def decimal_ceil_log10_abs(x_dec):
    k = decimal_floor_log10_abs(x_dec)
    assert(abs(decimal_1ek(k)) <= abs(x_dec))
    y = k + int(decimal_1ek(k, signed=x_dec.is_signed()) != x_dec)
    return y


def decimal_next_pow10(x_dec):
    if x_dec == 0:
        return x_dec  # Note: this keeps sign and precision of original 0.
    k = decimal_ceil_log10_abs(x_dec)
    y = decimal_1ek(k, signed=x_dec.is_signed())
    return y


def decimal_to_dot(x_dec):
    return x_dec.is_finite() and (x_dec.as_tuple().exponent <= 0)


def create_decimal(x, digits, rounding=decimal.ROUND_HALF_UP):
    assert(digits >= 1)  # Makes not sense otherwise
    with decimal.localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = rounding
        y = +decimal.Decimal(x)
    return y


def digit_str(x_dec):
    x_tup = as_tuple_chk(x_dec)
    return ''.join(str(digit) for digit in x_tup.digits)

# ============================================================================
# Convert into decimal
# ============================================================================


def decimalize(perf_tbl, err_digits=2, pval_digits=4, default_digits=5,
               EB_limit={}):
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
    assert(min_clip == D_NINF or min_clip.is_finite())
    assert(max_clip == D_INF or max_clip.is_finite())
    assert(min_clip < max_clip)

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


def print_pval(pval, non_finite_fmt={}):
    pval_eps = decimal_eps(pval)
    if pval.is_nan():
        pval_str = non_finite_fmt.get(NAN_STR, NAN_STR)
    elif pval <= pval_eps:
        assert(0 <= pval and pval <= pval_eps)
        # Note this assumes that if pvalue was rounded up to 0.0001
        # then the actual value must be stricly <0.0001 and not equal
        # to 0.0001. This sounds shaky but 1ek is not representable
        # exactly in binary fp anyway, so it is true.
        pval_str = BELOW_FMT.format(pval_eps)
    else:
        assert(pval_eps < pval and pval <= 1)
        # Some style guides suggest we should remove the leading zero
        # here, but format strings give no easy to do that. we could
        # still add that option later.
        pval_str = GEN_FMT.format(pval)
    return pval_str


def get_shift_range(x_dec_list, shift_mod=1):
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
    assert(any(k % shift_mod == 0 for k in xrange(min_shift, max_shift + 1)))
    return min_shift, max_shift, all_small


def find_shift(mean_list, err_list, shift_mod=1):
    '''This function is fairly inefficient and could be done implicitly, but it
    shouldn't be the bottleneck anyway for most usages.'''
    assert(len(mean_list) == len(err_list))
    # Check all non-finite values for mean removed, but allow non-finite EB
    assert(all(x.is_finite() for x in mean_list))
    assert(shift_mod >= 1)

    if len(mean_list) == 0:
        return 0  # Just return 0 to keep it simple (if all is clipped)

    min_shift, max_shift, _ = get_shift_range(mean_list, shift_mod)

    # Build an order that prefers small magnitude shifts as tie breaker
    L = np.array(xrange(min_shift, max_shift + 1))
    idx = np.argsort(np.abs(L))
    L = L[idx]

    best_shift = None
    best_len = np.inf
    zip_list = zip(mean_list, err_list)
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
    pos = num_str.find('.')
    assert(pos != 0)

    if pos < 0:
        pos = num_str.find('(')
        assert(pos != 0)

    if pos < 0:
        pos = len(num_str)
        assert(pos != 0)
    return pos - 1


def pad_num_str(num_str_list, pad=' '):
    max_right = max(len(ss) - find_last_dig(ss) for ss in num_str_list)
    L = [ss + ' ' * (max_right - (len(ss) - find_last_dig(ss)))
         for ss in num_str_list]
    return L


def format_table(perf_tbl_dec, shift_mod=None, pad=True,
                 crap_limit_max={}, crap_limit_min={}, non_finite_fmt={}):
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
        best_shift = 0
        if shift_mod is not None:
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
    to_str = table_to_latex if use_tex else table_to_string

    perf_tbl_dec = decimalize(perf_tbl_fp, EB_limit=EB_limit)
    perf_tbl_str, shifts = format_table(perf_tbl_dec, shift_mod=shift_mod,
                                        crap_limit_max=crap_limit_max,
                                        crap_limit_min=crap_limit_min,
                                        non_finite_fmt=non_finite_fmt)
    str_out = to_str(perf_tbl_str, shifts, unit_dict, use_prefix=use_prefix)
    return str_out
