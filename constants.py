# Ryan Turner (turnerry@iro.umontreal.ca)

# Settings
PAIRWISE_DEFAULT = False

# Pandas indices
METRIC = 'metric'
METHOD = 'method'
STAT = 'stat'
HORIZON = 'horizon'

# Pandas columns
MEAN_COL = 'mean'
ERR_COL = 'error'
PVAL_COL = 'p'
STD_STATS = (MEAN_COL, ERR_COL, PVAL_COL)

EST_COL = 'estimate'  # mean and error in one col
FMT_STATS = (EST_COL, PVAL_COL)

# STD formatting operations
GEN_FMT = '{0:,f}'
ABOVE_FMT = '>' + GEN_FMT
BELOW_FMT = '<' + GEN_FMT

# For curves
XGRID = 'xgrid'
YGRID = 'ygrid'
LB = 'LB'
UB = 'UB'
CURVE_STATS = ('xgrid', 'ygrid', 'LB', 'UB')

_PREFIX = {-24: 'y',
           -21: 'z',
           -18: 'a',
           -15: 'f',
           -12: 'p',
           -9: 'n',
           -6: 'u',
           -3: 'm',
           -2: 'c',
           -1: 'd',
           0: '',
           3: 'k',
           6: 'M',
           9: 'G',
           12: 'T',
           15: 'P',
           18: 'E',
           21: 'Z',
           24: 'Y'}

_PREFIX_TEX = dict(_PREFIX)
_PREFIX_TEX[-6] = r'$\mu$'
