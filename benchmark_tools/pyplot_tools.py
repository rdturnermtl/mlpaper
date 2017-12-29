# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function
import matplotlib.pyplot as plt
from benchmark_tools.constants import METHOD, METRIC, STAT, HORIZON
from benchmark_tools.constants import MEAN_COL, ERR_COL

DATASET = 'dataset'  # TODO move to constants
UNIT_FMT = '%s (%s)'


def plot_perf_table(perf_tbl, bar=True, figsize_inches=None, grid='on',
                    style=None, fontsize=None, metric_units={}, **kwds):
    '''pandas seems good for limits auto so we don't take that input.'''
    # TODO allow fine grain control of tick, label, legend font size
    # TODO do better tweaking of axis limits than default, option for include 0
    # TODO units on horizon
    assert(list(perf_tbl.columns.names) in ([METRIC, STAT],
                                            [METRIC, STAT, HORIZON],
                                            [METRIC, STAT, DATASET]))
    assert(perf_tbl.index.name == METHOD)
    simple = len(perf_tbl.columns.names) == 2
    time_series = perf_tbl.columns.names[-1] == HORIZON

    kind = 'bar' if bar else 'line'
    metrics = perf_tbl.columns.levels[0]

    fig_dict = {}
    for metric_name in metrics:
        df = perf_tbl.xs(metric_name, axis=1, level=METRIC)
        mu = df[MEAN_COL] if simple else df.xs(MEAN_COL, axis=1, level=STAT).T
        EB = df[ERR_COL] if simple else df.xs(ERR_COL, axis=1, level=STAT).T

        # sort descending for horizon, so furthest horizon is to left
        if time_series:
            mu = mu.sort_index(axis=0, ascending=False)
            assert(mu.index.name == HORIZON and mu.columns.name == METHOD)
            # Pandas can actually handle it if the index of mu and EB are in
            # different order, so this is not really needed, altho it is
            # cleaner to keep everthing in same order. Could delete this.
            EB = EB.sort_index(axis=0, ascending=False)
            assert(EB.index.name == HORIZON and EB.columns.name == METHOD)
            assert(np.all(mu.index == EB.index))
            assert(np.all(mu.columns == EB.columns))

        fig_dict[metric_name], ax = plt.subplots()
        # Keep y-ticks as default, but force to index for x-ticks. Default
        # legend placement sucks so we set to false and do outside.
        xticks = None if bar or simple else mu.index  # pyplot confused for bar
        mu.plot(yerr=EB, kind=kind, ax=ax, figsize=figsize_inches, grid=grid,
                legend=False, style=style, xticks=xticks, fontsize=fontsize,
                **kwds)

        # Put a legend to the right of the current axis, not needed if just a
        # simple plot without grouping or multiple curves, must first shrink
        # current axis by 10% to make space.
        if not simple:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=METHOD)

        unit = metric_units.get(metric_name, None)
        y_str = metric_name if unit is None else UNIT_FMT % (metric_name, unit)
        ax.set_ylabel(y_str)
    return fig_dict

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    np.random.seed(7643)

    metrics = ['ff', 'gg', 'hh']
    m_units = {'ff': 'm', 'hh': '$US'}

    methods = ['foo', 'bar', 'baz', 'qux']
    h = [1.2, 5.5, 3.0]  # Out of order

    col_names = pd.MultiIndex.from_product([metrics, (MEAN_COL, ERR_COL), h],
                                           names=[METRIC, STAT, HORIZON])
    dat = np.random.rand(len(methods), len(col_names))
    df = pd.DataFrame(data=dat, index=methods, columns=col_names)
    df.index.name = METHOD

    fig_dict = plot_perf_table(df, grid='on', bar=False, metric_units=m_units)
    fig_dict = plot_perf_table(df, grid='on', bar=True, metric_units=m_units)

    df0 = df.xs(h[0], axis=1, level=HORIZON)
    fig_dict = plot_perf_table(df0, grid='on', bar=False, metric_units=m_units)
    fig_dict = plot_perf_table(df0, grid='on', bar=True, metric_units=m_units)

    df.columns.names = [METRIC, STAT, DATASET]
    fig_dict = plot_perf_table(df, grid='on', bar=False, metric_units=m_units)
    fig_dict = plot_perf_table(df, grid='on', bar=True, metric_units=m_units)
