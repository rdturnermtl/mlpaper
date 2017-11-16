# Ryan Turner (turnerry@iro.umontreal.ca)
import itertools
import numpy as np
import pandas as pd
from pandas import MultiIndex
from constants import MEAN_COL, ERR_COL
from constants import METHOD, METRIC, STAT, HORIZON
# Plotting last
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.models.sources import ColumnDataSource
from bokeh.palettes import Category10, Category20

DEFAULT_NAME = 'benchmark run'
PM_CHAR = u'\u00B1'
FNAME_FMT = '%s_%s.html'


def default_palette(N):
    return Category10 if N <= 10 else Category20


def file_safe(fname):
    # Could extend this to handle other non alpha-num chars as well
    return fname.replace(' ', '_')


def build_color_dict(labels, palette=Category20):
    '''Pallet should in form of std Bokeh pallet object which is
    dict of lists of color strings.'''
    colors = itertools.cycle(palette[len(labels)])
    D = dict(itertools.izip(labels, colors))
    return D


def errorbar(fig, x, y, err, name='data', vertical=True, color='blue',
             line_kwargs={}, point_kwargs={}, error_kwargs={}):
    data = {'x': x, 'y': y, 'err': err, 'name': [name] * len(x)}
    # Name gets used for legend with line
    fig.line('x', 'y', source=ColumnDataSource(data),
             legend=name, color=color, **line_kwargs)
    # Name gets used for hover tool with circle
    fig.circle('x', 'y', source=ColumnDataSource(data),
               name=name, color=color, **point_kwargs)

    # This could be vectorized, but probably not worth the trouble
    if vertical:
        y_err_x = []
        y_err_y = []
        for px, py, perr in zip(x, y, err):
            y_err_x.append((px, px))
            y_err_y.append((py - perr, py + perr))
        data = {'xs': y_err_x, 'ys': y_err_y}
    else:
        x_err_x = []
        x_err_y = []
        for px, py, perr in zip(x, y, err):
            x_err_x.append((px - perr, px + perr))
            x_err_y.append((py, py))
        data = {'xs': x_err_x, 'ys': x_err_y}
    fig.multi_line('xs', 'ys', source=ColumnDataSource(data),
                   color=color, **error_kwargs)

# TODO make bar version


def plot_perf_table(perf_tbl, run_name=DEFAULT_NAME, max_offset=0.0):
    '''Make plot of results over diff horizons'''
    # TODO option for preformatted text, also display side col
    assert(list(perf_tbl.columns.names) == [METRIC, STAT, HORIZON])
    assert(perf_tbl.index.name == METHOD)
    done = bp.save  # show or save

    methods = perf_tbl.index.values
    method_to_color = build_color_dict(methods, default_palette(len(methods)))

    n_methods = np.maximum(1.0, len(methods) - 1.0)   # Now a float
    for metric_name in perf_tbl.columns.levels[0]:
        fname = file_safe('%s_%s.html' % (run_name, metric_name))
        bp.output_file(fname)
        # option to add units
        fig = bp.figure(title=run_name,
                        x_axis_label=HORIZON, y_axis_label=metric_name)

        my_hover = HoverTool(names=methods)
        # TODO include color in hover tip
        my_hover.tooltips = [(METHOD, '@name'),
                             (metric_name, u'@y %s @err' % PM_CHAR)]
        fig.add_tools(my_hover)

        for ii, method_name in enumerate(methods):
            mu = perf_tbl.loc[method_name, (metric_name, MEAN_COL)]
            EB = perf_tbl.loc[method_name, (metric_name, ERR_COL)]
            assert(all(mu.index == EB.index))  # Should now be series

            offset = (ii / n_methods) * max_offset
            x_grid = (-mu.index.values) + offset

            errorbar(fig, x_grid, mu.values, EB.values,
                     name=method_name, color=method_to_color[method_name],
                     line_kwargs={'line_width': 2, 'line_alpha': 0.5},
                     point_kwargs={'size': 10})
        done(fig)

if __name__ == '__main__':
    np.random.seed(8123)

    C = MultiIndex.from_product([('score',), [MEAN_COL, ERR_COL], xrange(5)],
                                names=[METRIC, STAT, HORIZON])
    methods = ('foo', 'bar', 'baz')

    df = pd.DataFrame(data=np.random.rand(len(methods), len(C)),
                      index=methods, columns=C, dtype=float)
    df.index.name = METHOD

    plot_perf_table(df)
