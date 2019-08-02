import numpy as np
import matplotlib.pyplot as plt


def logish_axes(ax, logstep=0.5):
    """ Display linearly spaced log ticks on a logarithmic plot. """
    ticks = 10**np.arange(-10, 10, logstep)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Remove minor ticks of the log axes
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.yaxis.set_major_locator(plt.FixedLocator(ticks))

    # Display a log of the tick value, e.g. `-1` instead if `0.1`
    def formatter(tick, num):
        return np.log10(tick)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))

    return ax
