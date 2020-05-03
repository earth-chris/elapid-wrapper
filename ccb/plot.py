"""
"""
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.stats import _gaussian_kde


def density_dist(
    ydata,
    plot=None,
    color=None,
    aei_color=None,
    fill=True,
    fill_alpha=0.3,
    label=None,
    linewidth=2,
    xlabel="Values",
    ylabel="Density",
    title="Density Distributions",
    xlim=None,
    ylim=None,
    covar=0.25,
    cutoff=2,
    **kwargs,
):
    """
    Plots a density distribution. all data will be displayed on the same figure.
    :param ydata: a list of numpy arrays, or a 1- or 2-d numpy array of values to plot in one figure.
    :param plot: a matplotlib pyplot object. creates one if not set.
    :param color: a single color or an array of colors to plot with
    :param fill: set this to true to color the space beneath the distribution
    :param fill_alpha: the alpha value for the fill
    :param label: the labels to assign in the legend
    :param linewidth: the width of the density plot line
    :param xlabel: the x-axis label
    :param ylabel: the y-axis label
    :param title: the plot title
    :param xlim: a 2-element list of [xmin, xmax] for plotting
    :param ylim: a 2-element list for [ymin, ymax] for plotting !! NOT IMPLEMENTED
    :param covar: the covariance scalar for calculating the density dist.
    :param cutoff: the 0-100 based cutoff for clipping min/max values (e.g. use 2 to clip from 2-98% of the values)
    :param **kwargs: matplotlib pyplot.plot keyword arguments
    :return plot: a matplotlib pyplot object
    """

    # we want ydata to come as a list form to handle uneven sample sizes
    if type(ydata) is list:
        ncol = len(ydata)

    # set up a function to handle numpy arrays
    elif type(ydata) is _np.ndarray:

        # if the ndarray is only 1-d, convert it to a list
        if ydata.ndim == 1:
            ydata = [ydata]

        # otherwise, loop through each column and assign as unique items in list
        else:
            newdata = []
            for i in range(ydata.shape[1]):
                newdata.append(ydata[:, i])
            ydata = newdata
        ncol = len(ydata)
    else:
        print("[ ERROR ]: unsupported ydata format. must be a list or np.ndarray")

    # if a plot object isn't provided, create one
    if not plot:
        plot = _plt
        plot.figure(_np.random.randint(100))

    # handle colors. if only one is passed, set it as a list for indexing
    if color is not None:
        if type(color) is str:
            color = list(color)

    # handle labels similar to color, but don't assign defaults
    if label is not None:
        if type(label) is str:
            label = list(label)
        else:
            if len(label) < ncol:
                print("[ ERROR ]: number of labels specified doesn't match number of columns")
                label = []
                for i in range(ncol):
                    label.append(None)
    else:
        label = []
        for i in range(ncol):
            label.append(None)

    # if xlim isn't set, find the min/max range for plot based on %cutoff
    if not xlim:
        xmin = []
        xmax = []
        for i in range(ncol):
            xmin.append(_np.percentile(_np.array(ydata[i]), cutoff))
            xmax.append(_np.percentile(_np.array(ydata[i]), 100 - cutoff))
        xlim = [min(xmin), max(xmax)]

    # set the x plot size
    xs = _np.linspace(xlim[0], xlim[1])

    # loop through each feature, calculate the covariance, and plot
    for i in range(ncol):
        dns = _gaussian_kde(_np.array(ydata[i]))
        dns.covariance_factor = lambda: covar
        dns._compute_covariance()
        ys = dns(xs)

        # plotting functions
        plot.plot(xs, ys, label=label[i], color=color[i], linewidth=linewidth, **kwargs)
        if fill:
            plot.fill_between(xs, ys, color=color[i], alpha=fill_alpha)

    # finalize other meta plot routines
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.title(title)
    if label[0] is not None:
        plot.legend()
    plot.tight_layout()

    # return the final plot object for further manipulation
    return plot
