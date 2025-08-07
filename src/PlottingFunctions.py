#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class related to making figure-quality plots.

Probably best to set your default sans-serif font to Helvetica before you make
figures: https://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/

The maximum published width for a one-column
figure is 3.33 inches (240 pt). The maximum width for a two-column
figure is 6.69 inches (17 cm). The maximum depth of figures should
be 8 ¼ in. (21.1 cm).

panel labels are 8 point font, ticks are 7 point font,
annotations and legends are 6 point font.

"""
import matplotlib  # requires 3.8.0
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import mpltern
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.font_manager import FontProperties

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)


class Plotter:
    def __init__(self, poster=False, dark_background=False):
        self.poster = poster
        self = self
        if dark_background == True:
            self.db = True
        else:
            self.db = False
        return

    def one_column_plot(
        self, npanels=1, ratios=[1], height=None, width=None, projection=None
    ):
        """one_column_plot function
        takes data and makes a one-column width figure

        Args:
            nrows (int): number of rows
            npanels (int): number of panels
            ratios (list): list of heights of same length as nrows
            height (float): overridden height of figure
            width (float): overriden width of figure
        Returns:
            fig (figure): figure object
            ax (axes): axes object"""

        # first, check everything matches
        try:
            if len(ratios) != npanels:
                raise Exception("Number of ratios incorrect")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 0.5

        xsize = 3.33  # 3.33 inches for one-column figure
        if (height is not None) and (width is not None):
            ysize = np.min([height, width])  # maximum size in y can be 8.25
        elif (height is not None) and (width is None):
            ysize = np.min([height, 8.25])  # maximum size in y can be 8.25
        else:
            ysize = np.min([3.5 * npanels, 8.25])  # maximum size in y can be 8.25

        plt.rcParams["figure.figsize"] = [xsize, ysize]
        plt.rcParams["font.size"] = fontsz
        plt.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.rcParams["axes.linewidth"] = lw  # set the value globally
        plt.rcParams["figure.constrained_layout.use"] = True
        if self.db == True:
            plt.style.use("dark_background")

        fig, axs = plt.subplots(
            npanels, 1, height_ratios=ratios, frameon=False
        )  # create number of panels

        # clean up axes, tick parameters
        if npanels == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw * 4)
            axs.yaxis.set_tick_params(width=lw, length=lw * 4)
            axs.tick_params(axis="both", pad=1.2)
        else:
            for i in np.arange(npanels):
                axs[i].xaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].tick_params(axis="both", pad=1.2)
        return fig, axs

    def two_column_plot(
        self,
        nrows=1,
        ncolumns=1,
        heightratio=[1],
        widthratio=[1],
        width=0,
        height=0,
        big=False,
    ):
        """two_column_plot function
        takes data and makes a two-column width figure

        Args:
            nrows (int): number of rows
            ncolumns (int): number of columns
            heightratio (list): list of heights of same length as nrows
            widthratio (list): list of widths of same length as ncolumns
            height (float): overridden height of figure
            big (boolean): if big is True, uses larger font sizes

        Returns:
            fig (figure): figure object
            ax (axes): axes object"""

        # first, check everything matches
        try:
            if len(heightratio) != nrows:
                raise Exception("Number of height ratios incorrect")
            if len(widthratio) != ncolumns:
                raise Exception("Number of width ratios incorrect")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 1

        if width == 0:
            if big == True:
                xsize = 5 * ncolumns
            else:
                xsize = 6.69  # 3.33 inches for one-column figure
        else:
            xsize = width

        if height == 0:
            if big == True:
                ysize = 5 * nrows
            else:
                ysize = 3 * nrows
        else:
            ysize = height

        plt.rcParams["figure.figsize"] = [xsize, ysize]
        plt.rcParams["font.size"] = fontsz
        plt.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.rcParams["axes.linewidth"] = lw  # set the value globally
        plt.rcParams["figure.constrained_layout.use"] = True
        if self.db == True:
            plt.style.use("dark_background")

        fig, axs = plt.subplots(
            nrows,
            ncolumns,
            height_ratios=heightratio,
            width_ratios=widthratio,
            frameon=False,
        )  # create number of panels

        # clean up axes, tick parameters
        if nrows * ncolumns == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw * 4)
            axs.yaxis.set_tick_params(width=lw, length=lw * 4)
            axs.tick_params(axis="both", pad=1.2)
        elif nrows * ncolumns == 2:
            for i in np.arange(2):
                axs[i].xaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].tick_params(axis="both", pad=1.2)
        elif nrows * ncolumns == len(widthratio):
            for i in np.arange(len(widthratio)):
                axs[i].xaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].yaxis.set_tick_params(width=lw, length=lw * 4)
                axs[i].tick_params(axis="both", pad=1.2)
        else:
            for i in np.arange(nrows):
                for j in np.arange(ncolumns):
                    axs[i, j].xaxis.set_tick_params(width=0.5, length=lw * 4)
                    axs[i, j].yaxis.set_tick_params(width=0.5, length=lw * 4)
                    axs[i, j].tick_params(axis="both", pad=1.2)
        return fig, axs

    def ternary_scatter_plot(
        self,
        fig,
        axs,
        R,
        G,
        B,
        colours,
        xlevel=1,
        ylevel=2,
        location_pos=2,
        maj_loc=0.2,
        min_loc=0.1,
        gridsize=100,
        bins=None,
        cmap="gist_gray",
        maxt=1,
        maxl=1,
        maxr=1,
        trianglesize=1,
        s=25,
        lws=0.5,
    ):
        """ternary_contour_plot function
        takes data and makes a ternary contour plot

        Args:
            fig (object): figure object
            R (np.1darray): R scatter point
            G (np.1darray): G scatter point
            B (np.1darray): B scatter point
            colours (np.2darray): rgba for each scatter point

        Returns:
            ax is axis object"""

        axs[1].remove()
        ax = fig.add_subplot(xlevel, ylevel, location_pos, projection="ternary")

        ax.taxis.set_major_locator(MultipleLocator(maj_loc))
        ax.laxis.set_major_locator(MultipleLocator(maj_loc))
        ax.raxis.set_major_locator(MultipleLocator(maj_loc))

        ax.taxis.set_minor_locator(MultipleLocator(min_loc))
        ax.laxis.set_minor_locator(MultipleLocator(min_loc))
        ax.raxis.set_minor_locator(MultipleLocator(min_loc))

        ax.set_ternary_lim(
            maxt - trianglesize,
            maxt,  # tmin, tmax
            maxl - trianglesize,
            maxl,  # lmin, lmax
            maxr - trianglesize,
            maxr,  # rmin, rmax
        )

        ax.set_tlabel(r"pixel 1 QE")
        ax.set_llabel(r"pixel 3 QE")
        ax.set_rlabel(r"pixel 2 QE")

        ax.grid(lw=0.5, alpha=0.25, ls="--", which="both", axis="both", color="white")
        ax.scatter(
            R,
            G,
            B,
            s=s,
            facecolors="None",
            edgecolors=colours,
            lw=lws,
            marker="o",
        )
        return fig, axs

    def ternary_contour_plot(
        self,
        fig,
        axs,
        t,
        l,
        r,
        R,
        G,
        B,
        maj_loc=0.2,
        min_loc=0.1,
        gridsize=100,
        bins=None,
        cmap="gist_gray",
        maxt=1,
        maxl=1,
        maxr=1,
        trianglesize=1,
        ecolour="red",
        s=25,
        lws=0.5,
    ):
        """ternary_contour_plot function
        takes data and makes a ternary contour plot

        Args:
            fig (object): figure object
            t (np.1darray): t data
            l (np.1darray): l data
            r (np.1darray): r data
            R (np.1darray): R scatter point
            G (np.1darray): G scatter point
            B (np.1darray): B scatter point

        Returns:
            ax is axis object"""

        axs[1].remove()
        ax = fig.add_subplot(2, 1, 2, projection="ternary")

        ax.taxis.set_major_locator(MultipleLocator(maj_loc))
        ax.laxis.set_major_locator(MultipleLocator(maj_loc))
        ax.raxis.set_major_locator(MultipleLocator(maj_loc))

        ax.taxis.set_minor_locator(MultipleLocator(min_loc))
        ax.laxis.set_minor_locator(MultipleLocator(min_loc))
        ax.raxis.set_minor_locator(MultipleLocator(min_loc))
        ax.hexbin(
            t,
            l,
            r,
            gridsize=gridsize,
            edgecolors="none",
            bins=bins,
            cmap=cmap,
            rasterized=True,
        )

        ax.set_ternary_lim(
            maxt - trianglesize,
            maxt,  # tmin, tmax
            maxl - trianglesize,
            maxl,  # lmin, lmax
            maxr - trianglesize,
            maxr,  # rmin, rmax
        )

        ax.set_tlabel(r"pixel 1 QE")
        ax.set_llabel(r"pixel 3 QE")
        ax.set_rlabel(r"pixel 2 QE")

        ax.grid(lw=0.5, alpha=0.25, ls="--", which="both", axis="both", color="white")
        ax.scatter(
            R,
            G,
            B,
            s=s,
            facecolors="None",
            edgecolors=ecolour,
            lw=lws,
            marker="o",
        )
        return fig, axs

    def line_plot(
        self,
        axs,
        x,
        y,
        xlim=None,
        ylim=None,
        color="k",
        lw=0.75,
        label="",
        xaxislabel="x axis",
        yaxislabel="y axis",
        ls="-",
        alpha=1,
    ):
        """line_plot function
        takes data and makes a line plot

        Args:
            x (np.1darray): x data
            y (np.1darray): y data
            xlim is x limits; default is None (which computes max/min)
            ylim is y limits; default is None (which computes max/min)
            color is line colour; default is black
            lw is line width (default 0.75)
            label is label; default is nothing
            xaxislabel is x axis label (default is 'x axis')
            yaxislabel is y axis label (default is 'y axis')

        Returns:
            axs is axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(x) * 0.9, np.max(x) * 1.1])
        if ylim is None:
            ylim = np.array([np.min(y) * 0.9, np.max(y) * 1.1])
        axs.plot(x, y, lw=lw, color=color, label=label, ls=ls, alpha=alpha)
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        if self.db == True:
            axs.grid(True, which="both", ls="--", c="white", lw=0.25, alpha=0.25)
        else:
            axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)
        return axs

    def line_error_plot(
        self,
        axs,
        x,
        y,
        yerror,
        xlim=None,
        ylim=None,
        color="k",
        lw=0.75,
        label="",
        xaxislabel="x axis",
        yaxislabel="y axis",
        ls="-",
        alpha=1.0,
    ):
        """line_plot function
        takes data and makes a line plot

        Args:
            x (np.1darray): x data
            y (np.1darray): y data
            yerror (np.1darray): y error data
            xlim is x limits; default is None (which computes max/min)
            ylim is y limits; default is None (which computes max/min)
            color is line colour; default is black
            lw is line width (default 0.75)
            label is label; default is nothing
            xaxislabel is x axis label (default is 'x axis')
            yaxislabel is y axis label (default is 'y axis')
            ls (str): line style
            alpha (float): alpha of error shading

        Returns:
            axs is axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(x) * 0.9, np.max(x) * 1.1])
        if ylim is None:
            ylim = np.array([np.min(y) * 0.9, np.max(y) * 1.1])
        axs.plot(x, y, lw=lw, color=color, label=label, ls=ls)
        axs.fill_between(x, y - yerror, y + yerror, color=color, alpha=alpha)
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        if self.db == True:
            axs.grid(True, which="both", ls="--", c="white", lw=0.25, alpha=0.25)
        else:
            axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)
        return axs

    def histogram_plot(
        self,
        axs,
        data,
        bins,
        xlim=None,
        ylim=None,
        histcolor="gray",
        xaxislabel="x axis",
        alpha=1,
        histtype="bar",
        density=True,
        label="",
    ):
        """histogram_plot function
        takes data and makes a histogram

        Args:
            axs (axis): axis object
            data (np.1darray): data array
            bins (np.1darray): bin array
            xlim (boolean or list of two floats): default is None (which computes min/max of x), otherwise provide a min/max
            ylim (boolean or list of two floats): default is None (which computes min/max of y), otherwise provide a min/max
            histcolor (string): histogram colour (default is gray)
            xaxislabel (string): x axis label (default is 'x axis')
            alpha (float): histogram transparency (default 1)
            histtype (string): histogram type, default bar
            density (boolean): if to plot as pdf, default True
            label (string): label for histogram

        Returns:
            axs (axis): axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(data), np.max(data)])

        axs.hist(
            data,
            bins=bins,
            density=density,
            color=histcolor,
            alpha=alpha,
            histtype=histtype,
            label=label,
        )
        if self.db == True:
            axs.grid(True, which="both", ls="--", c="white", lw=0.25, alpha=0.25)
        else:
            axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        if density == True:
            axs.set_ylabel("probability density", fontsize=fontsz)
        else:
            axs.set_ylabel("frequency", fontsize=fontsz)
        axs.set_xlim(xlim)
        if ylim is not None:
            axs.set_ylim(ylim)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        return axs

    def make_camera_pattern(
        self,
        ax,
        xsize,
        ysize,
        array,
        xlim=10,
        ylim=10,
        scatter=False,
        xscatter=0,
        yscatter=0,
        s=5,
        scolor="green",
    ):
        # Set the gridding interval: here we use the major tick interval
        import matplotlib.ticker as plticker

        myInterval = 1.0
        loc = plticker.MultipleLocator(base=myInterval)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        # Add the grid
        ax.grid(
            which="major",
            axis="both",
            linestyle="-",
            lw=(3 / 13) * xsize,
            color="white",
        )

        # Add the image
        ax.imshow(array, cmap="gist_gray")

        ax.set_ylim([0, xlim])
        ax.set_xlim([0, ylim])

        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        ax.tick_params("both", length=0, width=(2 / 13) * xsize, which="major")
        ax.tick_params("both", length=0, width=(1 / 13) * xsize, which="minor")

        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth((2 / 13) * xsize)

        for child in ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color("white")

        ystart = np.arange(0, 13)
        xstart = np.arange(0, 13)

        for yval in ystart:
            if not yval % 2:
                for xval in xstart:
                    if not xval % 2:
                        linepos = xval + np.arange(0.1, 1, 0.1)
                        for l in linepos:
                            ax.vlines(
                                x=l,
                                ymin=yval + 0,
                                ymax=yval + 1,
                                lw=(5 / 13) * xsize,
                                color="white",
                            )
            else:
                for xval in xstart:
                    if xval % 2:
                        linepos = xval + np.arange(0.1, 1, 0.1)
                        for l in linepos:
                            ax.vlines(
                                x=l,
                                ymin=yval + 0,
                                ymax=yval + 1,
                                lw=(5 / 13) * xsize,
                                color="white",
                            )

        xlin = np.linspace(0, 1, 1000)
        yscalingfactor = np.arange(-0.85, 1.05, 0.3)
        for yval in ystart:
            if yval % 2:
                for xval in xstart:
                    if not xval % 2:
                        for ys in yscalingfactor:
                            xlin_temp = xlin + xval
                            ylin = xlin + yval + ys
                            xlin_temp = xlin_temp[
                                (ylin > yval + 0.01) & (ylin < yval + 0.99)
                            ]
                            ylin = ylin[(ylin > yval + 0.01) & (ylin < yval + 0.99)]
                            ax.plot(xlin_temp, ylin, lw=(5 / 13) * xsize, color="white")
            else:
                for xval in xstart:
                    if xval % 2:
                        for ys in yscalingfactor:
                            xlin_temp = xlin + xval
                            ylin = (1 - xlin) + yval + ys
                            xlin_temp = xlin_temp[
                                (ylin > yval + 0.01) & (ylin < yval + 0.99)
                            ]
                            ylin = ylin[(ylin > yval + 0.01) & (ylin < yval + 0.99)]
                            ax.plot(xlin_temp, ylin, lw=(5 / 13) * xsize, color="white")

        if scatter is not False:
            ax.scatter(
                xscatter,
                yscatter,
                facecolor=scolor,
                edgecolors=None,
                lw=0,
                s=s,
                marker="o",
                zorder=np.inf,
            )
            ax.set_ylim([0, xlim])
            ax.set_xlim([0, ylim])
        return ax

    def scatter_plot(
        self,
        axs,
        x,
        y,
        xlim=None,
        ylim=None,
        label="",
        edgecolor="k",
        facecolor="white",
        s=5,
        lw=0.75,
        xaxislabel="x axis",
        yaxislabel="y axis",
        alpha=1,
        marker="o",
        rasterized=False,
    ):
        """scatter_plot function
        takes data and makes a scatter plot
        Args:
            x is x data
            y os y data
            xlim is x limits; default is None (which computes max/min)
            ylim is y limits; default is None (which computes max/min)
            label is label; default is nothing
            edgecolor is edge colour; default is black
            facecolor is face colour; default is white
            s is size of scatter point; default is 5
            lw is line width (default 0.75)
            xaxislabel is x axis label (default is 'x axis')
            yaxislabel is y axis label (default is 'y axis')
            alpha is alpha
            marker is marker
            rasterized will raster scatter points
        Returns:
            axs is axis object"""
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 7.0

        if xlim is None:
            xlim = np.array([np.min(x) * 0.9, np.max(x) * 1.1])
        if ylim is None:
            ylim = np.array([np.min(y) * 0.9, np.max(y) * 1.1])
        axs.scatter(
            x,
            y,
            s=s,
            edgecolors=edgecolor,
            facecolor=facecolor,
            lw=lw,
            label=label,
            alpha=alpha,
            marker=marker,
            rasterized=rasterized,
        )
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        if self.db == True:
            axs.grid(True, which="both", ls="--", c="white", lw=0.25, alpha=0.25)
        else:
            axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)
        return axs

    def contourf_plot(
        self,
        axs,
        X,
        Y,
        Z,
        levels=10,
        cmap="gist_gray",
        cbar="on",
        cbarlabel="photons",
        label="",
        labelcolor="white",
        alpha=1,
        xaxislabel="xaxislabel",
        yaxislabel="yaxislabel",
    ):
        """contourf function
        takes X, Y, Z and makes a contourf plot

        Args:
            axs (axis): axis object
            data (np.2darray): image
            vmin (float): minimum pixel intensity displayed (default 0.1%)
            vmax (float): minimum pixel intensity displayed (default 99.9%)
            cmap (string): colour map used; default gray)
            cbarlabel (string): colour bar label; default 'photons'
            label (string): is any annotation
            labelcolor (string): annotation colour
            pixelsize (float): pixel size in nm for scalebar, default 110
            scalebarsize (float): scalebarsize in nm, default 5000
            scalebarlabel (string): scale bar label, default 5 um

        Returns:
            axs (axis): axis object"""

        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        im = axs.contourf(X, Y, Z, levels=levels, cmap=cmap)

        if cbar == "on":
            cbar = plt.colorbar(im, fraction=0.045, pad=0.1, ax=axs, location="right")
            cbar.set_label(cbarlabel, rotation=90, labelpad=0.1, fontsize=fontsz)
            cbar.ax.tick_params(labelsize=fontsz - 1, pad=0.1, width=0.5, length=2)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)

        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=fontsz - 1,
        )

        return axs

    def image_plot(
        self,
        axs,
        data,
        vmin=None,
        vmax=None,
        cmap="gist_gray",
        cbar="on",
        cbarlabel="photoelectrons",
        label="",
        labelcolor="white",
        pixelsize=69,
        sbar="on",
        scalebarsize=10000,
        scalebarlabel=r"10$\,\mu$m",
        alpha=1,
        plotmask=False,
        mask=None,
        maskcolor="white",
    ):
        """image_plot function
        takes image data and makes an image plot

        Args:
            axs (axis): axis object
            data (np.2darray): image
            vmin (float): minimum pixel intensity displayed (default 0.1%)
            vmax (float): minimum pixel intensity displayed (default 99.9%)
            cmap (string): colour map used; default gray)
            cbarlabel (string): colour bar label; default 'photons'
            label (string): is any annotation
            labelcolor (string): annotation colour
            pixelsize (float): pixel size in nm for scalebar, default 110
            scalebarsize (float): scalebarsize in nm, default 5000
            scalebarlabel (string): scale bar label, default 5 um

        Returns:
            axs (axis): axis object"""

        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if vmin is None:
            vmin = np.percentile(data.ravel(), 0.1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )
        if cbar == "on":
            cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location="left")
            cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
            cbar.ax.tick_params(labelsize=fontsz - 1, pad=0.1, width=0.5, length=2)
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize / pixelsize
        scalebar = AnchoredSizeBar(
            axs.transData,
            pixvals,
            scalebarlabel,
            "lower right",
            pad=0.1,
            color=labelcolor,
            frameon=False,
            size_vertical=1,
        )
        if sbar == "on":
            axs.add_artist(scalebar)
        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=fontsz - 1,
        )

        if plotmask == True:
            axs.contour(mask, [0.5], linewidths=0.75, colors=maskcolor)

        return axs

    def make_animated_gif_bleaching(
        self,
        fig,
        axs,
        locations,
        trace_matrix,
        image_data,
        im_size,
        time,
        filename,
        width=3,
        vmin=0.1,
        vmax=99.9,
        pixelsize=69,
        scalebarsize=300,
        scalebarlabel="300 nm",
        fps=25,
        interval=40,
    ):

        from matplotlib.animation import FuncAnimation, PillowWriter
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        fp = FontProperties()
        fp.set_size(8)
        line = []
        im = []
        for i in np.arange(axs.shape[0]):
            for j in np.arange(axs.shape[1]):
                if i == 0:
                    i_l = im_size / 2
                    image_toplot = image_data[
                        :,
                        int(np.around(locations[0, j] - i_l)) : int(
                            np.around(locations[0, j] + i_l)
                        ),
                        int(np.around(locations[1, j] - i_l)) : int(
                            np.around(locations[1, j] + i_l)
                        ),
                    ]
                    im.append(
                        axs[i, j].imshow(
                            image_toplot[0, :, :],
                            cmap="gist_gray",
                            vmin=np.percentile(image_toplot, vmin),
                            vmax=np.percentile(image_data, vmax),
                        )
                    )
                    pixvals = scalebarsize / pixelsize
                    scalebar = AnchoredSizeBar(
                        axs[i, j].transData,
                        pixvals,
                        scalebarlabel,
                        "lower right",
                        pad=0.5,
                        color="white",
                        frameon=False,
                        size_vertical=(1 / width),
                        fontproperties=fp,
                    )
                    axs[i, j].add_artist(scalebar)
                    axs[i, j].axis("off")

                else:
                    line.append(
                        axs[i, j].plot(
                            time[0],
                            trace_matrix[j, 0],
                            color="white",
                            zorder=np.inf,
                            lw=0.5,
                        )[0]
                    )
                    axs[i, j].set_ylabel("photoelectrons/frame", fontsize=8)
                    axs[i, j].set_xlabel("time/s", fontsize=8)
                    axs[i, j].set_xlim([0, np.max(time)])
                    axs[i, j].set_ylim(
                        [np.min(trace_matrix[j, :]), 1.1 * np.max(trace_matrix[j, :])]
                    )

        def animate(k):
            for i in np.arange(axs.shape[0]):
                for j in np.arange(axs.shape[1]):
                    if i == 0:
                        axs[i, j].clear()
                        image_toplot = image_data[
                            :,
                            int(np.around(locations[0, j]) - i_l) : int(
                                np.around(locations[0, j]) + i_l
                            ),
                            int(np.around(locations[1, j]) - i_l) : int(
                                locations[1, j] + i_l
                            ),
                        ]
                        im[j] = axs[i, j].imshow(
                            image_toplot[k, :, :],
                            vmin=np.percentile(image_toplot, vmin),
                            vmax=np.percentile(image_toplot, vmax),
                            cmap="gist_gray",
                        )
                        pixvals = scalebarsize / pixelsize
                        scalebar = AnchoredSizeBar(
                            axs[i, j].transData,
                            pixvals,
                            scalebarlabel,
                            "lower right",
                            pad=0.5,
                            color="white",
                            frameon=False,
                            size_vertical=(0.5 / width),
                            fontproperties=fp,
                        )
                        axs[i, j].add_artist(scalebar)
                        axs[i, j].axis("off")
                    else:
                        line[j].set_xdata(time[:k])
                        line[j].set_ydata(trace_matrix[j, :k])
            return [*im, *line]

        ani = FuncAnimation(
            fig,
            animate,
            interval=interval,
            blit=True,
            repeat=True,
            frames=image_data.shape[0],
        )
        ani.save(
            filename,
            dpi=400,
            writer=PillowWriter(fps=fps),
            savefig_kwargs={"transparent": True},
        )
        return

    def make_animated_gif_multipanel(
        self,
        fig,
        axs,
        plot_types,
        xpositions,
        ypositions,
        image_data,
        n_pixels,
        n_frames,
        filename,
        s=150,
        scolors=["#99ff99", "#ffc04d", "#ff6666"],
        width=3,
        height=3,
        xlim_scatter=4,
        ylim_scatter=4,
        vmin=0.1,
        vmax=99.9,
        pixelsize=69,
        scalebarsize=300,
        scalebarlabel="300 nm",
    ):

        from matplotlib.animation import FuncAnimation, PillowWriter
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        fp = FontProperties()
        fp.set_size(8)

        scat = []
        im = []
        for i in np.arange(axs.shape[0]):
            for j in np.arange(axs.shape[1]):
                if plot_types[i, j] == "pattern":
                    image = np.zeros([n_pixels, n_pixels])
                    axs[i, j] = self.make_camera_pattern(
                        ax=axs[i, j],
                        xsize=width,
                        ysize=height,
                        array=image,
                        xlim=xlim_scatter,
                        ylim=ylim_scatter,
                        scatter=False,
                    )

                    scat.append(
                        axs[i, j].scatter(
                            xpositions[0],
                            ypositions[0],
                            edgecolor=None,
                            facecolor=scolors[j],
                            s=s,
                            zorder=np.inf,
                        )
                    )

                    pixvals = 100 / pixelsize
                    scalebar = AnchoredSizeBar(
                        axs[i, j].transData,
                        pixvals,
                        "100 nm",
                        "lower center",
                        pad=0.5,
                        color="white",
                        frameon=False,
                        size_vertical=(0.3 / width),
                        fontproperties=fp,
                    )
                    axs[i, j].add_artist(scalebar)
                    axs[i, j].axis("off")

                elif plot_types[i, j] == "image":
                    im.append(
                        axs[i, j].imshow(
                            image_data[j, 0, :, :],
                            cmap="gist_gray",
                            vmin=np.percentile(image_data[j, :, :, :], vmin),
                            vmax=np.percentile(image_data[j, :, :, :], vmax),
                        )
                    )
                    pixvals = scalebarsize / pixelsize
                    scalebar = AnchoredSizeBar(
                        axs[i, j].transData,
                        pixvals,
                        scalebarlabel,
                        "lower center",
                        pad=0.5,
                        color="white",
                        frameon=False,
                        size_vertical=(0.5 / width),
                        fontproperties=fp,
                    )
                    axs[i, j].add_artist(scalebar)
                    axs[i, j].axis("off")
                else:
                    axs[i, j].axis("off")

        def animate(k):
            for i in np.arange(axs.shape[0]):
                for j in np.arange(axs.shape[1]):
                    if plot_types[i, j] == "pattern":
                        x = xpositions[k]
                        y = ypositions[k]
                        data = np.stack([x, y]).T
                        scat[j].set_offsets(data)
                    elif plot_types[i, j] == "image":
                        axs[i, j].clear()
                        im[j] = axs[i, j].imshow(
                            image_data[j, k, :, :],
                            vmin=np.percentile(image_data[j, :, :, :], vmin),
                            vmax=np.percentile(image_data[j, :, :, :], vmax),
                            cmap="gist_gray",
                        )
                        pixvals = scalebarsize / pixelsize
                        scalebar = AnchoredSizeBar(
                            axs[i, j].transData,
                            pixvals,
                            scalebarlabel,
                            "lower center",
                            pad=0.5,
                            color="white",
                            frameon=False,
                            size_vertical=(1 / width),
                            fontproperties=fp,
                        )
                        axs[i, j].add_artist(scalebar)
                        axs[i, j].axis("off")

            return [*im, *scat]

        ani = FuncAnimation(
            fig, animate, interval=40, blit=True, repeat=True, frames=n_frames
        )
        ani.save(
            filename,
            dpi=400,
            writer=PillowWriter(fps=25),
            savefig_kwargs={"transparent": True},
        )
        return

    def make_animated_gif_plot(
        self,
        xpositions,
        ypositions,
        n_pixels,
        n_frames,
        filename,
        s=25,
        scolor="green",
        width=3,
        height=3,
        xlim=10,
        ylim=10,
    ):

        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = self.one_column_plot(width=width, height=height)

        image = np.zeros([n_pixels, n_pixels])

        ax = self.make_camera_pattern(
            ax=ax,
            xsize=width,
            ysize=height,
            array=image,
            xlim=xlim,
            ylim=ylim,
            scatter=False,
        )

        scat = ax.scatter(
            xpositions[0],
            ypositions[0],
            edgecolor=None,
            facecolor=scolor,
            s=s,
            zorder=np.inf,
        )

        def animate(i):
            x = xpositions[i]
            y = ypositions[i]
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            return [scat]

        ani = FuncAnimation(
            fig, animate, interval=25, blit=True, repeat=True, frames=n_frames
        )
        ani.save(
            filename,
            dpi=400,
            writer=PillowWriter(fps=25),
            savefig_kwargs={"transparent": True},
        )
        return

    def make_animated_gif_image(
        self,
        image,
        n_frames,
        filename,
        vmin=0,
        vmax=150,
        pixelsize=69,
        scalebarsize=300,
        scalebarlabel="300 nm",
        label="",
        fontsz=6,
        cbarlabel="# of photoelectrons",
        cbar=False,
        width=3,
        height=3,
    ):

        from matplotlib.animation import FuncAnimation, PillowWriter
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        fig, ax = self.one_column_plot(width=width, height=height)

        divider = make_axes_locatable(ax)

        im = ax.imshow(image[0, :, :], cmap="gist_gray", vmin=vmin, vmax=vmax)
        if cbar == True:
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, orientation="vertical", cax=cax)
            cbar.set_label(cbarlabel, rotation=270, labelpad=8, fontsize=7)
        xy_coord = int(image.shape[0] * 0.05)

        def animate(i):
            ax.clear()
            im = ax.imshow(image[i, :, :], vmin=vmin, vmax=vmax, cmap="gist_gray")
            pixvals = scalebarsize / pixelsize
            scalebar = AnchoredSizeBar(
                ax.transData,
                pixvals,
                scalebarlabel,
                "lower right",
                pad=0.5,
                color="white",
                frameon=False,
                size_vertical=(1 / width),
            )
            ax.add_artist(scalebar)
            ax.annotate(
                label,
                xy=(xy_coord, xy_coord),
                xytext=(xy_coord, xy_coord),
                xycoords="data",
                color="white",
                fontsize=fontsz + 1,
            )
            ax.axis("off")
            return [im]

        ani = FuncAnimation(
            fig, animate, interval=25, blit=True, repeat=True, frames=n_frames
        )
        ani.save(
            filename,
            dpi=400,
            writer=PillowWriter(fps=25),
            savefig_kwargs={"transparent": True},
        )
        return

    def image_scatter_plot(
        self,
        axs,
        data,
        xdata,
        ydata,
        vmin=None,
        vmax=None,
        cmap="gist_gray",
        cbar="on",
        cbarlabel="photons",
        label="",
        labelcolor="white",
        pixelsize=110,
        scalebarsize=10000,
        scalebarlabel=r"10$\,\mu$m",
        alpha=1,
        scatteralpha=1,
        scattercolor="red",
        s=20,
        lws=0.75,
    ):
        """image_plot function
        takes image data and makes an image plot

        Args:
            axs (axis): axis object
            data (np.2darray): image
            xdata (np.1darray): scatter points, x
            ydata (np.1darray): scatter points, y
            vmin (float): minimum pixel intensity displayed (default 0.1%)
            vmax (float): minimum pixel intensity displayed (default 99.9%)
            cmap (string): colour map used; default gray)
            cbarlabel (string): colour bar label; default 'photons'
            label (string): is any annotation
            labelcolor (string): annotation colour
            pixelsize (float): pixel size in nm for scalebar, default 110
            scalebarsize (float): scalebarsize in nm, default 5000
            scalebarlabel (string): scale bar label, default 5 um

        Returns:
            axs (axis): axis object"""

        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if vmin is None:
            vmin = np.percentile(data.ravel(), 0.1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )
        if cbar == "on":
            cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location="left")
            cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
            cbar.ax.tick_params(labelsize=fontsz - 1, pad=0.1, width=0.5, length=2)
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize / pixelsize
        scalebar = AnchoredSizeBar(
            axs.transData,
            pixvals,
            scalebarlabel,
            "lower right",
            pad=0.1,
            color=labelcolor,
            frameon=False,
            size_vertical=1,
        )

        axs.add_artist(scalebar)
        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=fontsz - 1,
        )
        axs.scatter(
            ydata,
            xdata,
            lw=lws,
            edgecolor=scattercolor,
            s=s,
            facecolors="None",
            alpha=scatteralpha,
        )
        return axs

    # def image_bayer_plot(
    #     self,
    #     axs,
    #     data,
    #     vmin=None,
    #     vmax=None,
    #     cmap="gist_gray",
    #     cbar="on",
    #     cbarlabel="photons",
    #     label="",
    #     labelcolor="white",
    #     pixelsize=110,
    #     scalebarsize=10000,
    #     scalebarlabel=r"10$\,\mu$m",
    #     alpha=0.3,
    #     masks=None,
    # ):
    #     """image_plot function
    #     takes image data and makes an image plot

    #     Args:
    #         axs (axis): axis object
    #         data (np.2darray): image
    #         vmin (float): minimum pixel intensity displayed (default 0.1%)
    #         vmax (float): minimum pixel intensity displayed (default 99.9%)
    #         cmap (string): colour map used; default gray)
    #         cbarlabel (string): colour bar label; default 'photons'
    #         label (string): is any annotation
    #         labelcolor (string): annotation colour
    #         pixelsize (float): pixel size in nm for scalebar, default 110
    #         scalebarsize (float): scalebarsize in nm, default 5000
    #         scalebarlabel (string): scale bar label, default 5 um
    #         masks (dict): bayer masks

    #     Returns:
    #         axs (axis): axis object"""

    #     fontsz = 15 if self.poster else 8
    #     axs.patch.set_alpha(0.0)
    #     # Set xlim, ylim if not provided
    #     vmin = vmin or np.percentile(data.ravel(), 0.1)
    #     vmax = vmax or np.percentile(data.ravel(), 99.9)

    #     overlay = np.zeros((data.shape[0], data.shape[1], 3))  # RGB array
    #     overlay[masks["R"]] = [1, 0, 0]  # Red
    #     overlay[masks["G"]] = [0, 1, 0]  # Green
    #     overlay[masks["B"]] = [0, 0, 1]  # Blue

    #     from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    #     im = axs.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=1, origin="lower")
    #     axs.imshow(overlay, origin="lower", alpha=alpha)
    #     if cbar == "on":
    #         cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location="left")
    #         cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
    #         cbar.ax.tick_params(labelsize=fontsz - 1, pad=0.1, width=0.5, length=2)
    #     axs.set_xticks([])
    #     axs.set_yticks([])
    #     pixvals = scalebarsize / pixelsize
    #     scalebar = AnchoredSizeBar(
    #         axs.transData,
    #         pixvals,
    #         scalebarlabel,
    #         "lower right",
    #         pad=0.1,
    #         color=labelcolor,
    #         frameon=False,
    #         size_vertical=1,
    #     )

    #     axs.add_artist(scalebar)
    #     axs.annotate(
    #         label,
    #         xy=(5, 5),
    #         xytext=(20, 60),
    #         xycoords="data",
    #         color=labelcolor,
    #         fontsize=fontsz - 1,
    #     )

    #     return axs
