"""Data-vector plots shared by the EXAMPLE_EVALUATE notebooks.

Triangle plots (one panel per tomographic bin pair) of angular power
spectra and of the real-space shear correlation functions. Pure
matplotlib and numpy: nothing here touches CAMB or the compiled
cosmolike interface, so every project shares these functions
unchanged. Figure styling (fonts, usetex, rcParams) stays in the
notebooks; these functions only build the figures.
"""

import math
import itertools
import warnings

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def plot_C_ss_tomo_limber(ell, C_ss, C_ss_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, colorbarshrink=0.3,
                          cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                          legend = None, legendloc = (0.6,0.78), yaxislabelsize = 12, yaxisticklabelsize = 10, 
                          xaxisticklabelsize = 6, bintextpos = [0.2, 0.85], bintextsize = 13, figsize = (18, 18), 
                          show = 1, colorbar=1, wspace=0.25, hspace=0.05):
    """Triangle plot of tomographic angular power spectra.

    One panel per tomographic bin pair (i, j), lower triangle only.
    Without C_ss_ref each curve is ell(ell+1) C_ell / 2 pi on a log
    scale; with C_ss_ref each curve is the fractional difference
    C / C_ref - 1 on a linear scale.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_ss entry.
      C_ss     = list of 3D arrays (n_ell, n_bins, n_bins), one per
                 curve.
      C_ss_ref = None, or one 3D array used as the ratio reference.
      param    = list of parameter values (one per curve) coloring
                 the curves and the colorbar, or None.
      colorbarlabel = colorbar label (LaTeX string), or None.
      lmin, lmax    = x-axis range in ell.
      ylim     = without C_ss_ref, multipliers on each panel's
                 min/max; with it, the band around 1 (drawn as
                 ylim - 1).
      linestyle, linewidth = lists cycled across curves, or None.
      legend   = one label per curve, or None; legendloc places it.
      cmap, colorbarshrink, yaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize, bintextpos, bintextsize, figsize, wspace,
      hspace = matplotlib layout knobs.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """

    # each C_ss entry is a 3D array; unpacking .shape reads its three
    # sizes (number of ell values, bins, bins) in one line
    nell, ntomo, ntomo2 = C_ss[0].shape
    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
      
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    if not (C_ss_ref is None):
        nell2, ntomo3, ntomo4 = C_ss_ref.shape
        if (ntomo3 != ntomo4) or (nell != nell2):
            print(f"notomo = {ntomo}, ntomo_REF = {ntomo3}")
            print(f"Nell = {nell}, Nell_REF = {nell2}")
            return 0   
        
    if C_ss_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': wspace, 'hspace': hspace})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 18, weight = 'bold', labelpad = 2)
        if len(param) != len(C_ss):
            print("Bad Input")
            return 0

    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following style, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])

    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
    
    # axes is the 2D array of panels plt.subplots returned; [j, i]
    # is row j, column i, and the upper triangle is switched off
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                clmin = []
                clmax = []
                for Cl in C_ss:  
                    tmp = ell * (ell + 1) * Cl[:,i,j] / (2 * math.pi)
                    clmin.append(np.min(tmp))
                    clmax.append(np.max(tmp))
     
                axes[j,i].set_xlim([lmin, lmax])
                
                if C_ss_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(clmin)), np.max(ylim[1]*np.array(clmax))])
                    axes[j,i].set_yscale('log')
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                    axes[j,i].set_yscale('linear')
                    
                axes[j,i].set_xscale('log')
                
                if i == 0:
                    if C_ss_ref is None:
                        axes[j,i].set_ylabel(r"$\ell (\ell+1) C_{\ell}^{EE}/(2 \pi)$", fontsize=yaxislabelsize)
                    else:
                        axes[j,i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)
                
                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\ell$", fontsize=16)
                
                axes[j,i].text(bintextpos[0], bintextpos[1], 
                    "$(" +  str(i) + "," +  str(j) + ")$", 
                    horizontalalignment = 'center', 
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    usetex = True,
                    transform = axes[j,i].transAxes)
                
                for x, Cl in enumerate(C_ss):
                    if C_ss_ref is None:
                        tmp = ell * (ell + 1) * Cl[:,i,j] / (2 * math.pi)
                    else:
                        tmp = Cl[:,i,j] / C_ss_ref[:,i,j] - 1
                    lines = axes[j,i].plot(ell, tmp, 
                                           color=cm(x/len(C_ss)), 
                                           linewidth=next(linewidthcycler), 
                                           linestyle=next(linestylecycler))
    
    if not (legend is None):
        if len(legend) != len(C_ss):
            print("Bad Input")
            return 0
        fig.legend(
            legend, 
            loc=legendloc,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)

    # the with-block silences warnings only inside it: fig.show()
    # outside a notebook warns about non-interactive backends
    if not (show is None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.show()
    else:
        return (fig, axes)

def plot_xi(pm, xi, xi_ref = None, param = None, colorbarlabel = None, marker = None, colorbarshrink=0.3,
                linestyle = None, linewidth = None, ylim = [0.88,1.12], 
                cmap = 'gist_rainbow', legend = None, legendloc = (0.6,0.78), yaxislabelsize = 10, 
                yaxisticklabelsize = 6, xaxisticklabelsize = 20, bintextpos = [[0.8, 0.875],[0.2,0.875]], 
                bintextsize = 15, figsize = (18, 18), show = 1, thetashow=[3,250], colorbar=1, wspace=0.25,hspace=0.05):
    """Triangle plot of the real-space shear correlation functions.

    One panel per tomographic bin pair (i, j), lower triangle only.
    pm > 0 plots xi_plus, pm <= 0 plots xi_minus. Without xi_ref each
    curve is theta * xi * 10^4; with xi_ref each curve is the
    fractional difference xi / xi_ref - 1.

    Arguments:
      pm     = +1 for xi_plus, -1 for xi_minus.
      xi     = list of (theta, xi_plus, xi_minus) tuples, one per
               curve, as the notebook xi wrapper returns them; the
               xi arrays are 3D (n_theta, n_bins, n_bins).
      xi_ref = None, or one such tuple used as the ratio reference.
      param  = list of parameter values (one per curve) coloring the
               curves and the colorbar, or None.
      colorbarlabel = colorbar label (LaTeX string), or None.
      marker = list of matplotlib markers cycled across curves
               (points instead of lines), or None for lines.
      linestyle, linewidth = lists cycled across curves, or None.
      ylim   = without xi_ref, multipliers on each panel's min/max;
               with it, the band around 1 (drawn as ylim - 1).
      thetashow = x-axis range in arcmin.
      legend = one label per curve, or None; legendloc places it.
      cmap, colorbarshrink, yaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize, bintextpos, bintextsize, figsize, wspace,
      hspace = matplotlib layout knobs.
      show   = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """
    
    # unpack the first curve's tuple to size the panels; every other
    # entry must share the same theta grid and bin count
    (theta, xip, xim) = xi[0]
    (ntheta, ntomo, ntomo2) = xip.shape    

    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
            
    if ntheta != len(theta):
        print("Bad Input (theta)")
        return 0

    if xi_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': wspace, 'hspace': hspace}
        )
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0.0, 'hspace': 0.0}
        )    

    cm = plt.get_cmap(cmap)

    if not (param is None or colorbar is None):
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = colorbarshrink
        )
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(xi):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
    
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
        
    # axes is the 2D array of panels plt.subplots returned; [j, i]
    # is row j, column i, and the upper triangle is switched off
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                ximin = []
                ximax = []
                for (theta, xip, xim) in xi:
                    if pm > 0:
                        ximin.append(np.min(theta*xip[:,i,j]*10**4))
                        ximax.append(np.max(theta*xip[:,i,j]*10**4))
                    else:
                        ximin.append(np.min(theta*xim[:,i,j]*10**4))
                        ximax.append(np.max(theta*xim[:,i,j]*10**4))
                        
                axes[j,i].set_xlim(thetashow)
                
                if xi_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)), np.max(ylim[1]*np.array(ximax))])
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('linear')
                
                if i == 0:
                    if xi_ref is None:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"$\theta \xi_{+} \times 10^4$", fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"$\theta \xi_{-} \times 10^4$", fontsize=yaxislabelsize)
                    else:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{+})$", fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{-})$", fontsize=yaxislabelsize)

                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)

                if pm > 0:
                    axes[j,i].text(bintextpos[0][0], 
                                   bintextpos[0][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=bintextsize,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)
                else:
                    axes[j,i].text(bintextpos[1][0], 
                                   bintextpos[1][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=15,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)

                if xi_ref is None:
                    for x, (theta, xip, xim) in enumerate(xi):
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, theta*xip[:,i,j]*10**4, color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, theta*xip[:,i,j]*10**4, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), linestyle='None', markersize=3)
                        else:
                            if marker is None:   
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, color=cm(x/len(xi)), 
                                    linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), linestyle='None', markersize=3)
                else:
                    (theta_ref, xip_ref, xim_ref) = xi_ref
                    for x, (theta, xip, xim) in enumerate(xi):
                        if theta != theta_ref:
                            print("inconsistent theta bins")
                            return 0
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), markerfacecolor='None',
                                               marker=next(markercycler),  markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)
                        else:
                            if marker is None:   
                                lines = axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                                       linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)    
    if not (legend is None):
        if len(legend) != len(xi):
            print("Bad Input")
            return 0
        fig.legend(legend, 
                   loc=legendloc,
                   borderpad=0.1,
                   handletextpad=0.4,
                   handlelength=1.5,
                   columnspacing=0.35,
                   scatteryoffsets=[0],
                   frameon=False)  
    # the with-block silences warnings only inside it: fig.show()
    # outside a notebook warns about non-interactive backends
    if not (show is None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.show()
    else:
        return (fig, axes)