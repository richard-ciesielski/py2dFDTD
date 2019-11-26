"""Plotting routines for 2D FDTD with YEE-Algorithmus: leapfrog and staggered grid
according to Taflove's book
- in vacuum
- BC: PEC and PML
- TE-mode

Principal units are micrometers (1um = 1e-6m) and femtoseconds (1fs = 1e-15s)
The resulting speed of light in vacuum is c0 = 0.3 um/fs

(c) Richard Ciesielski, 2009-2019
"""

__version__ = '0.5'
__author__ = 'Richard Ciesielski'

import numpy
import pylab
from numpy import pi, inf

def cut_infinities(grid, excitations, pm=5):
    """cuts of high values at the center of point sources"""
    for ex in excitations:
        if ex.type == "LineDipole2DHz":
            i, j = grid.getIndex(ex.x, ex.y)
            grid.Ex[i-pm:i+pm, j-pm:j+pm] = 0
            grid.Ey[i-pm:i+pm, j-pm:j+pm] = 0
            grid.Jx[i-pm:i+pm, j-pm:j+pm] = 0 # necessary?
            grid.Jy[i-pm:i+pm, j-pm:j+pm] = 0
            grid.Hz[i-pm:i+pm, j-pm:j+pm] = 0

def plot_matrix(m, grid, plotPML=-1, xlim=(-inf,inf), ylim=(-inf,inf)):
    # there could be inf....
    xlim = list(xlim)
    ylim = list(ylim)
    if abs(xlim[0])==inf: xlim[0] = grid.mesh_x.min()
    if abs(xlim[1])==inf: xlim[1] = grid.mesh_x.max()
    if abs(ylim[0])==inf: ylim[0] = grid.mesh_y.min()
    if abs(ylim[1])==inf: ylim[1] = grid.mesh_y.max()

    """plots a matrix as a color diagram"""
    if plotPML == 1:
        i0, i1, j0, j1 = 0, -1, 0, -1
    else:
        """cut off parts with a PML"""
        section=[0, -1, 0, -1]
        il = numpy.where(grid.d_PML_x[:, int(grid.Nx/2)]==0)
        jl = numpy.where(grid.d_PML_y[int(grid.Ny/2), :]==0)
        i0, i1, j0, j1 = [il[0][0], il[0][-1], jl[0][0], jl[0][-1]]

    x0, x1 = grid.mesh_x[0, j0], grid.mesh_x[0, j1]
    y0, y1 = grid.mesh_y[j0, 0], grid.mesh_y[i1, 0]
    
    # include the external limits
    x0 = max(x0, xlim[0])
    x1 = min(x1, xlim[1])
    y0 = max(y0, ylim[0])
    y1 = min(y1, ylim[1])
    
    # now refresh the indicees
    j0 = (abs(grid.mesh_x[0,:] - x0)).argmin()
    j1 = (abs(grid.mesh_x[0,:] - x1)).argmin()
    i0 = (abs(grid.mesh_y[:,0] - y0)).argmin()
    i1 = (abs(grid.mesh_y[:,0] - y1)).argmin()

    return pylab.imshow(m[i0:i1, j0:j1],
                            interpolation="nearest",extent=(x0, x1, y0, y1) )
    
def plot_vectorfield(x, y, di, grid, plotPML=-1):
    """plots a vectorfield"""
    if plotPML == 1:
        i0, j0 = 0, 0
        i1, j1 = min(numpy.shape(x), numpy.shape(y),
              numpy.shape(grid.mesh_x), numpy.shape(grid.mesh_y)) - numpy.array([1, 1])
        
    else:
        """cut off parts with a PML"""
        section=[0, -1, 0, -1]
        il = numpy.where(grid.d_PML_x[:, int(grid.Nx/2)]==0)
        jl = numpy.where(grid.d_PML_y[int(grid.Ny/2), :]==0)
        i0, i1, j0, j1 = [il[0][0], il[0][-1], jl[0][0], jl[0][-1]]
        
    return pylab.quiver(grid.mesh_x[numpy.arange(i1, i0, -di),j0:j1:di],
                                grid.mesh_y[numpy.arange(i1, i0, -di),j0:j1:di],
                                x[i0:i1:di,j0:j1:di], y[i0:i1:di,j0:j1:di])
    
def update_matrix_plot(plt, m, grid, plotPML=-1):
    """updates the plot of a matrix as a color diagram"""
    if plotPML == 1:
        i0, i1, j0, j1 = 0, -1, 0, -1
    else:
        """cut off parts with a PML"""
        section=[0, -1, 0, -1]
        il = numpy.where(grid.d_PML_x[:, int(grid.Nx/2)]==0)
        jl = numpy.where(grid.d_PML_y[int(grid.Ny/2), :]==0)
        i0, i1, j0, j1 = [il[0][0], il[0][-1], jl[0][0], jl[0][-1]]
        
    x0, x1 = grid.mesh_x[0, j0], grid.mesh_x[0, j1]
    y0, y1 = grid.mesh_y[j0, 0], grid.mesh_y[i1, 0]
    
    plt.set_data(m[i0:i1, j0:j1])
    
    return plt
