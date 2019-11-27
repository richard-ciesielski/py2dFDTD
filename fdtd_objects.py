"""Object library for 2D FDTD with YEE-Algorithmus: leapfrog and staggered grid
according to Taflove's book
- in vacuum
- BC: PEC and PML
- TE-mode

Principal units are micrometers (1um = 1e-6m) and femtoseconds (1fs = 1e-15s)
The resulting speed of light in vacuum is c0 = 0.3 um/fs

(c) Richard Ciesielski, 2009-2019
"""

from __future__ import division

__version__ = '0.5'
__author__ = 'Richard Ciesielski'

import numpy


def addPlasmonicCircle(grid, x, y, R, wp, wj, gamma, sigma):
    """adds a circular area of uniform material constants to a grid
        wp - plasma frequency
        wj - eigenfrequency of the Lorentz pole
        gamma - damping
        sigma - conductivity"""
    i0, j0 = grid.getIndex(x, y)
    i1, j1 = grid.getIndex(x-R, y+R)
    i2, j2 = grid.getIndex(x+R, y-R)
    
    for i in numpy.arange(i1, i2):
        for j in numpy.arange(j1, j2):
            x1 = grid.mesh_x[i, j]
            y1 = grid.mesh_y[i, j]

            if R**2 > (x-x1)**2 + (y-y1)**2:
                grid.wp[i, j] = wp
                grid.wj[i, j] = wj
                grid.gamma[i, j] = gamma
                grid.sigma[i, j] = sigma
                
                
    return grid
    
def addDielectricCircle(grid, x, y, R, n):
    """adds a circular area of uniform material constants to a grid
        n - refractive index, assuming mu==1"""
    eps = n**2
    
    i0, j0 = grid.getIndex(x, y)
    i1, j1 = grid.getIndex(x-R, y+R)
    i2, j2 = grid.getIndex(x+R, y-R)
    
    for i in numpy.arange(i1, i2):
        for j in numpy.arange(j1, j2):
            x1 = grid.mesh_x[i, j]
            y1 = grid.mesh_y[i, j]

            if R**2 > (x-x1)**2 + (y-y1)**2:
                grid.Eps[i, j]*= eps                
                
    return grid
    
    
def addPlasmonicRectangle(grid, x0, y0, x1, y1, wp, wj, gamma, sigma, R=0):
    """adds a rectangular area of uniform material constants to a grid
        wp - plasma frequency
        wj - eigenfrequency of the Lorentz pole
        gamma - damping
        sigma - conductivity
        R - radius for rounded edges"""
    # bring (x0,y0) & (y0,y1) in the correct order:
    if x0 > x1:
        a = x1 * 1.
        x1 = x0 * 1.
        x0 = a * 1.
    if y0 < y1:
        a = y1 * 1.
        y1 = y0 * 1.
        y0 = a * 1.
        
    i0, j0 = grid.getIndex(x0, y0)
    i1, j1 = grid.getIndex(x1, y1)
    
    grid.wp[i0:i1, j0:j1] = wp
    grid.wj[i0:i1, j0:j1] = wj
    grid.gamma[i0:i1, j0:j1] = gamma
    grid.sigma[i0:i1, j0:j1] = sigma

    if R!=0:
        # cut the edges:
        grid =  addRectangle(grid, x0, y0, x0+R,y0-R, 0, 0, 0, 0, 0)
        grid =  addRectangle(grid, x1-R, y0, x1,y0-R, 0, 0, 0, 0, 0)
        grid =  addRectangle(grid, x0, y1+R, x0+R,y1, 0, 0, 0, 0, 0)
        grid =  addRectangle(grid, x1-R, y1+R, x1,y1, 0, 0, 0, 0, 0)
        # round the edges:
        grid =  addCircle(grid, x0+R, y0-R, R, wp, wj, gamma, sigma)
        grid =  addCircle(grid, x1-R, y0-R, R, wp, wj, gamma, sigma)
        grid =  addCircle(grid, x0+R, y1+R, R, wp, wj, gamma, sigma)
        grid =  addCircle(grid, x1-R, y1+R, R, wp, wj, gamma, sigma)
        
    return grid
   
def addPlasmonicTriangle(grid, x0, y0, x1, y1, wp, wj, gamma, sigma):
    """adds a triangular area of uniform material constants to a grid
        wp - plasma frequency
        wj - eigenfrequency of the Lorentz pole
        gamma - damping
        sigma - conductivity
        R - radius for rounded edges"""
    i0, j0 = grid.getIndex(x0, y0)
    i1, j1 = grid.getIndex(x1, y1)
    
    #i = numpy.arange(i0, i1, numpy.sign(i1 - i0))
    #j = numpy.arange(j0, j1, numpy.sign(j1 - j0))
    
    #ij = ((i[:,newaxis]*ones((len(i),len(j)))/ abs(i1 - i0)) > ( j[newaxis,:]*ones((len(i),len(j)))/abs(j1 - j0)))
    
    for i in numpy.arange(i0, i1, numpy.sign(i1 - i0)):
        for j in numpy.arange(j0, j1, numpy.sign(j1 - j0)):
            if abs((i - i0) / (i1 - i0)) > abs((j - j0) / (j1 - j0)):   
                grid.wp[i, j] = wp
                grid.wj[i, j] = wj
                grid.gamma[i, j] = gamma
                grid.sigma[i, j] = sigma

    return grid
