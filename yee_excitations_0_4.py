"""Excitation library for 2D FDTD with YEE-Algorithmus: leapfrog and staggered 
    grid according to Taflove's book
    - in vacuum
    - BC: PEC and PML
    - TE-mode
.....................................................................
Institut fuer Angewandte Photophysik, Technische Universitaet Dresden
Institute home page: http://www.iapp.de
Erstellt im Rahmen einer Belegarbeit 2009/10 von Richard Ciesielski
email: Richard.Ciesielski@gmail.com
....................................................................."""

import numpy
from numpy import pi

class PWleft_Exc(object):
    """A plane wave excitation from the left side"""
    def __init__(self, H, T, i0=0, i1=-1, t_end=100000, a=0):
        self.type = "PWleft_Exc"
        self.H = H
        self.T = T
        self.i0 = i0
        self.i1 = i1
        self.t_end = t_end
        self.a = a
        
    def excite(self, grid, t):
        # unnormalized excitation (sine):
        i, j = grid.getIndex(self.a, 0)
        w = 2 * pi / self.T
        t0 = self.T
        
        if t <= t0:
            grid.Ey[self.i0:self.i1, j+1] = grid.Ey[self.i0:self.i1, j+1] +\
                grid.dt * self.H / 2 * w * numpy.cos(w * t) * \
                (1+numpy.cos(pi+pi*t/t0))

        elif t0 < t < self.t_end - t0:
            grid.Ey[self.i0:self.i1, j+1] = grid.Ey[self.i0:self.i1, j+1] +\
                grid.dt * w * self.H * numpy.cos(w * t)
            
        elif self.t_end - t0 < t < self.t_end:
            t_ = t - self.t_end + 2.*t0
            grid.Ey[self.i0:self.i1, j+1] = grid.Ey[self.i0:self.i1, j+1] +\
                grid.dt * self.H / 2 * w * numpy.cos(w * t) * \
                (1+numpy.cos(pi+pi*t_/t0))
 
        return grid

class LineDipole2DHz_Exc(object):
    """A line dipole which excites Hz"""
    def __init__(self, x, y, H, T):
        self.type = "LineDipole2DHz"
        self.x = x
        self.y = y
        self.H = H
        self.T = T
        
    def excite(self, grid, t):
        i, j = grid.getIndex(self.x, self.y)
        # unnormalized excitation (sine):
        grid.Hz[i, j] = self.H * self.__ramp(t) * numpy.sin(2 * pi / self.T * t)
        
        return grid
    def __ramp(self, t):
        """Creates a smooth ramp function in time"""
        t0 = .1
        if t < t0:
            return .5 * (1 + numpy.cos(pi*(1 + t/t0)))
        else:
           return 1

class testpulse_Exc(object):
    """instantaneous gaussian test pulse"""
    def __init__(self, x, y, H=1, T=1, sigma=1, rc=2):
        self.type = "testpulse"
        self.x = x
        self.y = y
        self.H = H
        self.T = T                          # time to set it
        self.rc = rc                        # cutoff radius
        self.sigma = sigma
        
    def excite(self, grid, t):
        if t == 0:
            i0, j0 = grid.getIndex(self.x - self.rc, self.y + self.rc)
            i1, j1 = grid.getIndex(self.x + self.rc, self.y - self.rc)
            im, jm = grid.getIndex(self.x, self.y)
            r0 = (jm - j0)**2               # squared "radius" of indicees
            
            for i in numpy.arange(i0, i1):
                for j in numpy.arange(j0, j1):
                    
                    if (i - im)**2 + (j - jm)**2 < r0:
                        r = (grid.mesh_y[i, 0] - self.y)**2 +\
                            (grid.mesh_x[0, j] - self.x)**2 
                        grid.Hz[i, j] = 1 / (2 * pi * self.sigma**2) * (
                            numpy.exp(-r / (2 * self.sigma**2)) - 
                            numpy.exp(-self.rc**2 / (2 * self.sigma**2)))

        return grid
    
class testpulse_x_Exc(object):
    """instantaneous gaussian test pulse, travelling in x direction"""
    def __init__(self, x, H=1, T=1, sigma=1, rc=2):
        self.type = "testpulse"
        self.x = x
        self.H = H                          # positive=travelling to the right
        self.T = T                          # time to set it
        self.rc = rc                        # cutoff radius
        self.sigma = sigma
        
    def excite(self, grid, t):
        if t == 0:
            i0, j0 = grid.getIndex(self.x - self.rc, 0)
            i1, j1 = grid.getIndex(self.x + self.rc, 0)
            i0, i1 = 0, grid.Ny-1
            im, jm = grid.getIndex(self.x, 0)
            r0 = (jm - j0)**2               # squared "radius" of indicees
            
            for i in numpy.arange(i0, i1):
                for j in numpy.arange(j0, j1):
                    
                    if  (j - jm)**2 < r0:
                        r =  (grid.mesh_x[0, j] - self.x)**2 
                        grid.Hz[i, j] = grid.Hz[i, j] + 1 / (2 * pi * self.sigma**2) * (
                            numpy.exp(-r / (2 * self.sigma**2)) - 
                            numpy.exp(-self.rc**2 / (2 * self.sigma**2)))
                        
                        grid.Ey[i, j] = grid.Ey[i, j] -self.H / (2 * pi * self.sigma**2) * (
                            numpy.exp(-r / (2 * self.sigma**2)) - 
                            numpy.exp(-self.rc**2 / (2 * self.sigma**2)))

        return grid
    
    
class PW_pulse_x_TFSF_Exc(object):
    """gaussian test pulse, travelling in x direction, uses the total field/scattered field approach"""
    def __init__(self, A, t0, x0, dt, w0):
        self.type = "PW_pulse_x_TFSF"
        self.A = A
        self.t0 = t0
        self.x0 = x0
        self.dt = dt
        self.w0 = w0
        
    def excite(self, grid, t):
        i_list = grid.ij_SFTF[0,:]
        j_list = grid.ij_SFTF[1,:]
        
        for i,j in zip(i_list,j_list):
            # distance from start point
            d = (grid.mesh_x[0, j] - self.x0) 
            
            # amplitude 
            #A = self.A * numpy.exp(-0.5 * (t - d / grid.c0 - self.t0)**2 / self.dt**2 )
            dA = self.A * (t - d / grid.c0 - self.t0) / self.dt**2 * numpy.exp(-0.5 * (t - d / grid.c0 - self.t0)**2 / self.dt**2 )
            
            # add the amplitude to the fields
            grid.Hz[i, j] = grid.Hz[i, j] + dA
            grid.Ey[i, j] = grid.Ey[i, j] + dA

        return grid
        
  
class testpulse_y_Exc(object):
    """instantaneous gaussian test pulse, travelling in y direction"""
    def __init__(self, y, H=1, T=1, sigma=1, rc=2):
        self.type = "testpulse"
        self.y = y
        self.H = H                          # positive=travelling down
        self.T = T                          # time to set it
        self.rc = rc                        # cutoff radius
        self.sigma = sigma
        
    def excite(self, grid, t):
        if t == 0:
            i0, j0 = grid.getIndex(0, self.y + self.rc)
            i1, j1 = grid.getIndex(0, self.y - self.rc)
            j0, j1 = 0, grid.Nx-1
            im, jm = grid.getIndex(self.y, 0)
            r0 = (im - i0)**2               # squared "radius" of indicees

            for i in numpy.arange(i0, i1):
                for j in numpy.arange(j0, j1):
                    
                    if  (i - im)**2 < r0:
                        r =  (grid.mesh_y[i, 0] - self.y)**2 
                        grid.Hz[i, j] = 1 / (2 * pi * self.sigma**2) * (
                            numpy.exp(-r / (2 * self.sigma**2)) - 
                            numpy.exp(-self.rc**2 / (2 * self.sigma**2)))
                        grid.Ex[i, j] = -self.H / (2 * pi * self.sigma**2) * (
                            numpy.exp(-r / (2 * self.sigma**2)) - 
                            numpy.exp(-self.rc**2 / (2 * self.sigma**2)))

        return grid
    
