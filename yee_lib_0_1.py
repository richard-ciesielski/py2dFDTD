"""Main library for 2D FDTD with YEE-Algorithmus: leapfrog and staggered grid
    according to Taflove's book
    - in vacuum
    - BC: PEC and PML
    - TE-mode
.....................................................................
Institut fuer Angewandte Photophysik, Technische Universitaet Dresden
Institute home page: http://www.iapp.de
Erstellt im Rahmen einer Belegarbeit 2009/10 von Richard Ciesielski
email: Richard.Ciesielski@gmail.com
....................................................................."""
from __future__ import division     # 1/2 = .5
import numpy
import pylab

from numpy import pi

class staggered_grid_2D(object): 
    def __init__(self, Lx, Ly, Nx, Ny, eps=1, mu=1):
        """sets up a grid with homogeneous material constants
            parameters:
                Lx,Ly - grid dimensions
                Nx,Ny - number of knots"""
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        # integer gridpoints:
        x, h1 = numpy.linspace(0, Lx, Nx, retstep=True)
        self.mesh_x = x * numpy.ones((Ny, 1))
        y, h2 = numpy.linspace(Ly, 0, Ny, retstep=True)
        h2 = -h2 # see grid
        y = numpy.transpose(y)
        self.mesh_y = numpy.transpose([y]) * numpy.ones((1, Nx))
        if h1 != h2:
            """Error message!"""
            print "Unfortunate choice of the simulation box. Chose it in a way\
            that Length_x / dx is integer (same for y)."
            print "Lx :", Lx, "Ly :", Ly, "Nx :",Nx, "Ny :", Ny, "dx :", h1, \
                "dy :", h2
            raise SystemExit()
            
        self.h = h1
        
        # electric field: integer times n
        #   x-component: integer positions (i,j)
        self.Ex = numpy.zeros((Ny, Nx)) 
        #   y-component: half integer positions (i+.5,j+.5)
        self.Ey = numpy.zeros((Ny-1, Nx-1)) 
        # magnetic field: half integer times n + .5
        #   z-component: integer x, half integer y: (i+.5,j)
        self.Hz = numpy.zeros((Ny-1, Nx)) 
        
        # distance function of the PML (here empty):
        self.d_PML_x = numpy.zeros((Ny, Nx))
        self.d_PML_y = numpy.zeros((Ny, Nx))
        self.d_PML_z = numpy.zeros((Ny, Nx))  
        
        # material constants:
        self.eps = numpy.ones((Ny, Nx)) * eps
        self.mu = numpy.ones((Ny, Nx)) * mu
        
        # determine vacuum speed of light and time step:
        self.c0 = 1. / numpy.sqrt(mu * eps)
        self.dt = self.h * .7 / self.c0
        
    def getIndex(self, x, y):
        """finds the indices of a given position on the grid"""
        i = numpy.int32((self.mesh_y[0, 0] - y) / self.h)
        j = numpy.int32((x - self.mesh_x[0, 0]) / self.h)
        return i, j
        

    def addPML(self, a, direction, ex=10, type="selective"):
        """adds a distance function for a PML layer"""
        id = int(a / self.h)
        if (direction == "left"): # vertical layer
            self.d_PML_y[:, :id] = self.d_PML_y[:, :id] + ((a -
                                        self.mesh_x[:, :id]) / a)**2. * ex 
            if type=="full":
                self.d_PML_z[:, :id] = self.d_PML_z[:, :id] + ((a -
                                        self.mesh_x[:, :id]) / a)**2. * ex 
                self.d_PML_x[:, :id] = self.d_PML_x[:, :id] + ((a -
                                        self.mesh_x[:, :id]) / a)**2. * ex

        if (direction == "right"):
            self.d_PML_y[:, -id:] = self.d_PML_y[:, -id:] + ((a -
                    self.mesh_x[0, -1] + self.mesh_x[:, -id:]) / a)**2. * ex
            if type=="full":
                self.d_PML_z[:, -id:] = self.d_PML_z[:, -id:] + ((a -
                    self.mesh_x[0, -1] + self.mesh_x[:, -id:]) / a)**2. * ex
                self.d_PML_x[:, -id:] = self.d_PML_x[:, -id:] + ((a -
                    self.mesh_x[0, -1] + self.mesh_x[:, -id:]) / a)**2. * ex
            
                    
        if (direction == "up"): # horizontal layer
            self.d_PML_x[:id, :] = self.d_PML_x[:id, :] + ((a - self.mesh_y[0, 0]
                                        + self.mesh_y[:id, :]) / a)**2. * ex
            if type=="full":
                self.d_PML_z[:id, :] = self.d_PML_z[:id, :] + ((a - self.mesh_y[0, 0]
                                        + self.mesh_y[:id, :]) / a)**2. * ex
                self.d_PML_y[:id, :] = self.d_PML_y[:id, :] + ((a - self.mesh_y[0, 0]
                                        + self.mesh_y[:id, :]) / a)**2. * ex
            
        if (direction == "down"):
            self.d_PML_x[-id:, :] = self.d_PML_x[-id:, :] + ((a - 
                                        self.mesh_y[-id:, :]) / a)**2. * ex
            if type=="full":
                self.d_PML_z[-id:, :] = self.d_PML_z[-id:, :] + ((a - 
                                        self.mesh_y[-id:, :]) / a)**2. * ex
                self.d_PML_y[-id:, :] = self.d_PML_y[-id:, :] + ((a - 
                                        self.mesh_y[-id:, :]) / a)**2. * ex
            
    
def CalculationStep(grid0, grid1, Eps0, Mu0, t, dt, excitations):
    """does the next step of the calculation
           - uses leapfrog and a staggered grid
        parameters:
            grid0 - contains all the information about grid and fields
            Eps0, Mu0 - dielectric constant and permittivity
            dt - time step
        output:
            Ex1,Ey1 - E-field, time n + 1
            Hz1 - H-field, time n + 1.5
    """
    # delete old field values:
    grid1.Ex = grid1.Ex * 0
    grid1.Ey = grid1.Ey * 0
    grid1.Hz = grid1.Hz * 0
    
    def rhsE():
        """right hand side of the first part: E-field
           -> E - field is calculated at times N*dt on the integer grid"""
        # calculate the components of rot H at x = i*dx, t = (N+1)*dt
        
        # without border, central differece, 2nd order:
        rotH_x = grid0.Hz[1:, :] - grid0.Hz[:-1, :]  # dHz/dy
        rotH_y = -(grid0.Hz[:, 1:] - grid0.Hz[:, :-1])  # -dHz/dx
        
        # approximate points at the border by their neighbours 
        # (only the necessary ones)
        rotH_x = numpy.concatenate(([rotH_x[0, :]], rotH_x, [rotH_x[-1, :]]), 0) 
        # divide by dy, thus: derivative complete
        rotH_x = rotH_x / grid0.h    
        rotH_y = rotH_y / grid0.h
        
        # iterate Ampere's law:
        grid1.Ex = grid0.Ex + dt / (Eps0 * grid0.eps) * rotH_x
        grid1.Ey = grid0.Ey + dt / (Eps0 * grid0.eps[:-1, :-1]) * rotH_y
        
        # PMLs:
        grid1.Ex = grid1.Ex - grid0.d_PML_x * grid1.Ex * grid0.dt
        grid1.Ey = grid1.Ey - grid0.d_PML_y[:-1, :-1] * grid1.Ey * grid0.dt
        
        # PEC as BC for the borders (E-Field perpendicular to the surface): 
        grid1.Ey[:, -2:] = 0 # right
        grid1.Ey[:, :1] = 0 # left
        grid1.Ex[:1, :] = 0 # top
        grid1.Ex[-2:, :] = 0 # bottom

    def rhsH():
        """Right hand side of the second part: H-field
           -> H - field is calculated at times (N+1.5)*dt on the half integer 
                    grid"""        
        #calculate components of rot E at points (i+.5)*dx and times (N+1.5)*dt:
        rotE_z1 = (grid1.Ey[:, 1:] - grid1.Ey[:, :-1]) # dEy/dx
        rotE_z2 =- (grid1.Ex[1:, :] - grid1.Ex[:-1, :]) # dEx/dy
        
        # approximate points at the border by their neighbours
        # (only the necessary ones)
        rotE_z1 = numpy.transpose(numpy.concatenate(([rotE_z1[:, 0]],
                                numpy.transpose(rotE_z1), [rotE_z1[:, -1]]))) 
        # divide by dx, thus: derivative complete
        rotE_z = (rotE_z1 + rotE_z2) / grid0.h 
        
        # iterate Faraday's law:
        grid1.Hz = grid0.Hz - dt / (Mu0 * grid0.mu[:-1, :]) * rotE_z
        
        # PMLs:
        grid1.Hz = grid1.Hz - grid0.d_PML_z[:-1, :] * grid1.Hz * grid0.dt

        
    # given: E0, H0
    # wanted: E1, H1
    # the calculation:
    rhsE()                   # given: E0, H0    
    rhsH()                   # given: E1, H0
    
    # excitations:
    for obj in excitations:
        grid1 = obj.excite(grid1, t)
       
    return grid1.Ex, grid1.Ey, grid1.Hz

