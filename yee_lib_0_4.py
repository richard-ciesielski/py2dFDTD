"""Main library for 2D FDTD with YEE-Algorithmus: leapfrog and staggered grid
    - in vacuum
    - materials modeled via the Lorentz model (one pole):
        eps(w) = 1 + wp  / (wj^2 + i * w * gamma - w^2)
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
    def __init__(self, Lx, Ly, Nx, Ny, Eps0, Mu0):
        """sets up a grid with homogeneous material constants
            parameters:
                Lx,Ly - grid dimensions
                Nx,Ny - number of knots"""
        # vacuum parameters:
        self.Eps0 = Eps0
        self.Mu0 = Mu0
        
        # grid:
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
        
        # determine vacuum speed of light and time step:
        self.c0 = 1. / numpy.sqrt(Mu0 * Eps0)
        self.dt = self.h * .7 / self.c0
        
        # electric field, polarization current: integer time n
        #   x-component: integer positions (i,j)
        self.Ex = numpy.zeros((Ny, Nx)) 
        self.Jx = numpy.zeros((Ny, Nx)) 
        #   y-component: half integer positions (i+.5,j+.5)
        self.Ey = numpy.zeros((Ny-1, Nx-1)) 
        self.Jy = numpy.zeros((Ny-1, Nx-1)) 
        
        # electric field, polarization current: integer time n-1
        #   x-component: integer positions (i,j)
        self.oEx = numpy.zeros((Ny, Nx))  # o = old
        self.oJx = numpy.zeros((Ny, Nx)) 
        #   y-component: half integer positions (i+.5,j+.5)
        self.oEy = numpy.zeros((Ny-1, Nx-1)) 
        self.oJy = numpy.zeros((Ny-1, Nx-1)) 
        
        # magnetic field: half integer time n + .5
        #   z-component: integer x, half integer y: (i+.5,j)
        self.Hz = numpy.zeros((Ny-1, Nx)) 
        
        # distance function of the PML (here empty):
        self.d_PML_x = numpy.zeros((Ny, Nx))
        self.d_PML_y = numpy.zeros((Ny, Nx))
        self.d_PML_z = numpy.zeros((Ny, Nx))  
        
        # material constants, set to vacuum:
        # Lorentz-model (one pole)
        self.wj = numpy.zeros((Ny, Nx))             # freq. of pole p j
        self.wp = numpy.zeros((Ny, Nx))             # plasma frequency
        self.gamma = numpy.zeros((Ny, Nx))          # damping of the oscillator
        
        self.sigma = numpy.zeros((Ny, Nx))          # conductivity
    
    def __blurry2D(self, b, n=1):
        """blurries a matrix a little bit in order to get rid of unphysical 
            steps"""
        a = numpy.zeros(numpy.shape(b) + numpy.array([2,2]))
        a[1:-1,1:-1] = b
        
        for i in range(n):
            a[0,1:-1] = b[0,:]
            a[-1,1:-1] = b[-1,:]
            a[1:-1,0] = b[:,0]
            a[1:-1,-1] = b[:,-1]
            a[0,0] = b[0,0]
            a[-1,-1] = b[-1,-1]
            a[0,-1] = b[0,-1]
            a[-1,0] = b[-1,0]
            #a[1:-1,1:-1] = 1/9. * (a[1:-1,1:-1] + a[:-2,1:-1] + a[2:,1:-1] +
            #    a[1:-1,:-2] + a[1:-1,2:] + a[:-2,:-2] + a[2:,:-2] + 
            #    a[:-2,2:] + a[2:,2:])
            a[1:-1,1:-1] = 1/5. * (a[1:-1,1:-1] + a[:-2,1:-1] + a[2:,1:-1] +
                a[1:-1,:-2] + a[1:-1,2:])
            
        return a[1:-1, 1:-1]
        
    def updateConstants(self):
        """updates some numeric constants for the calculation of the ADE"""
        # blurry the material constants:
        self.sigma = self.__blurry2D(self.sigma, 0)
        self.gamma = self.__blurry2D(self.gamma, 0)
        self.wj = self.__blurry2D(self.wj, 0)
        self.wp = self.__blurry2D(self.wp, 0)
        
        dt = self.dt
        
        self.alpha = (2 - self.wj**2 * dt**2) / (1 + self.gamma * dt / 2)
        self.xsi = (self.gamma * dt - 2) / (self.gamma * dt + 2)
        self.delta = self.Eps0 * 2 * self.wp * dt**2 / \
                        (2 + self.gamma * dt)
        x = (2 * self.Eps0 + .5 * self.delta + self.sigma * dt)
        self.A = .5 * self.delta / x
        self.B = (2 * self.Eps0 - self.sigma * dt) / x
        self.C = 2 * dt / x
        
        
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
    
def CalculationStep(grid0, grid1, t, excitations):
    """does the next step of the calculation
           - uses leapfrog and a staggered grid
        parameters:
            dt - time step
            t - actual time
        output:
            Ex1,Ey1 - E-field, time n + 1
            Hz1 - H-field, time n + 1.5
    """
    # delete old field values:
    grid1.Ex = grid1.Ex * 0
    grid1.Ey = grid1.Ey * 0
    grid1.Jx = grid1.Jx * 0
    grid1.Jy = grid1.Jy * 0
    grid1.Hz = grid1.Hz * 0
    
    def rhsE():
        """right hand side of the first part: E-field (Ampere's law)
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
        
        # iterate Ampere's law with polarization current ADE:
        grid1.Ex = grid0.A * grid0.oEx + grid0.B * grid0.Ex + grid0.C * \
            (rotH_x - .5 * (1 + grid0.alpha) * grid0.Jx + grid0.xsi * grid0.oJx)
        grid1.Ey = grid0.A[:-1, :-1] * grid0.oEy + grid0.B[:-1, :-1] * grid0.Ey\
            + grid0.C[:-1, :-1] * (rotH_y - .5 * (1 + grid0.alpha[:-1, :-1]) *
                                    grid0.Jy + grid0.xsi[:-1, :-1] * grid0.oJy)
    
        # PMLs:
        grid1.Ex = grid1.Ex - grid0.d_PML_x * grid1.Ex * grid0.dt
        grid1.Ey = grid1.Ey - grid0.d_PML_y[:-1, :-1] * grid1.Ey * grid0.dt
        
        # PEC as BC for the borders (E-Field perpendicular to the surface): 
        grid1.Ey[:, -2:] = 0 # right
        grid1.Ey[:, :1] = 0 # left
        grid1.Ex[:1, :] = 0 # top
        grid1.Ex[-2:, :] = 0 # bottom
    
    def rhsJ():
        """right hand side of the second part: polarization current J (ADE)
           -> J is calculated at times N*dt on the integer grid"""
        # J:
        grid1.Jx = grid0.alpha * grid0.Jx + grid0.xsi * grid0.oJx + \
                    grid0.delta * (grid1.Ex - grid0.oEx) / (2 * grid0.dt)
        grid1.Jy = grid0.alpha[:-1, :-1] * grid0.Jy + \
                grid0.xsi[:-1, :-1] * grid0.oJy + \
                grid0.delta[:-1, :-1] * (grid1.Ey - grid0.oEy) / (2 * grid0.dt)
        
    def rhsH():
        """Right hand side of the third part: H-field (Faraday's law)
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
        grid1.Hz = grid0.Hz - grid0.dt / grid0.Mu0 * rotE_z
        
        # PMLs:
        grid1.Hz = grid1.Hz - grid0.d_PML_z[:-1, :] * grid1.Hz * grid0.dt
        
    
    
    # given: E0, H0
    # wanted: E1, H1
    # the calculation:
    rhsE()                   # given: E0, J0, H0 
    rhsJ()                   # given: E1, J0
    rhsH()                   # given: E1, H0
    
    # excitations:
    for obj in excitations:
        grid1 = obj.excite(grid1, t)
    
       
    return grid0.Ex, grid0.Ey, grid0.Jx, grid0.Jy, \
            grid1.Ex, grid1.Ey, grid1.Jx, grid1.Jy, grid1.Hz

