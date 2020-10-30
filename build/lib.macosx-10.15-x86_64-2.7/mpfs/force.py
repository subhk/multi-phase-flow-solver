"""
This file contains different forcing terms of 
2D Navier-Stokes equation.
"""

import numpy as np

class Force(object):

    LEFT    = 'left'
    RIGHT   = 'right'
    UP      = 'up'
    DOWN    = 'down'

    def __init__(self, grid, u, w, p):
        
        self.grid = grid
        self.u = u
        self.w = w
        self.p = p

        self._bc = {self.LEFT:{}, self.RIGHT:{}, self.UP:{}, self.DOWN:{}}
    
#    def _cal_νΔu_(self, μ, ρ):
    def _cal_vis_force_(self, mu, rho):    
        """
        viscous force
        """

        du = np.zeros( self.u.shape, np.float )
        dw = np.zeros( self.w.shape, np.float )

        mu_avg = 0.25 * ( mu[1:,1:] + mu[1:,:-1] + mu[:-1,1:] + mu[:-1,:-1] )

        d = self.grid.d

        rho_u = 0.5 * (rho[1:,:] + rho[:-1,:])
        rho_w = 0.5 * (rho[:,1:] + rho[:,:-1])

        # viscous terms in u-equation: (1/ρ)∂/∂x(2μ∂u/∂x) + (1/ρ)∂/∂y(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂x(2μ∂u/∂x):
        #  
        du[1:-1,1:-1] += ( mu[2:-1,1:-1] * (self.u[2:,1:-1] -self. u[1:-1,1:-1] ) - \
                    mu[1:-2,1:-1] * ( self.u[1:-1,1:-1] - self.u[:-2,1:-1]) ) / \
                    ( 0.5 * d[0]**2 * rho_u[1:-1,1:-1] )

        # ∂/∂y(μ(∂u/∂z+∂w/∂x)):
        du[:,1:-1] += ( mu_avg[:,1:] * ( (self.u[:,2:] - self.u[:,1:-1]) / (d[1]**2) + \
                    (self.w[1:,1:] - self.w[:-1,1:]) / (d[0] * d[1]) ) - \
                    mu_avg[:,:-1] * ( (self.u[:,1:-1] - self.u[:,:-2]) / (d[1]**2) + \
                    (self.w[1:,:-1] - self.w[:-1,:-1]) / (d[0]*d[1]) ) ) / \
                    rho_u[:,1:-1]

        # at the boundaries: (dirichlet condition )
        du[0,1:-1] -= ( mu[1,1:-1] * (self.w[1,1:] - self.w[1,:-1]) - \
                       mu[0,1:-1] * (self.w[0,1:] - self.w[0,:-1]) ) / \
                      ( 0.5 * d[0] * d[1] * rho_u[0,1:-1] )
        du[-1,1:-1] -= ( mu[-1,1:-1] * (self.w[-1,1:] - self.w[-1,:-1]) - \
                        mu[-2,1:-1] * (self.w[-2,1:] - self.w[-2,:-1]) ) / \
                       ( 0.5 * d[0] * d[1] * rho_u[-1,1:-1] ) 

        # // TODO need to evaulate the viscous terms it at the boundaries. (done)
        
        # viscous terms in w-equation: (1/ρ)∂/∂y(2μ∂w/∂z) + (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂y(2μ∂w/∂z):
        #
        dw[1:-1,1:-1] += ( mu[1:-1,2:-1] * (self.w[1:-1,2:] - self.w[1:-1,1:-1]) - \
                        mu[1:-1,1:-2] * (self.w[1:-1,1:-1] - self.w[1:-1,:-2]) ) / \
                        ( 0.5 * d[1]**2 * rho_w[1:-1,1:-1] )

        # (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        dw[1:-1,:] += ( mu_avg[1:,:] * ( (self.w[2:,:] - self.w[1:-1,:])/(d[0]**2) + \
                      (self.u[1:,1:] - self.u[1:,:-1]) / (d[0]*d[1]) ) - \
                      mu_avg[:-1,:] * ( (self.w[1:-1,:] - self.w[:-2,:]) / (d[0]**2) + \
                      (self.u[:-1,1:] - self.u[:-1,:-1])/(d[0]*d[1]) ) ) / \
                      rho_w[1:-1,:]

        # // TODO need to evaulate the viscous terms it at the boundaries. (done)
        # at the boundaries:
        dw[1:-1,0] -= ( mu[1:-1,1] * (self.u[1:,1] - self.u[:-1,1]) - \
                       mu[1:-1,0] * (self.u[1:,0] - self.u[:-1,0]) ) / \
                      ( 0.5 * d[0] * d[1] * rho_w[1:-1,0] )
                      
        dw[1:-1,-1] -= ( mu[1:-1,-1] * (self.u[1:,-1] - self.u[:-1,-1]) - \
                        mu[1:-1,-2] * (self.u[1:,-2] - self.u[:-1,-2]) ) / \
                       ( 0.5 * d[0] * d[1] * rho_w[1:-1,-1] )

        return du, dw

    
#    def _cal_ΔP_( self, ρ, β=1.0 ):
    def _cal_gradP_( self, rho, beta=1.0 ):    
        """
        calculate the x and y-pressure gradients.
        """
        du = np.zeros( self.u.shape, np.float )
        dw = np.zeros( self.w.shape, np.float )

        d = self.grid.d
        # -∂/∂x(p):
        du[:,1:-1] -= beta/d[0] * ( self.p[1:,1:-1] - self.p[:-1,1:-1] ) / ( 0.5 * (rho[1:,1:-1] + rho[:-1,1:-1]) )

        # -∂/∂z(p):
        dw[1:-1,:] -= beta/d[1] * ( self.p[1:-1,1:] - self.p[1:-1,:-1] ) / ( 0.5 * (rho[1:-1,1:] + rho[1:-1,:-1]) )

        return du, dw


    # def _cal_σ_force( self, ρ, dt ):
    #     """
    #     calculate force due to the surface tension, σ
    #     """
    #     δu = np.zeros( self.u.shape, np.float )
    #     δw = np.zeros( self.w.shape, np.float )

    #     if self._multi_phase and self.σ_coeff != 0.:
    #         ℙx, ℙz = self._cal_σ_()

    #         ℙx /= ρ[1:-1,1:-1] 
    #         ℙz /= ρ[1:-1,1:-1]
         
    #         δu[1:-1,1:-1] += ℙx
    #         δw[1:-1,1:-1] += ℙz

    #     return δu * dt, δw * dt
    

    # def _cal_σ_( self, dt ):
        
    #     χ = self.χ
    #     mask = self.mask
    #     δ = self._grid.δ
    #     u, w = self.u, self.w

