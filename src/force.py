"""
This file contains different forcing terms of 
2D Navier-Stokes equation.
"""

import numpy as np

class _Force_(object):

    def __init__(self, grid, u, w):
        
        self.grid = grid
        self.u = u
        self.w = w

        self._bc = {self.LEFT:{}, self.RIGHT:{}, self.UP:{}, self.DOWN:{}}
    
    def _cal_νΔu_(self, μ, ρ):
        """
        viscous force
        """

        δu = np.zeros( self.u.shape, np.float )
        δw = np.zeros( self.w.shape, np.float )

        μ_avg = 0.25 * ( μ[1:,1:] + μ[1:,:-1] + μ[:-1,1:] + μ[:-1,:-1] )

        δ = self.grid.δ

        # viscous terms in u-equation: (1/ρ)∂/∂x(2μ∂u/∂x) + (1/ρ)∂/∂y(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂x(2μ∂u/∂x):
        #  
        δu[1:-1,1:-1] += ( μ[2:-1,1:-1] * (self.u[2:,1:-1] -self. u[1:-1,1:-1] ) - \
                    μ[1:-2,1:-1] * ( self.u[1:-1,1:-1] - self.u[:-2,1:-1]) ) / \
                    ( 0.5 * (δ[0]**2) * ρ[1:-1,1:-1] )

        # ∂/∂y(μ(∂u/∂z+∂w/∂x)):
        δu[1:-1,1:-1] += ( μ_avg[:,1:] * ( (self.u[:,2:] - self.u[:,1:-1]) / (δ[1]**2) + \
                    (self.w[1:,1:] - self.w[:-1,1:]) / (δ[0] * δ[1]) ) - \
                    μ_avg[:,:-1] * ( (self.u[:,1:-1] - self.u[:,:-2]) / (δ[1]**2) + \
                    (self.w[1:,:-1] - self.w[:-1,:-1]) / (δ[0]*δ[1]) ) ) / \
                    ρ[:,1:-1]

        # at the boundaries: (dirichlet condition )
        δu[0,1:-1] -= ( μ[1,1:-1] * (self.w[1,1:] - self.w[1,:-1]) - \
                       μ[0,1:-1] * (self.w[0,1:] - self.w[0,:-1]) ) / \
                      (0.5 * δ[0] * δ[1] * ρ[0,1:-1])
        δu[-1,1:-1] -= ( μ[-1,1:-1] * (self.w[-1,1:] - self.w[-1,:-1]) - \
                        μ[-2,1:-1] * (self.w[-2,1:] - self.w[-2,:-1]) ) / \
                       (0.5 * δ[0] * δ[1] * ρ[-1,1:-1])

        # // done: TODO need to evaulate the viscous terms it at the boundaries.
        
        # viscous terms in w-equation: (1/ρ)∂/∂y(2μ∂w/∂z) + (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂y(2μ∂w/∂z):
        #
        δw[1:-1,1:-1] += (μ[1:-1,2:-1] * (self.w[1:-1,2:] - self.w[1:-1,1:-1]) - \
                        μ[1:-1,1:-2] * (self.w[1:-1,1:-1] - self.w[1:-1,:-2])) / \
                        (0.5 * (δ[1]**2) * ρ[1:-1,1:-1])

        # (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        δw[1:-1,1:-1] += (μ_avg[1:,:] * ( (self.w[2:,:] - self.w[1:-1,:])/(δ[0]**2) + \
                      (self.u[1:,1:] - self.u[1:,:-1]) / (δ[0]*δ[1]) ) - \
                      μ_avg[:-1,:] * ( (self.w[1:-1,:] - self.w[:-2,:]) / (δ[0]**2) + \
                      (self.u[:-1,1:] - self.u[:-1,:-1])/(δ[0]*δ[1]) ) ) / \
                      ρ[1:-1,:]

        # // done: TODO need to evaulate the viscous terms it at the boundaries.
        # at the boundaries:
        δw[1:-1,0] -= ( μ[1:-1,1] * (self.u[1:,1] - self.u[:-1,1]) - \
                       μ[1:-1,0] * (self.u[1:,0] - self.u[:-1,0]) ) / \
                      (0.5 * δ[0] * δ[1] * ρ[1:-1,0])
        δw[1:-1,-1] -= ( μ[1:-1,-1] * (self.u[1:,-1] - self.u[:-1,-1]) - \
                        μ[1:-1,-2] * (self.u[1:,-2] - self.u[:-1,-2]) ) / \
                       (0.5 * δ[0] * δ[1] * ρ[1:-1,-1])

        return δu, δw

    
    def _cal_ΔP_( self, ρ, β=1.0 ):
        """
        calculate the x and y-pressure gradients.
        """
        δu = np.zeros( self.u.shape, np.float )
        δw = np.zeros( self.w.shape, np.float )

        δ = self.grid.δ
        # ∂/∂x(p):
        δu[:,1:-1] -= β/δ[0] * ( self.p[1:,1:-1] - self.p[:-1,1:-1]  ) / ρ[:,1:-1]

        # ∂/∂z(p):
        δw[1:-1,:] -= β/δ[1] * ( self.p[1:-1,1:] - self.p[1:-1,:-1] ) / ρ[1:-1,:]

        return δu, δw


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

