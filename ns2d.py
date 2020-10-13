"""
This file contains the main program file
for NS2D solver
"""
from functools import partial
import numpy as np

from domain import Domain


class NS2Dsolver(object):

    LEFT    = 'left'
    RIGHT   = 'right'
    UP      = 'up'
    DOWN    = 'down'

    MASS_SCALE = 'scale'
    MASS_ADD = 'add'
    MASS_IGNORE = 'ignore'

    CURV_CSF = 'csf'
    CURV_DAC = 'dac'
    CURV_MDAC = 'mdac'

    def __init__(self, grid, **args):

        # Initialise data structure:
        self.grid = grid
        self.time = 0.
        self.p = np.zeros( np.array(grid.shape)+1, dtype=np.float64  )
        self.u = np.zeros( (self.p.shape[0]-1, np.p.shape[1]), dtype=np.float64 )
        self.w = np.zeros( (self.p.shape[0], np.p.shape[1]-1), dtype=np.float64 )
        self.mask = np.ones(self.p.shape, int)
        self.χ = np.zeros(self.p.shape, dype=np.float64 )  # vof value
        self.tracer = np.zeros(self.p.shape, dtype=np.float64 )
        self._bc = {self.LEFT:{}, self.RIGHT:{}, self.UP:{}, self.DOWN:{}}
        self._bc_finalised = False
        self.iter = 0
        self._sim_time = []

        #default paramters:
        default_params = {'ρ₁': 1000, 'ρ₂': 10, 
                        'μ₁': 1.0, 'μ₂': 1.0,
                        'σ': 0.0, 'gra': [0.0, 0.0],
                        'multi_phase': False, 'use_passive_tracer': False,
                        'curve_method': self.CURV_MDAC,
                        'mass_conservation': self.MASS_ADD, 
                        'property_smoothing': False    }

        for key in default_params:
            self.__dict__[key] = default_params[key]

        
        # optional parameters

        if 'μ' in args: args['μ₁'] = args['μ']
        if 'ν' in args: args['ν₁'] = args['ν']
        if 'ρ' in args: args['ρ₁'] = args['ρ']

        if 'ρ₁' in args: self.ρ₁ = args['ρ_liquid']
        if 'ρ_gas' in args: self.ρ_gas = args['ρ_gas']

        # calculating dynamic viscosity
        if 'ν_liquid' in args:
            args['μ_liquid'] = args['ν_liquid'] * self.ρ_liquid
        
        if 'ν_gas' in args:
            args['ν_gas'] = args['ν_gas'] * self.ρ_gas
        
        # store all the parameters
        for key in default_params:
            if key in args:
                self.__dict__[key] = args[key]




    
    def _set_ic_(self, **args):
        """
        set initial condition for the simuation.
        the arguments should be in terms of 'u', 'w', 'p', or 'χ',
        it can be constant or function of (x, z)
        """

        for _ic_ in args:

            if not _ic_ in ['u', 'w', 'p', 'χ']:
                continue  # should not show any error in any case
                
            fun_ = args[_ic_]

            # if _ic_ = 'u', then _set_ic_ =self.u, likewise.
            # gets updated every time calling, cool!
            _set_ic_ = self.__dict__[_ic_]

            #
            if callable(fun_):
                # Calculate node offsets.
                # True is converted to integer one, False to integer zero.
                xOffset = 0.5 * (_ic_ in ('p', 'w', 'χ'))
                yOffset = 0.5 * (_ic_ in ('p', 'u', 'χ'))

                # iterate over all nodes.
                for i in range(_set_ic_.shape[0]):
                    for j in range(_set_ic_.shape[1]):
                        _node_ = self.grid[i - xOffset, j - yOffset]
                        _set_ic_[i,j] = fun_( _node_[0], _node_[1] )
            else:
                _set_ic_[:,:] = fun_ # 'fun_' is a constant.



    def _remove_singularity(self):
        """
        if Neumann bcs apply all along the walls, then the 
        Poisson equation becomes signular. This function fixes
        it to have an unique sol.
        """

        remove_singularity(self.mask)
        



    def _cal_RHS_poisson_eq_(self, dt):
        """
        calculate RHS of the pressure poisson equation 
        ∇⋅(1/ρ ∇Ξ) = -∇.u* /Δt
        """
        u, w, p, δ, mask = self.u, self.w, self.grid.δ, self.mask

        net_outflow = (u[-1,1:-1]-u[0,1:-1]).sum() * δ[1] + \
            (w[1:-1,-1]-w[1:-1,0]).sum() * δ[0]

        outflow_len =  0.
        outflow = 0.

        dirichlet_used = ('p' in self.bc['self.UP']) or \
            ('p' in self._bc['self.DOWN']) or \
            ('p' in self._bc['self.LEFT']) or \
            ('p' in self._bc['self.RIGHT'])

        if 'dwdn' in self._bc[self.UP]:
            outflow_len += mask[1:-1,-2].sum() * δ[0] 
            outflow += w[1:-1,-1].sum() * δ[0]

        if 'dwdn' in self._bc[self.DOWN]:
            outflow_len += mask[1:-1,1].sum() * δ[0]
            outflow -= w[1:-1,0].sum() * δ[0]

        if 'dudn' in self._bc[self.LEFT]:
            outflow_len += mask[1,1:-1].sum() * δ[1]
            outflow += u[0,1:-1].sum() * δ[1]

        if 'dudn' in self._bc[self.RIGHT]:
            outflow_len += mask[-2,1:-1].sum() * δ[1]
            outflow += u[-1,1:-1].sum() * δ[1]

        if not dirichlet_used and outflow_len > 0. and \
            self._mass_conservation in (self.MASS_ADD, self.MASS_SCALE):

            if outflow == 0. or self._mass_conservation == self.MASS_ADD:
                
                flow_correction = net_outflow/outflow_len

                if 'dwdn' in self._bc[self.UP]:     w[1:-1,-1] -= mask[1:-1,-2] * flow_correction
                if 'dwdn' in self._bc[self.DOWN]:   w[1:-1,0]  += mask[1:-1,1]  * flow_correction

                if 'dudn' in self._bc[self.LEFT]:   u[0,1:-1]  += mask[1,1:-1]  * flow_correction
                if 'dudn' in self._bc[self.RIGHT]:  u[-1,1:-1] -= mask[-2,1:-1] * flow_correction
            
            else:
                flow_correction = 1. - net_outflow / outflow
                if 'dwdn' in self._bc[self.UP]:     w[1:-1,-1] *= flow_correction
                if 'dwdn' in self._bc[self.DOWN]:   w[1:-1,0]  *= flow_correction

                if 'dudn' in self._bc[self.LEFT]:   u[0,1:-1]  *= flow_correction
                if 'dudn' in self._bc[self.RIGHT]:  u[-1,1:-1] *= flow_correction


        # calculate RHS of the PPE here:
        ℝ = np.zeros( p.shape, np.float )
        ℝ[1:-1,1:-1] = ( u[1:,1:-1] - u[:-1,1:-1] ) / δ[0] + \
            ( w[1:-1,1:] - w[1:-1,:-1] ) / δ[1]
        

        # for mass conservation
        if not dirichlet_used and (outflow_len == 0.0 or \
            not self._mass_conservation in (self.MASS_ADD, self.MASS_SCALE)):

            ℝ[1:-1,1:-1] -= ℝ[1:-1,1:-1].sum() / ℝ[1:-1,1:-1].size

        return -1./dt * ℝ

    
    def _cal_nonlinear_terms_(self, dt, γ=0.0 ):
        """
        calculate the nonlinear terms for the u and w-equations.
        """
        δ = self._grid.δ
        u, w = self.u, self.w

        δu = np.zeros( self.u.shape, np.float )
        δw = np.zeros( self.w.shape, np.float )

        ###########
        # u-equation::nonlinear terms → ∂/∂x(u²) + ∂/∂z(uw)
        # ∂/∂x(u²):
        δu[1:-1,1:-1] -= 0.25 * ( (u[1:-1,1:-1]-u[2:,1:-1])**2 - \
                                (u[1:-1,1:-1]-u[:-2,1:-1])**2 ) / δ[0]
        
        # at the bounadry: ∂/∂x(u²) = 2u∂/∂x(u) = -2u∂/∂z(w)
        δu[0,1:-1]  += u[-1,1:-1] * ( (w[0,1:] + w[1,1:]) - \
                                (w[0,:-1] + w[1,:-1]) ) / δ[1]
        δu[-1,1:-1] += u[-1,1:-1] * ( (w[-2,1:] + w[-1,1:]) - \
                                     (w[-2,:-1] + w[-1,:-1]) ) / δ[1]

        # ∂/∂y(uv):
        δu[:,1:-1] -= 0.25 * ( (u[:,1:-1] + u[:,2:]) * (w[:-1,1:] + w[1:,1:]) - \
            (u[:,1:-1] + u[:,:-2]) * (w[:-1,:-1] + w[1:,:-1]) ) / δ[1]        

        ###########
        # w-equation::nonlinear terms → ∂/∂z(w²) + ∂/∂x(uw)
        # ∂/∂z(w²):
        δw[1:-1,1:-1] -= 0.25 * ( (w[1:-1,1:-1] + w[1:-1,2:])**2 - \
                                 (w[1:-1,1:-1] + w[1:-1,:-2])**2 ) / δ[1]
        
        # at the bounadry: ∂/∂z(w²) = 2w∂/∂z(w) = -2w∂/∂x(u)
        δw[1:-1,0]  += w[1:-1,0] * ( (u[1:,0] + u[1:,1]) - \
                                   (u[:-1,0] + u[:-1,1]) ) / δ[0]
        δw[1:-1,-1] += w[1:-1,-1] * ( (u[1:,-2] + u[1:,-1]) - \
                                     (u[:-1,-2] + u[:-1,-1]) ) / δ[0]

        # ∂/∂x(uw):
        δw[1:-1,:] -= 0.25 * ( (w[1:-1,:] + w[2:,:]) * (u[1:,:-1] + u[1:,1:]) - \
            (w[1:-1,:] + w[:-2,:]) * (u[:-1,:-1] + u[:-1,1:]) ) / δ[0] 


        # // TODO: need to implement the upwind scheme for stabilisation.
        

        return δu * dt, δw * dt
    

    def _cal_ΔP_( self, ρ, β=1.0, dt ):
        """
        calculate the x and y-pressure gradients.
        """
        δu = np.zeros( self.u.shape, np.float )
        δw = np.zeros( self.v.shape, np.float )

        p = self.p
        δ = self._grid.δ
        # ∂/∂x(p):
        δu[:,1:-1] -= β/δ[0] * ( p[1:,1:-1] - p[:-1,1:-1]  ) / ρ[:,1:-1]

        # ∂/∂z(p):
        δw[1:-1,:] -= β/δ[1] * ( p[1:-1,1:] - p[1:-1,:-1] ) / ρ[1:-1,:]

        return δu * dt, δw * dt

    
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





    #
    # now, simuation starts, finally!
    #
    def simuation(self, dt, β=1., γ=0.):

        δ, mask, χ = self.grid.δ, self.mask, self.χ
        u, w, p = self.u, self.w, self.p

        
        # Impose boundary conditions.
        self._update_vel_bc_()
        self._update_pressure_bc_()


        # if passive tracer used:
        if self.use_passive_tracer:
            # add the tracer advection code.
        
        # if multi-phase used:
        if self.multi_phase:
            # add multiphase code here

        
        

        



