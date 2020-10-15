"""
This file contains the main program file
for NS2D solver
"""
from functools import partial
import numpy as np

from domain import Domain
from bc import _bc_ns2d_
from force import _Force_


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
        #self._bcs_ = _bcs_
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

        if 'ρ₁' in args: self.ρ1 = args['ρ₁']
        if 'ρ₂' in args: self.ρ2 = args['ρ₂']

        # calculating dynamic viscosity
        if 'ν₁' in args:
            args['μ₁'] = args['ν₁'] * self.ρ1
        
        if 'ν₂' in args:
            args['μ₂'] = args['ν₂'] * self.ρ2
        
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
                xOffset = 0.5 * ( _ic_ in ('p', 'w', 'χ') )
                yOffset = 0.5 * ( _ic_ in ('p', 'u', 'χ') )

                # iterate over all nodes.
                for i in range(_set_ic_.shape[0]):
                    for j in range(_set_ic_.shape[1]):
                        _node_ = self.grid[i - xOffset, j - yOffset]
                        _set_ic_[i,j] = fun_( _node_[0], _node_[1] )
            else:
                _set_ic_[:,:] = fun_ # 'fun_' is a constant.



    # def _remove_singularity(self):
    #     """
    #     if Neumann bcs apply all along the walls, then the 
    #     Poisson equation becomes signular. This function fixes
    #     it to have an unique sol.
    #     """

    #     remove_singularity(self.mask)
        

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

    
    def _cal_nonlinear_terms_(self, γ=0.0 ):
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
        

        return δu, δw 


    def _cfl_dt_(self, safety=0.8):

        """
        Calculate a time stepping value that should give
        a stable simulation. It will get called every iteration.
        PS: may call it once 10 or some time-instant, could result
        in more-effective: // TODO requires testing
        """

        δ = self.grid.δ
        u = abs(self.u) + 1.e-7 * (self.u == 0.0) # avoid division by zero
        w = abs(self.v) + 1.e-7 * (self.w == 0.0)

        # may be multi-phase requires different criretia
        # // TODO: need to dig some literatures.

        ν = self.μ / self.ρ

        dt1 = min( δ[0] / u.max(), δ[1] / w.max() )
        dt2 = 0.5 / (ν * (delta**-2).sum())
        dt3 = 2.0 * nu / max(u.max(), v.max())**2

        return min(dt1, dt2, dt3) * safety_factor


    #
    # now, simuation starts, finally!
    #
    def simuation(self, dt, β=1., γ=0.):
        """
        Simulation start
        Args:
            dt ([float]): [time styep value of the simulation.]

            β:  factor in the projection method
                See "H. P. Langtangen, K.-A. Mardal and R. Winther:
                Numerical Methods for Incompressible Viscous Flow"
            
            γ:  Upwind differencing factor
                See "Numerical Simulation in Fluid Dynamics: A Practical 
                Introduction" (1997) by Griebel, Dornsheifer and Neunhoeffer.
        """

        δ, mask, χ = self.grid.δ, self.mask, self.χ
        u, w, p = self.u, self.w, self.p

        # imposed boundary conditions:
        _bc2d_ = _bc_ns2d_(u, w, p)
        _bc2d_._update_vel_bc_()
        _bc2d_._update_pressure_bc_()

        # if passive tracer used:
        if self.use_passive_tracer:
            # add the tracer advection code.
            print( 'no passive tracer' )
        
        # if multi-phase used:
        if self.multi_phase:
            # add multiphase code here
            print( 'no multi-phase simulation' )

        # let's keep it for multi-phase case:
        # for χ=1, it would be for one-phase.
        ρ = self.ρ1 * (1. - χ) + self.ρ2 * χ
        μ = self.μ1 * (1. - χ) + self.μ2 * χ


        # RHS for the intermediate velocity equations.
        δu0, δw0  = self._cal_nonlinear_terms_(γ)
        
        # pressure gradient
        fc = _Force_(u, w, p)
        δu1, δw1 = fc._cal_ΔP_(ρ, β)
        # viscous force
        δu2, δw2 = fc._cal_νΔu_(μ, ρ)

        du = dt*( δu0 + δu1 + δu2 )
        dw = dt*( δw0 + δw1 + δw2 )

        # and ofcourse, gravity!
        du += dt*self.gra[0]
        dw += dt*self.gra[1]

        # update time
        self.time += dt
        # update iteration
        self.iter += 1

        # imposed bcs to intermediate velocity.
        _bc2d_._update_intermediate_vel_bc_() 

        # calculate RHS of PPE
        ppe = self._cal_RHS_poisson_eq_()

        # Calculate boundary conditions for 'Ξ' and store the values in
        # the ghost cells.   
        # 
        Ξ = np.zeros(p.shape, float)
        self._update_phi_bc(Ξ, β)  

        # Calculate pressure correction. (phi = p^{n+1} - β * p^(n))
        # 'mask' defines where to use Dirichlet and Neumann boundary conditions.
        p *= β               
        p += poisson(Ξ, ppe, δ, mask, ρ, self._poisson_data, 3) # LU 

        # Correct velocity on boundary for Dirichlet pressure boundary condition
        # do not worry it's taken care by mask function
        u[0,1:-1] -= dt * (Ξ[1,1:-1] - Ξ[0,1:-1]) / \
            (δ[0] * ρ[0,1:-1]) * (mask[1,1:-1] & 1)

        u[-1,1:-1] -= dt * (Ξ[-1,1:-1] - Ξ[-2,1:-1]) / \
            (δ[0] * ρ[-1,1:-1]) * (mask[-2,1:-1] & 1)

        w[1:-1,0] -= dt * (Ξ[1:-1,1] - Ξ[1:-1,0]) / \
            (δ[1] * ρ[1:-1,0]) * (mask[1:-1,1] & 1)
        w[1:-1,-1] -= dt * (Ξ[1:-1,-1] - Ξ[1:-1,-2]) / \
            (δ[1] * ρ[1:-1,-1]) * (mask[1:-1,-2] & 1)

        













        



