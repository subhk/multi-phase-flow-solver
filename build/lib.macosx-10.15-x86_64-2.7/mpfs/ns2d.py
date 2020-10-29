"""
This file contains the main program file
for NS2D solver
"""
from functools import partial
import numpy as np
from ..poisson import *

from .force import Force
import time

import logging
logger = logging.getLogger(__name__.split('.')[-1])

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

    def __init__(self, grid, bc2d, **kwargs):

        # Initialise data structure:
        self.grid = grid

        # boundary conditions
        self.bc2d = bc2d

        #self._bcs_ = _bcs_
        self.time = 0.

        # all the prognostic variables.
        self.p = np.zeros( np.array(grid.shape)+1, dtype=np.float64  )
        self.u = np.zeros( (self.p.shape[0]-1, np.p.shape[1]), dtype=np.float64 )
        self.w = np.zeros( (self.p.shape[0], np.p.shape[1]-1), dtype=np.float64 )

        self.mask = np.ones(self.p.shape, int)

        # disclaimer: not doing anything
        self.chi = np.zeros(self.p.shape, dype=np.float64 )  # vof value

        # disclaimer: not doing anything
        self.tracer = np.zeros(self.p.shape, dtype=np.float64 )

        self._bc = {self.LEFT:{}, self.RIGHT:{}, self.UP:{}, self.DOWN:{}}
        self._bc_finalised = False
        self.poisson_data = poisson_data()
        
        self.start_time = self.get_world_time()

        # Attributes:
        self.sim_time = self.initial_sim_time = 0.
        self.iteration = self.initial_iteration = 0

        # Default integration parameters:
        self.stop_sim_time = np.inf
        self.stop_wall_time = np.inf
        self.stop_iteration = np.inf

        #default paramters:
        default_params = {'rho1': 1000, 'rho2': 1000, 
                        'mu1': 1.0, 'mu2': 1.0,
                        'sigma': 0.0, 'gra': [0.0, 0.0],
                        'multi_phase': False, 'use_passive_tracer': False }

        for key in default_params:
            self.__dict__[key] = default_params[key]

        
        # optional parameters
        if 'mu'  in kwargs: kwargs['mu1']  = kwargs['mu']
        if 'nu'  in kwargs: kwargs['nu1']  = kwargs['nu']
        if 'rho' in kwargs: kwargs['rho1'] = kwargs['rho']

        # options for two-phase
        if 'rho1' in kwargs: self.rho1 = kwargs['rho1']
        if 'rho2' in kwargs: self.rho2 = kwargs['rho2']

        # calculating dynamic viscosity
        if 'mu1' in kwargs:
            kwargs['mu1'] = kwargs['nu1'] * self.rho1
        
        if 'mu2' in kwargs:
            kwargs['mu2'] = kwargs['nu2'] * self.rho2
        
        # store all the parameters
        for key in default_params:
            if key in kwargs:
                self.__dict__[key] = kwargs[key]

            
    @property
    def ok(self):
        """Check that current time and iteration pass stop conditions."""
        if self.sim_time >= self.stop_sim_time:
            logger.info('Simulation stop time reached.')
            return False
        elif (self.get_world_time() - self.start_time) >= self.stop_wall_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

    @property
    def sim_time(self):
        return self._sim_time.value

    def get_world_time(self):
        self._float_array[0] = time.time()
        return self._float_array[0]

    def _set_ic_(self, **kwargs):
        """
        set initial condition for the simuation.
        the arguments should be in terms of 'u', 'w', 'p', or 'χ',
        it can be constant or function of (x, z)
        """

        for _ic_ in kwargs:

            if not _ic_ in ['u', 'w', 'p', 'chi']:
                continue  # should not show any error in any case
                
            fun_ = kwargs[_ic_]

            # if _ic_ = 'u', then _set_ic_ =self.u, likewise.
            # gets updated every time calling, cool!
            _set_ic_ = self.__dict__[_ic_]

            #
            if callable(fun_):
                # Calculate node offsets.
                # True is converted to integer one, False to integer zero.
                xOffset = 0.5 * ( _ic_ in ('p', 'w', 'chi') )
                yOffset = 0.5 * ( _ic_ in ('p', 'u', 'chi') )

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
    #       have to add it!!!!
        

    def _cal_RHS_poisson_eq_(self, dt):
        """
        calculate RHS of the pressure poisson equation 
        ∇⋅(1/ρ ∇Ξ) = -∇.u* /Δt
        """
        u, w, p = self.u, self.w, self.w 
        d, mask = self.grid.d, self.mask

        net_outflow = (u[-1,1:-1]-u[0,1:-1]).sum() * d[1] + \
            (w[1:-1,-1]-w[1:-1,0]).sum() * d[0]

        outflow_len =  0.
        outflow = 0.

        dirichlet_used = ('p' in self.bc['self.UP']) or \
            ('p' in self._bc['self.DOWN']) or \
            ('p' in self._bc['self.LEFT']) or \
            ('p' in self._bc['self.RIGHT'])

        if 'dwdn' in self._bc[self.UP]:
            outflow_len += mask[1:-1,-2].sum() * d[0] 
            outflow += w[1:-1,-1].sum() * d[0]

        if 'dwdn' in self._bc[self.DOWN]:
            outflow_len += mask[1:-1,1].sum() * d[0]
            outflow -= w[1:-1,0].sum() * d[0]

        if 'dudn' in self._bc[self.LEFT]:
            outflow_len += mask[1,1:-1].sum() * d[1]
            outflow += u[0,1:-1].sum() * d[1]

        if 'dudn' in self._bc[self.RIGHT]:
            outflow_len += mask[-2,1:-1].sum() * d[1]
            outflow += u[-1,1:-1].sum() * d[1]

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
        R = np.zeros( p.shape, np.float )
        R[1:-1,1:-1] = ( u[1:,1:-1] - u[:-1,1:-1] ) / d[0] + \
            ( w[1:-1,1:] - w[1:-1,:-1] ) / d[1]
        

        # for mass conservation
        if not dirichlet_used and (outflow_len == 0.0 or \
            not self._mass_conservation in (self.MASS_ADD, self.MASS_SCALE)):

            R[1:-1,1:-1] -= R[1:-1,1:-1].sum() / R[1:-1,1:-1].size

        return -1./dt * R

    
    def _cal_nonlinear_terms_(self, gamma=0.0 ):
        """
        calculate the nonlinear terms for the u and w-equations.
        """
        d = self.grid.d
        u, w = self.u, self.w

        du = np.zeros( self.u.shape, np.float )
        dw = np.zeros( self.w.shape, np.float )

        ###########
        # u-equation::nonlinear terms → ∂/∂x(u²) + ∂/∂z(uw)
        # ∂/∂x(u²):
        du[1:-1,1:-1] -= 0.25 * ( (u[1:-1,1:-1]-u[2:,1:-1])**2 - \
                                (u[1:-1,1:-1]-u[:-2,1:-1])**2 ) / d[0]
        
        # at the bounadry: ∂/∂x(u²) = 2u∂/∂x(u) = -2u∂/∂z(w)
        du[0,1:-1]  += u[-1,1:-1] * ( (w[0,1:] + w[1,1:]) - \
                                (w[0,:-1] + w[1,:-1]) ) / d[1]
        du[-1,1:-1] += u[-1,1:-1] * ( (w[-2,1:] + w[-1,1:]) - \
                                     (w[-2,:-1] + w[-1,:-1]) ) / d[1]

        # ∂/∂y(uv):
        du[:,1:-1] -= 0.25 * ( (u[:,1:-1] + u[:,2:]) * (w[:-1,1:] + w[1:,1:]) - \
            (u[:,1:-1] + u[:,:-2]) * (w[:-1,:-1] + w[1:,:-1]) ) / d[1]        

        ###########
        # w-equation::nonlinear terms → ∂/∂z(w²) + ∂/∂x(uw)
        # ∂/∂z(w²):
        dw[1:-1,1:-1] -= 0.25 * ( (w[1:-1,1:-1] + w[1:-1,2:])**2 - \
                                 (w[1:-1,1:-1] + w[1:-1,:-2])**2 ) / d[1]
        
        # at the bounadry: ∂/∂z(w²) = 2w∂/∂z(w) = -2w∂/∂x(u)
        dw[1:-1,0]  += w[1:-1,0] * ( (u[1:,0] + u[1:,1]) - \
                                   (u[:-1,0] + u[:-1,1]) ) / d[0]
        dw[1:-1,-1] += w[1:-1,-1] * ( (u[1:,-2] + u[1:,-1]) - \
                                     (u[:-1,-2] + u[:-1,-1]) ) / d[0]

        # ∂/∂x(uw):
        dw[1:-1,:] -= 0.25 * ( (w[1:-1,:] + w[2:,:]) * (u[1:,:-1] + u[1:,1:]) - \
            (w[1:-1,:] + w[:-2,:]) * (u[:-1,:-1] + u[:-1,1:]) ) / d[0] 


        # // TODO: need to implement the upwind scheme for stabilisation.
        

        return du, dw 


    def compute_cfl_dt_(self, safety=0.8):
        """
        Calculate a time stepping value that should give
        a stable (and correct!) simulation. It will get called every iteration.
        PS: good idea to call it once 10 or likewise iterations → could be
        in more-effective: // TODO requires testing
        """

        d = self.grid.d
        u_ = abs(self.u) + 1.e-7 * (self.u == 0.0) # avoid division by zero
        w_ = abs(self.v) + 1.e-7 * (self.w == 0.0)

        # may be multi-phase requires different criretia
        # // TODO: need to dig some literatures.

        nu = self.mu / self.rho

        dt_cfl = min( d[0] / u_.max(), d[1] / w_.max() )
        dt_vis = 0.5 / (  nu * (d**-2).sum() )

        return min( dt_cfl, dt_vis ) * safety


    #
    # now, simuation starts, finally!
    #
    def ns2d_simuation(self, dt, beta=1., gamma=0.):
        """
        Simulation start
        Args:
            dt ([float]): [time step value of the simulation.]

            β:  factor in the projection method
                See "H. P. Langtangen, K.-A. Mardal and R. Winther:
                Numerical Methods for Incompressible Viscous Flow"
            
            γ:  Upwind differencing factor
                See "Numerical Simulation in Fluid Dynamics: A Practical 
                Introduction" (1997) by Griebel, Dornsheifer and Neunhoeffer.
        """

        d, mask, chi = self.grid.d, self.mask, self.chi
        u, w, p = self.u, self.w, self.p

        # imposed boundary conditions:
        self.bc2d._update_vel_bc_(u, w)
        self.bc2d._update_pressure_bc_(p)

        # if passive tracer used: will add later on.
        #if self.use_passive_tracer:
            # add the tracer advection code.
            #print( 'no passive tracer' )
        
        # if multi-phase used: will add later on.
        #if self.multi_phase:
            # add multiphase code here
            #print( 'no multi-phase simulation' )

        # let's keep it for multi-phase case:
        # for χ=1, it would be for one-phase.
        rho = self.rho1 * (1. - chi) + self.rho2 * chi
        mu  = self.mu1  * (1. - chi) + self.mu2  * chi

        # RHS for the intermediate velocity equations.
        du0, dw0  = self._cal_nonlinear_terms_(gamma)
        
        ###
        ### all the terms in RHS of the momentum equation
        # pressure gradient
        fc = Force(u, w, p)
        du1, dw1 = fc._cal_gradP_(rho, beta)
        # viscous force
        du2, dw2 = fc._cal_vis_force_(mu, rho)

        du = dt * ( du0 + du1 + du2 )
        dw = dt * ( dw0 + dw1 + dw2 )

        ###
        ### end here!

        # and ofcourse, gravity!
        du += dt * self.gra[0]
        dw += dt * self.gra[1]

        # update time
        self.sim_time += dt
        # update iteration
        self.iteration += 1

        # imposed bcs to intermediate velocity.
        self.bc2d._update_intermediate_vel_bc_() 

        # calculate RHS of PPE
        ppe = self._cal_RHS_poisson_eq_()

        # Calculate boundary conditions for 'Ξ' and store the values in
        # the ghost cells.   
        # 
        xi = np.zeros(p.shape, np.float64)
        self.bc2d._update_xi_bc(p, xi, beta)  

        # Calculate pressure correction. (phi = p^{n+1} - β * p^(n))
        # 'mask' defines where to use Dirichlet and Neumann boundary conditions.
        p *= beta               
        p += poisson(xi, ppe, d, mask, rho, self.poisson_data, 3) # LU decomposition

        # Correct velocity on boundary for Dirichlet pressure boundary condition
        # do not worry it's taken care by mask function
        u[0,1:-1] -= dt * (xi[1,1:-1] - xi[0,1:-1]) / \
            (d[0] * rho[0,1:-1]) * (mask[1,1:-1] & 1)

        u[-1,1:-1] -= dt * (xi[-1,1:-1] - xi[-2,1:-1]) / \
            (d[0] * rho[-1,1:-1]) * (mask[-2,1:-1] & 1)

        w[1:-1,0] -= dt * (xi[1:-1,1] - xi[1:-1,0]) / \
            (d[1] * rho[1:-1,0]) * (mask[1:-1,1] & 1)
        w[1:-1,-1] -= dt * (xi[1:-1,-1] - xi[1:-1,-2]) / \
            (d[1] * rho[1:-1,-1]) * (mask[1:-1,-2] & 1)



    

    













        



