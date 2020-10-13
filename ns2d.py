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
                        'σ': 0.0, 'g†': [0.0, 0.0],
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


    def _set_bc_(self, bounadries, **args):
        """
        Set bonadry conditions for the computational domain.
        The 'bounadries' should be 'east', 'west', 'north', 'south', 
        or a combination with '+' sign, for exmaple: 'east'+'west'+'south'
        with a common boundary conditions. 
        The keywords should be 'u', 'w', 'p', 'χ', 
        The condition should be either constant, or can be function of
        (x,y,t).
        """

        for bounadry in bounadries.split('+'):

            if bounadry.self._bc:
                self._bc[bounadry].update(args)

            if 'p' in args:
                if bounadry == self.UP:
                    self.mask[:,-1] = 3
                elif bounadry == self.DOWN:
                    self.mask[:,0]= 3
                elif bounadry == self.LEFT:
                    self.mask[0,:] = 3
                elif bounadry == self.RIGHT:
                    self.mask[-1,:] = 3

            if 'w' in args:
                if bounadry in (self.UP, self.DOWN):
                    self._bc[bounadry]['dpdn'] = 0

            if 'u' in args:
                if bounadry in (self.LEFT, self.RIGHT):
                    self._bc[bounadry]['dpdn'] = 0

    
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
            # so, do not rquire any return function, cool!
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


    def _cal_νΔu_(self, μ, ρ, dt):
        """
        calculate viscous force 
        """

        δu = np.zeros( self.u.shape, np.float )
        δw = np.zeros( self.w.shape, np.float )

        μ_avg = 0.25 * ( μ[1:,1:] + μ[1:,:-1] + μ[:-1,1:] + μ[:-1,:-1] )

        δ = self.grid.δ
        u, w = self.u, self.w

        # viscous terms in u-equation: (1/ρ)∂/∂x(2μ∂u/∂x) + (1/ρ)∂/∂y(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂x(2μ∂u/∂x):
        #  
        δu[1:-1,1:-1] += ( μ[2:-1,1:-1] * (u[2:,1:-1] - u[1:-1,1:-1] ) - \
                    μ[1:-2,1:-1] * ( u[1:-1,1:-1] - u[:-2,1:-1]) ) / \
                    ( 0.5 * (δ[0]**2) * ρ[1:-1,1:-1] )

        # ∂/∂y(μ(∂u/∂z+∂w/∂x)):
        δu[1:-1,1:-1] += ( μ_avg[:,1:] * ( (u[:,2:] - u[:,1:-1]) / (δ[1]**2) + \
                    (w[1:,1:] - w[:-1,1:]) / (δ[0] * δ[1]) ) - \
                    μ_avg[:,:-1] * ( (u[:,1:-1] - u[:,:-2]) / (δ[1]**2) + \
                    (w[1:,:-1] - w[:-1,:-1]) / (δ[0]*δ[1]) ) ) / \
                    ρ[:,1:-1]

        
        # viscous terms in w-equation: (1/ρ)∂/∂y(2μ∂w/∂z) + (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂y(2μ∂w/∂z):
        #
        δw[1:-1,1:-1] += (μ[1:-1,2:-1] * (w[1:-1,2:] - w[1:-1,1:-1]) - \
                        μ[1:-1,1:-2] * (w[1:-1,1:-1] - w[1:-1,:-2])) / \
                        (0.5 * (δ[1]**2) * ρ[1:-1,1:-1])

        # (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        δw[1:-1,1:-1] += (μ_avg[1:,:] * ( (w[2:,:] - w[1:-1,:])/(δ[0]**2) + \
                      (u[1:,1:] - u[1:,:-1]) / (δ[0]*δ[1]) ) - \
                      μ_avg[:-1,:] * ( (w[1:-1,:] - w[:-2,:]) / (δ[0]**2) + \
                      (u[:-1,1:] - u[:-1,:-1])/(δ[0]*δ[1]) ) ) / \
                      ρ[1:-1,:]

        return δu * dt, δw * dt


    def _update_vel_bc_(self):

        u, w = self.u, self.w

        # top-boundary condition
        _bc_ = self._bc[self.UP]
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[:,-1] = w[:,-2] # extrapolate
    
        # TODO : require to handle 'w'-condition
        
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[:,-1] = u[:,-2] # extrapolate
        elif 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[0]):
                    node = self.grid[i, u.shape[1]-2]
                    u[i,-1] = 2. * fun_(node[0], node[1], self.time) - u[i,-2]
            else:
                u[:,-1] = 2. * fun_ - u[:,-2]

        # bottom-boundary condition
        _bc_ = self._bc[self.DOWN]
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[:,0] = w[:,1]
        
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[:,0] = u[:,1]
        elif 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[0]):
                    node = self.grid[i, 0]
                    u[i,0] = 2. * fun_(node[0], node[1], self.time) - u[i,1]
            else:
                u[:,0] = 2. * fun_ - u[:,1]

        # left-boundary condition
        _bc_ = self._bc[self.LEFT]
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[-1,:] = u[-2,:]
        
        if 'dwdn' in _bc_:
            if _bc_['dvdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[-1,:] = w[-2,:]
        elif 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(w.shape[1]):
                    node = self.grid[w.shape[0]-2, i]
                    w[-1,i] = 2. * fun_(node[0], node[1], self.time) - w[-2,i]
            else:
                w[-1,:] = 2. * fun_ - w[-2,:]

        # right-boundary condition
        _bc_ = self._bc[self.RIGHT]
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[0,:] = u[1,:]
        
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[0,:] = w[1,:]
        elif 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(v.shape[1]):
                    node = self.grid[0, i]
                    w[0,i] = 2. * fun_(node[0], node[1], self.time) - w[1,i]
            else:
                w[0,:] = 2. * fun_ - w[1,:]


    def _update_pressure_bc_(self):
        p = self.p

        # top-boundary condition
        _bc_ = self._bc[self.UP]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[:,-1] = p[:,-2]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self._grid[i-0.5, p.shape[1]-2]
                    p[i,-1] = 2. * fun_(node[0], node[1], self.time) - p[i,-2]
            else:
                p[:,-1] = 2. * fun_ - p[:,-2]

        # bottom-boundary condition
        _bc_ = self._bc[self.DOWN]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[:,0] = p[:,1]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, 0]
                    p[i,0] = 2. * fun_(node[0], node[1], self.time) - p[i,1]
            else:
                p[:,0] = 2. * fun_ - p[:,1]

        # left-boundary condition
        _bc_ = self._bc[self.LEFT]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[-1,:] = p[-2,:]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[p.shape[0]-2, i-0.5]
                    p[-1,i] = 2. * fun_(node[0], node[1], self.time) - p[-2,i]
            else:
                p[-1,:] = 2. * fun_ - p[-2,:]

        # right-boundary condition
        _bc_ = self._bc[self.RIGHT]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[0,:] = p[1,:]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[0, i-0.5]
                    p[0,i] = 2 * fun_(node[0], node[1], self.time) - p[1,i]
            else:
                p[0,:] = 2. * fun_ - p[1,:]


    def _update_Ξ_bc_(self, Ξ, β ):

        p = self.p
        
        # top-boundary condition
        _bc_ = self._bc[self.UP]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, p.shape[1]-2]
                    Ξ[i,-1] = 2. * fun_(node[0], node[1], self.time) - β * (p[i,-2] + p[i,-1])
            else:
                Ξ[:,-1] = 2 * fun_ - β * (p[:,-2] + p[:,-1])
        
        # bottom-boundary condition
        _bc_ = self._bc[self.DOWN]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, 0]
                    Ξ[i,0] = 2. * fun_(node[0], node[1], self.time) - β * (p[i,1] + p[i,0])
            else:
                Ξ[:,0] = 2. * fun_ - β * (p[:,1] + p[:,0])        

        # left-boundary condition
        _bc_ = self._bc[self.LEFT]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[p.shape[0]-2, i-0.5]
                    Ξ[-1,i] = 2. * f(node[0], node[1], self.time) - β * (p[-2,i] + p[-1,i])
            else:
                Ξ[-1,:] = 2 * fun_ - β * (p[-2,:] + p[-1,:])        

        # right-boundary condition
        _bc_ = self._bc[self.RIGHT]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[0, i-0.5]
                    Ξ[0,i] = 2. * fun_(node[0], node[1], self.time) - β * (p[1,i] + p[0,i])
            else:
                Ξ[0,:] = 2 * fun_ - β * (p[1,:] + p[0,:])        



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

        
        

        



