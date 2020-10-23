"""
Extra tools that are useful in hydrodynamical problems.
"""

import numpy as np

class GlobalFLowProperty:
    """
    Flow property on the grid scale
    Parameters
    ----------
    solver : solver object
        Problem solver
    cadence : int, optional
        Iteration cadence for property evaluation (default: 1)
    Examples
    --------
    >>> flow = GlobalFlowProperty(solver)
    >>> flow.add_property('sqrt(u*u + w*w) * L / ν', name='Re')
    ...
    >>> flow.max('Re')
    4000
    """


    def __init__(self, solver, domain) :
        self.solver = solver
        self.domain = domain 

    def min(self, data):
        """Compute global min of a property on the grid."""
        return data.min()

    def max(self, data):
        """Compute global max of a property on the grid."""
        return data.max()

    def grid_average(self, data):
        """Compute global avergae of a property on the grid."""
        global_sum  = np.sum()
        global_size = data.size

        return global_sum/global_size

    def volume_average(self, data):
        """Compute global min of a property on the grid."""


