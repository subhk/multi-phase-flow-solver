"""
This file contains different forcing terms of 
2D Navier-Stokes equation.
"""

class _Force_(object):


    def __init__(self, grid):

        self.grid = grid

        self._bc = {self.LEFT:{}, self.RIGHT:{}, self.UP:{}, self.DOWN:{}}
    
    def _cal_νΔu_(self, parameter_list):
        """
        viscous force
        """
        

