"""
This file contains the information about the 
2D computational grid points.
"""

import numpy as np
import os

from numpy.core.fromnumeric import shape

class Domain(object):

    def __init__(self, x_coord, z_coord, sze):
        """
        This creates a domain of uniform cartesian grids

        x_coord = [ x_min, x_max ],
        z_coord = [ z_min, z_max ],
        Size = No of grid points in x and z-directions.
        """

        super(Domain, self).__init__()
        self.x_coord = np.array( x_coord, np.float )
        self.z_coord = np.array( z_coord, np.float )
        self.sze    = np.array( sze, np.int )

    def __repr__(self):
        return "Domain(" + ", ".join([repr(self.x_coord), repr(self.z_coord), repr(self.sze)]) + ")"

    def __getitem__(self, index):
        """
        Return node (x_i, z_j) where index = (i,j)
        'i' and 'j' do not need to be integer.
        """

        i = np.array( index, float )
        min_coord = np.array( [ self.x_coord[0], self.z_coord[0] ]  )
        max_coord = np.array( [ self.x_coord[1], self.z_coord[1] ]  )
        return min_coord + i/self.sze * (max_coord - min_coord)

    def _get_shape_(self):
        """
        Return number of nodes in the x and y-directions.
        """
        return tuple(self.sze+1)

    def _get_node_spacing_(self):
        """
        Return node spacing in the the x and y-directions
        """
        min_coord = np.array( [ self.x_coord[0], self.z_coord[0] ]  )
        max_coord = np.array( [ self.x_coord[1], self.z_coord[1] ]  )
        return (max_coord - min_coord)/self.sze

    def _get_coordinate_(self, i):
        """
        Return one of the coordinates for the grid. i = [0,1]
        0 is for x-coordinate and 1 gives z-coordinate 
        For example _get_coordinate_(1)[8,9] will give
        the y-coorindate of the node (8,9)
        """

        gridP  = np.zeros( self.sze + 1, np.float )
        if i == 0:
            lamb = np.linspace( self.x_coord[0], self.x_coord[1], self.size+1 )
        elif i == 1:
            lamb = np.linspace( self.z_coord[0], self.z_coord[1], self.size+1 ) 
        else:
            raise ValueError('i must 0 or 1')

        sl = ( slice(None), )
        n  = len(self.sze)-1-i 
        for j in range(len(lamb)): gridP[sl*i+(j,)+sl*n] = lamb[j]

        return gridP


    shape = property( fget = _get_shape_ )
    d = property( fget = _get_node_spacing_ )
    x = property( fget = lambda self: self._get_coordinate_(0) )
    z = property( fget = lambda self: self._get_coordinate_(1) )

 
