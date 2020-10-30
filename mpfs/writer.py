from logging import FileHandler, Handler
import os
import h5py
import numpy as np


class Writer(object):

    def  __init__(self, ):



    def 




class FileHandler(Handler):

    """
    Handler that writes tasks to an HDF5 file.
    Parameters
    ----------
    base_path : str
        Base path for analyis output folder
    max_writes : int, optional
        Maximum number of writes per set (default: infinite)
    max_size : int, optional
        Maximum file size to write to, in bytes (default: 2**30 = 1 GB).
        (Note: files may be larger after final write.)
    mode : str, optional
        'overwrite' to delete any present analysis output with the same base path.
        'append' to begin with set number incremented past any present analysis output.
        Default behavior set by config option.
    """

    

