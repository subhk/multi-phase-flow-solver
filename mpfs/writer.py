import os
import h5py
import numpy as np
import re
import shutil
import uuid

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class write(object):

    def  __init__(self, u, w, p, time, save_rate):
        
        self.u = u
        self.w = w
        self.p = p
        self.time = time
        save_rate

    def _store_var_(self):

        u_ = [[self.u]]
        

    def _write_var_(self):

        u_ = []
        w_ = []






