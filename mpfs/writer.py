import os
import h5py
import numpy as np
import re
import shutil
import uuid

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Evaluator:


    def add_file_handler(self, filename, **kw):
        """Create a file handler and add to evaluator."""

        FH = FileHandler(filename, self.domain, self.vars, **kw)
        return self.add_handler(FH)

    def add_handler(self, handler):
        """Add a handler to evaluator."""

        self.handlers.append(handler)
        # Register with group
        if handler.group is not None:
            self.groups[handler.group].append(handler)
        return handler


class Handler:

    def __init__(self, grid, vars, group=None, wall_dt=np.inf, sim_dt=np.inf, iter=np.inf):

        # Attributes
        self.grid = grid
        self.vars = vars
        self.group = group
        self.wall_dt = wall_dt
        self.sim_dt = sim_dt
        self.iter = iter

        self.tasks = []
        # Set initial divisors to be scheduled for sim_time, iteration = 0
        self.last_wall_div = -1
        self.last_sim_div = -1
        self.last_iter_div = -1


    def add_task(self, name=None, scales=None):
        """Add task to handler."""

        # Default name
        if name is None:
            name = str('subha')

        # Build task dictionary
        task = dict()
        task['name'] = name
        task['scales'] = self.grid.remedy_scales(scales)

        self.tasks.append(task)

    
    def add_tasks(self, tasks, **kwargs):
        """Add multiple tasks."""

        name = kwargs.pop('name', '')
        for task in tasks:
            tname = name + str(task)
            self.add_task(task, name=tname, **kwargs)

    def add_system(self, system, **kwargs):
        """Add fields from a FieldSystem."""

        self.add_tasks(system.fields, **kwargs)



class FileHandler(Handler):

    def  __init__(self, u, w, p, time, save_rate):


        



    def _store_var_(self):

        u_ = [[self.u]]
        

    def _write_var_(self):

        u_ = []
        w_ = []






