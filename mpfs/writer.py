import os
import h5py
import pathlib
import numpy as np
import re
import shutil
import uuid

import logging
logger = logging.getLogger(__name__.split('.')[-1])

FILEHANDLER_MODE_DEFAULT = True

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

    def __init__(self, base_path, *args, max_writes=np.inf, max_size=2**30, parallel=None, mode=None, **kw):

        Handler.__init__(self, *args, **kw)

        # Resolve defaults from config
        if mode is None:
            mode = FILEHANDLER_MODE_DEFAULT
        else:
            raise ValueError("At present it takes default write modes.")

        # Check base_path
        base_path = pathlib.Path(base_path).resolve()
        if any(base_path.suffixes):
            raise ValueError("base_path should indicate a folder for storing HDF5 files.")

        # Attributes
        self.base_path = base_path
        self.max_writes = max_writes
        self.max_size = max_size
        self._sl_array = np.zeros(1, dtype=int)

        # Resolve mode
        mode = mode.lower()
        if mode not in ['overwrite', 'append']:
            raise ValueError("Write mode {} not defined.".format(mode))

        set_pattern = '%s_s*' % (self.base_path.stem)
        sets = list(self.base_path.glob(set_pattern))
        if mode == "overwrite":
            for set in sets:
                if set.is_dir():
                    shutil.rmtree(str(set))
                else:
                    set.unlink()
            set_num = 1
            total_write_num = 0
        elif mode == "append":
            set_nums = []
            if sets:
                for set in sets:
                    m = re.match("{}_s(\d+)$".format(base_path.stem), set.stem)
                    if m:
                        set_nums.append(int(m.groups()[0]))
                max_set = max(set_nums)
                joined_file = base_path.joinpath("{}_s{}.h5".format(base_path.stem,max_set))
                p0_file = base_path.joinpath("{0}_s{1}/{0}_s{1}_p0.h5".format(base_path.stem,max_set))
                if os.path.exists(str(joined_file)):
                    with h5py.File(str(joined_file),'r') as testfile:
                        last_write_num = testfile['/scales/write_number'][-1]
                elif os.path.exists(str(p0_file)):
                    with h5py.File(str(p0_file),'r') as testfile:
                        last_write_num = testfile['/scales/write_number'][-1]
                else:
                    last_write_num = 0
                    logger.warn("Cannot determine write num from files. Restarting count.")
            else:
                max_set = 0
                last_write_num = 0
            set_num = max_set + 1
            total_write_num = last_write_num
        
        self.set_num = set_num
        self.total_write_num = total_write_num


    
    def check_file_limits(self):
        """Check if write or size limits have been reached."""
        
        write_limit = (self.file_write_num >= self.max_writes)
        size_limit = (self.current_path.stat().st_size >= self.max_size)
        self._sl_array[0] = size_limit
        size_limit = self._sl_array[0]
        
        return (write_limit or size_limit)



    def get_file(self):
        """Return current HDF5 file, creating if necessary."""
        
        # Create new file if necessary
        if os.path.exists(str(self.current_path)):
            if self.check_file_limits():
                self.set_num += 1
                self.create_current_file()
        else:
            self.create_current_file()
        
        # Open current file
        h5file = h5py.File(str(self.current_path), 'r+')
        self.file_write_num = h5file['/scales/write_number'].shape[0]
        
        return h5file


    @property
    def current_path(self):

        domain = self.domain
        set_num = self.set_num

        # Save in folders for each filenum in base directory
        folder_name = '%s_s%i' %(self.base_path.stem, set_num)
        folder_path = self.base_path.joinpath(folder_name)
        file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, set_num)
            
        return folder_path.joinpath(file_name)



    def create_current_file(self):
        """Generate new HDF5 file in current_path."""

        self.file_write_num = 0
        comm = self.domain.distributor.comm_cart
        if self.parallel:
            file = h5py.File(str(self.current_path), 'w-', driver='mpio', comm=comm)
        else:
            # Create set folder
            with Sync(comm):
                if comm.rank == 0:
                    self.current_path.parent.mkdir()
            if FILEHANDLER_TOUCH_TMPFILE:
                tmpfile = self.base_path.joinpath('tmpfile_p%i' %(comm.rank))
                tmpfile.touch()
            file = h5py.File(str(self.current_path), 'w-')
            if FILEHANDLER_TOUCH_TMPFILE:
                tmpfile.unlink()
                
        self.setup_file(file)
        file.close()


    def _store_var_(self):

        u_ = [[self.u]]
        

    def _write_var_(self):

        u_ = []
        w_ = []






