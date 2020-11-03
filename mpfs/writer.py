import os
import h5py
import pathlib
import numpy as np
from collections import defaultdict
import re
import shutil
import uuid

import logging
logger = logging.getLogger(__name__.split('.')[-1])

FILEHANDLER_MODE_DEFAULT = 'overwrite'
FILEHANDLER_TOUCH_TMPFILE = False

class Evaluator:

    def __init__(self, domain, vars):

        self.domain = domain
        self.vars = vars
        self.handlers = []
        self.groups = defaultdict(list)

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

    def evaluate_group(self, group, **kwargs):
        """Evaluate all handlers in a group."""
        handlers = self.groups[group]
        self.evaluate_handlers(handlers, **kwargs)

    def evaluate_scheduled(self, wall_time, sim_time, iteration, **kwargs):
        """Evaluate all scheduled handlers."""

        scheduled_handlers = []
        for handler in self.handlers:
            # Get cadence devisors
            wall_div = wall_time // handler.wall_dt
            sim_div  = sim_time  // handler.sim_dt
            iter_div = iteration // handler.iter
            # Compare to divisor at last evaluation
            wall_up = (wall_div > handler.last_wall_div)
            sim_up  = (sim_div  > handler.last_sim_div)
            iter_up = (iter_div > handler.last_iter_div)

            if any((wall_up, sim_up, iter_up)):
                scheduled_handlers.append(handler)
                # Update all divisors
                handler.last_wall_div = wall_div
                handler.last_sim_div  = sim_div
                handler.last_iter_div = iter_div

        self.evaluate_handlers(scheduled_handlers, wall_time=wall_time, sim_time=sim_time, iteration=iteration, **kwargs)


    def evaluate_handlers(self, handlers, id=None, **kwargs):
            """Evaluate a collection of handlers."""

            # Default to uuid to cache within evaluation, but not across evaluations
            if id is None:
                id = uuid.uuid4()

            tasks = [t for h in handlers for t in h.tasks]
            for task in tasks:
                task['out'] = None

            # Attempt initial evaluation
            tasks = self.attempt_tasks(tasks, id=id)

            # Move all fields to coefficient layout
            fields = self.get_fields(tasks)
            self.require_coeff_space(fields)
            tasks = self.attempt_tasks(tasks, id=id)

            # Oscillate through layouts until all tasks are evaluated
            n_layouts = len(self.domain.dist.layouts)
            oscillate_indices = oscillate(range(n_layouts))
            current_index = next(oscillate_indices)



    @staticmethod
    def get_fields(tasks):
        """Get field set for a collection of tasks."""
        fields = OrderedSet()
        for task in tasks:
            fields.update(task['operator'].atoms(Field))
        return fields

    @staticmethod
    def attempt_tasks(tasks, **kw):
        """Attempt tasks and return the unfinished ones."""
        unfinished = []
        for task in tasks:
            output = task['operator'].attempt(**kw)
            if output is None:
                unfinished.append(task)
            else:
                task['out'] = output
        return unfinished

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
        self.current_path.parent.mkdir()

        #if FILEHANDLER_TOUCH_TMPFILE:
        #    tmpfile = self.base_path.joinpath('tmpfile_p%i' %(comm.rank))
        #    tmpfile.touch()
        
        file = h5py.File(str(self.current_path), 'w-')
        
        #if FILEHANDLER_TOUCH_TMPFILE:
        #    tmpfile.unlink()

        self.setup_file(file)
        file.close()



    def setup_file(self, file):

        domain = self.domain

        # Metadeta
        file.attrs['set_number'] = self.set_num
        file.attrs['handler_name'] = self.base_path.stem
        file.attrs['writes'] = self.file_write_num

        # Scales
        scale_group = file.create_group('scales')
        # Start time scales with shape=(0,) to chunk across writes
        for sn, dtype in [('sim_time', np.float64),
                          ('world_time', np.float64),
                          ('wall_time', np.float64),
                          ('timestep', np.float64),
                          ('iteration', np.int),
                          ('write_number', np.int)]:
            scale = scale_group.create_dataset(name=sn, shape=(0,), maxshape=(None,), dtype=dtype)
            scale.make_scale(sn)
        const = scale_group.create_dataset(name='constant', data=np.array([0.], dtype=np.float64))
        const.make_scale('constant')


        for axis, basis in enumerate(domain.bases):
            scale_group.create_group(basis.name)
            coeff_name = basis.element_label + basis.name
            scale = scale_group.create_dataset(name=coeff_name, data=basis.elements)
            scale.make_scale(coeff_name)

        
        # Tasks
        task_group =  file.create_group('tasks')
        for task_num, task in enumerate(self.tasks):
            layout = task['layout']
            constant = task['operator'].meta[:]['constant']
            scales = task['scales']
            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, scales, constant, index=0)
            if np.prod(write_shape) <= 1:
                # Start with shape[0] = 0 to chunk across writes for scalars
                file_shape = (0,) + tuple(write_shape)
            else:
                # Start with shape[0] = 1 to chunk within writes
                file_shape = (1,) + tuple(write_shape)
            file_max = (None,) + tuple(write_shape)
            dset = task_group.create_dataset(name=task['name'], shape=file_shape, maxshape=file_max, dtype=layout.dtype)
            
            dset.attrs['global_shape'] = gnc_shape
            dset.attrs['start'] = gnc_start
            dset.attrs['count'] = write_count

            # Metadata and scales
            dset.attrs['task_number'] = task_num
            dset.attrs['constant'] = constant
            dset.attrs['grid_space'] = layout.grid_space
            dset.attrs['scales'] = scales


            # Time scales
            dset.dims[0].label = 't'
            for sn in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                dset.dims[0].attach_scale(scale_group[sn])


            # Spatial scales
            for axis, basis in enumerate(domain.bases):
                if constant[axis]:
                    sn = lookup = 'constant'
                else:
                    if layout.grid_space[axis]:
                        sn = basis.name
                        axscale = scales[axis]
                        lookup = '/'.join((sn, str(axscale)))
                        if str(axscale) not in scale_group[sn]:
                            scale = scale_group[sn].create_dataset(name=str(axscale), data=basis.grid(axscale))
                            scale.make_scale(lookup)
                    else:
                        sn = lookup = basis.element_label + basis.name
                dset.dims[axis+1].label = sn
                dset.dims[axis+1].attach_scale(scale_group[lookup])



    def process(self, world_time, wall_time, sim_time, timestep, iteration, **kw):
        """Save task outputs to HDF5 file."""

        file = self.get_file()
        self.total_write_num += 1
        self.file_write_num += 1
        file.attrs['writes'] = self.file_write_num
        index = self.file_write_num - 1

        # Update time scales
        sim_time_dset = file['scales/sim_time']
        world_time_dset = file['scales/world_time']
        wall_time_dset = file['scales/wall_time']
        timestep_dset = file['scales/timestep']
        iteration_dset = file['scales/iteration']
        write_num_dset = file['scales/write_number']

        sim_time_dset.resize(index+1, axis=0)
        sim_time_dset[index] = sim_time
        world_time_dset.resize(index+1, axis=0)
        world_time_dset[index] = world_time
        wall_time_dset.resize(index+1, axis=0)
        wall_time_dset[index] = wall_time
        timestep_dset.resize(index+1, axis=0)
        timestep_dset[index] = timestep
        iteration_dset.resize(index+1, axis=0)
        iteration_dset[index] = iteration
        write_num_dset.resize(index+1, axis=0)
        write_num_dset[index] = self.total_write_num

        # Create task datasets
        for task_num, task in enumerate(self.tasks):
            out = task['out']
            out.set_scales(task['scales'], keep_data=True)
            out.require_layout(task['layout'])

            dset = file['tasks'][task['name']]
            dset.resize(index+1, axis=0)

            memory_space, file_space = self.get_hdf5_spaces(out.layout, task['scales'], out.meta[:]['constant'], index)
            if self.parallel:
                dset.id.write(memory_space, file_space, out.data, dxpl=self._property_list)
            else:
                dset.id.write(memory_space, file_space, out.data)

        file.close()



    def get_write_stats(self, layout, scales, constant, index):
        """Determine write parameters for nonconstant subspace of a field."""

        constant = np.array(constant)
        # References
        gshape = layout.global_shape(scales)
        lshape = layout.local_shape(scales)
        start = layout.start(scales)
        first = (start == 0)

        # Build counts, taking just the first entry along constant axes
        write_count = lshape.copy()
        write_count[constant & first] = 1
        write_count[constant & ~first] = 0

        # Collectively writing global data
        global_nc_shape = gshape.copy()
        global_nc_shape[constant] = 1
        global_nc_start = start.copy()
        global_nc_start[constant & ~first] = 1

        # Independently writing local data
        write_shape = write_count
        write_start = 0 * start

        return global_nc_shape, global_nc_start, write_shape, write_start, write_count


    def get_hdf5_spaces(self, layout, scales, constant, index):
        """Create HDF5 space objects for writing nonconstant subspace of a field."""

        constant = np.array(constant)
        # References
        lshape = layout.local_shape(scales)
        start = layout.start(scales)
        gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, scales, constant, index)

        # Build HDF5 spaces
        memory_shape = tuple(lshape)
        memory_start = tuple(0 * start)
        memory_count = tuple(write_count)
        memory_space = h5py.h5s.create_simple(memory_shape)
        memory_space.select_hyperslab(memory_start, memory_count)

        file_shape = (index+1,) + tuple(write_shape)
        file_start = (index,) + tuple(write_start)
        file_count = (1,) + tuple(write_count)
        file_space = h5py.h5s.create_simple(file_shape)
        file_space.select_hyperslab(file_start, file_count)

        return memory_space, file_space







