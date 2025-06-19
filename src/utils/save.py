import glob
import json
import os

import numpy as np

from src.utils.utils import to_tuple

"""Functions used to save and load data"""

def save_kwargs(**kwargs):
    GP_FILE = 'src.genetics'
    def func_to_string(obj):
        """Recursively replace functions with its name preceded by 'gp.'"""
        if type(obj) == dict:
            obj = obj.copy()
            for key in obj:
                obj[key] = func_to_string(obj[key])
            return obj
        elif type(obj) == list:
            obj = obj.copy()
            for i, item in enumerate(obj):
                obj[i] = func_to_string(item)
            return obj
        elif type(obj) == np.ndarray:
            return to_tuple(obj)
        elif hasattr(obj, '__name__'):
            return f'{GP_FILE}.{obj.__name__}'
        else:
            return obj
    path = f'{kwargs['saves_path']}/{kwargs['name']}/'
    os.makedirs(path, exist_ok=True)
    with open(path + 'kwargs.json', 'w') as f:
        json.dump(func_to_string(kwargs.copy()), f, indent=4)


def save_run(path, pops, fits, **kwargs):
    # Each test is saved in its own directory which is passed through the path
    path = f'{path}/{kwargs["seed"]}/'
    print('Saving run to ' + path)
    os.makedirs(path, exist_ok=True)
    np.save(path + 'pops', pops)
    # np.save(path + 'returned_value', all_returned_values)
    # np.save(path + 'prev_fit', all_prev_fits)
    np.save(path + 'fits', fits)


def load_kwargs(name, saves_path):
    GP_FILE = 'src.genetics'
    def string_to_func(obj):
        """Recursively replace strings preceded by $ to a function of the same name from gp"""
        if type(obj) is dict:
            for key in obj:
                obj[key] = string_to_func(obj[key])
        elif type(obj) is list:
            for i, item in enumerate(obj):
                obj[i] = string_to_func(item)
        elif type(obj) is type('') and obj.startswith(GP_FILE + '.'):
            # module = __import__(GP_FILE)
            import src.genetics as module
            return getattr(module, obj[len(GP_FILE) + 1:])
        return obj
    path = f'{saves_path}/{name}/'
    print('Loading kwargs')
    with open(path + 'kwargs.json', 'rb') as f:
        kwargs = string_to_func(json.load(f))
    kwargs['saves_path'] = saves_path
    return kwargs


def load_runs(**kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    # print('Loading data:')
    pops = []
    fits = []
    tests = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test in tests:
        pops.append([])
        fits.append([])
        test_path = f'{kwargs['saves_path']}/{kwargs['name']}/data/{test}/*/'
        for run_file_name in glob.glob(test_path):
            print(f'Loading {run_file_name}')
            pops[-1].append(np.load(run_file_name+'pops.npy', allow_pickle=True))
            fits[-1].append(np.load(run_file_name+'fits.npy'))
    # pops = np.array(pops, dtype=[('verts','object'),('edges','object')])
    pops = np.array(pops, dtype=object)
    fits = np.array(fits)
    return pops, fits