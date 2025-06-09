from matplotlib import pyplot as plt

from tm import TM
from utils import *

"""
Functions used in the evolution of Turing Machine based genetic programming
"""


#
# Initialization Functions
#

# def random_transition(**kwargs):
#     """Helper function for generating only a single transition"""
#     return [
#         choice(kwargs['states'], kwargs['rng']),
#         choice(kwargs['symbols'], kwargs['rng']),
#         choice(kwargs['states'], kwargs['rng']),
#         choice(kwargs['symbols'], kwargs['rng']),
#         *[choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
#     ]


def _random_transition(**kwargs):
    """Helper function for generating only a single transition"""
    return [
        choice(kwargs['states'], kwargs['rng']),
        to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape'])),
        choice(kwargs['states'], kwargs['rng']),
        to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape'])),
        *[choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
    ]


def random_trans(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    trans = [_random_transition(**kwargs) for _ in range(init_len)]
    return trans


#
# Fitness Functions
#

def _array_diff(a, b):
    """Helper function to find the differences in two arrays"""
    a = np.array(a)
    b = np.array(b)
    a = np.trim_zeros(a)
    b = np.trim_zeros(b)
    b_pad = [(0,i) for i in np.maximum(0, np.array(a.shape) - b.shape)]
    a_pad = [(0,i) for i in np.maximum(0, np.array(b.shape) - a.shape)]
    a = np.pad(a, a_pad)
    b = np.pad(b, b_pad)
    diff = np.sum(abs(0 + a - b))
    return diff


def diff_zeros(pop, **kwargs):
    """Difference in zeros"""
    fits = np.empty(len(pop))
    for i,trans in enumerate(pop):
        tape = TM(trans)(kwargs['tm_timeout'])
        fit = _array_diff(tape != 0, np.array(kwargs['target']) != 0)
        fits[i] = fit
    return fits


def diff_values(pop, **kwargs):
    """Difference in values"""
    fits = np.empty(len(pop))
    for i,trans in enumerate(pop):
        tape = TM(trans)(kwargs['tm_timeout'])
        fit = _array_diff(tape, kwargs['target'])
        fits[i] = fit
    return fits


def gen_maze(shape=None, rng=np.random.default_rng(), maze=None, pos=None, init_call=True):
    if init_call:
        maze = np.pad(np.ones(shape, int), 1)
        pos = (2,)*len(shape)
    # Set position to zero
    maze[pos] = 0
    # Cartesian product of [-2, 0, 2] repeated N times
    neighbors = np.array(np.meshgrid(*([[-2, 0, 2]] * len(shape)))).T.reshape(-1, len(shape))
    # Only keep rows with exactly one nonzero element
    neighbors = neighbors[np.sum(neighbors != 0, axis=1) == 1]
    # Advanced numpy indexing to get all neighbors
    neighbors = tuple((neighbors + [pos]).T)
    while True:
        # List of all valid new position
        options = np.array(neighbors).T[maze[neighbors] == 1]
        if len(options) == 0:
            # Backtrack if there are no options
            # Trim the edges of the maze if this is the initial function call
            if init_call:
                maze = maze[1:-1, 1:-1]
            return maze
        else:
            # Select new location
            new_pos = choice(options, rng)
            # Create a path from the current position to the new position
            maze[tuple((new_pos + pos) // 2)] = 0
            # Recursively call at the new point
            gen_maze(shape, rng, maze, tuple(new_pos), False)


def _format_maze(maze):
    shape = maze.shape
    maze = maze.repeat(3, axis=0).repeat(3, axis=1)
    # Cartesian product of [-1,0,1] repeated N times
    shifts = np.array(np.meshgrid(*([[-1, 0, 1]] * len(shape)))).T.reshape(-1, len(shape))
    # Remove the row with all zeros
    shifts = shifts[np.sum(shifts, axis=1) != 0]
    # Give a unique value for each wall depending on its neighbors
    pows_two = [2 ** x for x in range(len(shifts))]
    walls = [np.roll(maze, shift, (0, 1)) for shift in shifts]
    walls = np.array(pows_two).reshape(-1, 1, 1) * walls
    walls = sum(walls)
    # Replace the original wall value of 1 for each wall
    maze[maze == 1] = walls[maze == 1]
    maze = maze[1:-1, 1:-1]
    return maze


def _solve_maze(maze):
    """Solves the maze by replacing each value with the distance from the end"""
    maze = np.array(maze)
    shape = maze.shape
    # Floor value is an identity element when used with minimum
    FLOOR_VALUE = np.iinfo(np.int32).max
    # Wall value is an absorbing element when used with minimum
    WALL_VALUE = -FLOOR_VALUE
    # Modify maze to use the special floor and wall values
    maze = np.array([FLOOR_VALUE,WALL_VALUE])[maze]
    # Set the initial value at the start of the maze and let it spread outwards
    maze[1,1] = 1
    # Repeat until there are no empty floor values
    while FLOOR_VALUE in maze:
        # Cartesian product of [-1,0,1] repeated N times
        shifts = np.array(np.meshgrid(*([[-1, 0, 1]] * len(shape)))).T.reshape(-1, len(shape))
        # Only keep rows with exactly one nonzero element
        shifts = shifts[np.sum(shifts != 0, axis=1) == 1]
        # Expand the current path in all directions and increase the value by 1
        # Taking the abs of the maze results in neither the walls or floors having any influence on the path
        paths = np.min([np.roll(abs(maze), shift, (0, 1)) for shift in shifts], axis=0) + 1
        # Replace floor values with a path, wall values cannot be changed
        maze = np.minimum(maze, paths)
    maze[maze==WALL_VALUE] = 0
    return maze


def _run_maze_tm(trans, **kwargs):
    trans = trans + [
        [state, ((-1, -1, -1), (-1, 1, -1), (-1, -1, -1)), 'halt', ((-1, -1, -1), (-1, 1, -1), (-1, -1, -1))]
        for state in kwargs['states']
    ]
    tm = TM(trans)
    tm.write_tape(kwargs['target'])
    tape = tm(kwargs['tm_timeout'])
    return tape


def maze_fitness(pop, **kwargs):
    """Difference in values"""
    maze = np.array(kwargs['target'])
    sol = np.array(kwargs['maze_sol'])
    fits = np.empty(len(pop))
    for i, trans in enumerate(pop):
        tape = _run_maze_tm(trans, **kwargs)
        if tape.shape != sol.shape or (tape==maze).all():
            fit = 0
        else:
            # m = sol[tape!=maze]
            fit = np.max(sol[tape!=maze])
        fits[i] = fit
    return fits


#
# Mutation Functions
#

def point_mutation(trans, **kwargs):
    """Randomly change a value in a random transition"""
    # Duplicate the original transitions
    trans = [t.copy() for t in trans]
    # Select a random transition
    index = kwargs['rng'].integers(len(trans))
    t = trans[index]
    # Select a random argument within the transition
    sub_index = kwargs['rng'].integers(len(t))
    # Change the sub parameter
    if sub_index == 0 or sub_index == 2:
        trans[index][sub_index] = choice(kwargs['states'], kwargs['rng'])
    elif sub_index == 1 or sub_index == 3:
        # trans[index][sub_index] = choice(kwargs['symbols'], kwargs['rng'])
        trans[index][sub_index] = to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape']))
    else:
        trans[index][sub_index] = choice(kwargs['moves'], kwargs['rng'])
    return trans


#
# Crossover Functions
#

def one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a))
    cut_b = kwargs['rng'].integers(0, len(b))
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
    return new_a, new_b


def two_point_crossover(a, b, **kwargs):
    cut_a_0 = kwargs['rng'].integers(0, len(a))
    cut_b_0 = kwargs['rng'].integers(0, len(b))
    cut_a_1 = kwargs['rng'].integers(cut_a_0, len(a))
    cut_b_1 = kwargs['rng'].integers(cut_b_0, len(b))
    new_a = a[:cut_a_0] + b[cut_b_0:cut_b_1] + a[cut_a_1:]
    new_b = b[:cut_b_0] + a[cut_a_0:cut_a_1] + b[cut_b_1:]
    return new_a, new_b


#
# Target Functions
#


#
# Initial pops
#


#
# Debug
#

if __name__ == '__main__':

    pass

    # a = [
    #     [2,0],
    #     [0,0],
    # ]
    #
    # b = [
    #     [1,2,0],
    #     [0,0,0],
    #     [0,0,0],
    # ]
    #
    # diff = array_diff(a,b)
    # print(diff)

    maze = gen_maze((71,71))
    # plt.imshow(m)
    # plt.show()

    # m = format_maze(m)
    # plt.imshow(m)
    # plt.show()

    # m = _solve_maze(m)
    # plt.imshow(m)
    # plt.show()

    X=-1

    trans = [
        ['start', [[X,X,X],[X,0,X],[X,X,X]], 'start', [[X,X,X],[X,1,X],[X,X,X]], +1, 0],
    ]

    tape = _run_maze_tm(trans, tm_timeout=100, states=['start'], target=maze)

    plt.imshow(tape)
    plt.show()

    fits = maze_fitness([trans], tm_timeout=100, states=['start'], target=maze)
    print(fits)