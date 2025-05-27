import numpy as np

from tm import TM

"""Functions used in the evolution of Turing Machine based genetic programming"""


#
# Utility
#

def choice(arr, rng):
    """
    Return a random element of the given array without casting.
    This exists to simplify code.
    See: https://github.com/numpy/numpy/issues/10791
    """
    return arr[rng.choice(len(arr))]


#
# Initialization Functions
#

def random_transition(**kwargs):
    """Helper function for generating only a single transition"""
    return [
        choice(kwargs['states'], kwargs['rng']),
        choice(kwargs['symbols'], kwargs['rng']),
        choice(kwargs['states'], kwargs['rng']),
        choice(kwargs['symbols'], kwargs['rng']),
        *[choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
    ]

def random_trans(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    trans = [random_transition(**kwargs) for _ in range(init_len)]
    print(trans)
    return trans


#
# Fitness Functions
#

# def fitness_box(pop, **kwargs):
#     fits = np.empty(len(pop))
#     for i,trans in enumerate(pop):
#         tm = TM(trans)
#         tm.run(kwargs['tm_timeout'])
#         arr = tm.get_tape_as_array()
#         fit = np.sum(arr)
#         fits[i] = fit
#     return fits

# def pattern_diff(a,b):
#     a = np.array(a)
#     b = np.array(b)
#     lcm = np.lcm(a.shape, b.shape)
#     a = np.tile(a, lcm // a.shape)
#     b = np.tile(b, lcm // b.shape)
#     diff = np.sum(abs(a-b)) / a.size
#     return diff

def pattern_diff(a,b):
    a = np.array(a)
    b = np.array(b)
    a = np.trim_zeros(a)
    b = np.trim_zeros(b)
    # if a.size == 0:
    #     a = np.array([[0]])
    pad = np.maximum(0, np.array(b.shape)-a.shape)
    a = np.pad(a, pad)
    a = a[*[slice(0, i) for i in b.shape]]
    diff = np.sum(abs(a - b))
    return diff

def fitness_box(pop, **kwargs):
    pattern = [
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [0,0,1,1,0,0],
        [0,1,1,1,1,0],
        [0,1,1,1,1,0],
        [0,1,0,0,1,0],
    ]

    fits = np.empty(len(pop))
    for i,trans in enumerate(pop):
        tm = TM(trans)
        tm.run(kwargs['tm_timeout'])
        arr = tm.get_tape_as_array()
        fit = pattern_diff(arr, pattern)
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
        trans[index][sub_index] = choice(kwargs['symbols'], kwargs['rng'])
    else:
        trans[index][sub_index] = choice(kwargs['moves'], kwargs['rng'])
    return trans


#
# Crossover Functions
#

def one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a))
    cut_b = kwargs['rng'].integers(0, len(b))
    new_a = [*a[cut_a:], *b[:cut_b]]
    new_b = [*b[cut_b:], *a[:cut_a]]
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

    # def pattern_diff(a,b):
    #     a = np.array(a)
    #     b = np.array(b)
    #     a = np.trim_zeros(a)
    #     b = np.trim_zeros(b)
    #
    #     # a_shape = np.array(a.shape)
    #     # b_shape = np.array(b.shape)
    #
    #     a = a[*[slice(0, i) for i in b.shape]]
    #
    #     # a_pad = a_shape < b_shape
    #
    #     diff = a - b
    #
    #     # lcm = np.lcm(a.shape, b.shape)
    #     # a = np.tile(a, lcm // a.shape)
    #     # b = np.tile(b, lcm // b.shape)
    #     # diff = np.sum(abs(a-b)) / a.size
    #     return diff

    a = [
        [1,2],
        [3,4],
    ]

    b = [
        [1,2,1],
        [3,4,3],
        [1,2,1],
    ]

    # rng = np.random.default_rng()

    diff = pattern_diff(a,b)

    print(diff)