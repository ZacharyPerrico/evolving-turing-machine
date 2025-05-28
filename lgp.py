import numpy as np

from tm import TM

"""
Functions used in the evolution of Turing Machine based genetic programming
"""


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
    return trans


#
# Fitness Functions
#

# def array_diff(a, b):
#     """Helper function to find the differences in two arrays"""
#     a = np.array(a)
#     b = np.array(b)
#     a = np.trim_zeros(a)
#     b = np.trim_zeros(b)
#     # if a.size == 0:
#     #     a = np.array([[0]])
#     pad = np.maximum(0, np.array(b.shape)-a.shape)
#     a = np.pad(a, pad)
#     a = a[*[slice(0, i) for i in b.shape]]
#     diff = np.sum(abs(a - b))
#     return diff

def array_diff(a, b):
    """Helper function to find the differences in two arrays"""
    a = np.array(a)
    b = np.array(b)
    a = np.trim_zeros(a)
    b = np.trim_zeros(b)
    b_pad = [(0,i) for i in np.maximum(0, np.array(a.shape) - b.shape)]
    a_pad = [(0,i) for i in np.maximum(0, np.array(b.shape) - a.shape)]
    a = np.pad(a, a_pad)
    b = np.pad(b, b_pad)
    # a = a[*[slice(0, i) for i in b.shape]]
    # diff = np.sum(abs(a - b))
    # a = a!=0
    # b = b!=0
    diff = np.sum((a!=0)!=(b!=0))
    return diff

def pattern_diff(pop, **kwargs):
    fits = np.empty(len(pop))
    for i,trans in enumerate(pop):
        tape = TM(trans)(kwargs['tm_timeout'])
        fit = array_diff(tape, kwargs['target'])
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

    a = [
        [2,0],
        [0,0],
    ]

    b = [
        [1,2,0],
        [0,0,0],
        [0,0,0],
    ]

    diff = array_diff(a,b)
    print(diff)

    # for i in range(1000):
    #     a = '1'
    #     b = 'a'
    #     rng = np.random.default_rng()
    #     c = two_point_crossover(a,b,rng=rng)
    #     print(c)
