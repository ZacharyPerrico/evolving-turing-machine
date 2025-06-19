from src.utils.utils import choice

"""
Functions used in the evolution of Turing Machine based genetic programming
"""


#
# Initialization Functions
#

def _random_transition(**kwargs):
    """Helper function for generating only a single transition"""
    return [
        choice(kwargs['states'], kwargs['rng']),
        choice(kwargs['symbols'], kwargs['rng']),
        choice(kwargs['states'], kwargs['rng']),
        choice(kwargs['symbols'], kwargs['rng']),
        [choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
    ]

# def _random_transition(**kwargs):
#     """Helper function for generating only a single transition"""
#     return [
#         choice(kwargs['states'], kwargs['rng']),
#         to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape'])),
#         choice(kwargs['states'], kwargs['rng']),
#         to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape'])),
#         *[choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
#     ]


def random_trans(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    trans = [_random_transition(**kwargs) for _ in range(init_len)]
    return trans


#
# Fitness Functions
#


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
        # trans[index][sub_index] = to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape']))
    else:
        trans[index][sub_index] = [choice(kwargs['moves'], kwargs['rng']) for i in range(len(t[4]))]
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