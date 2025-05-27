from evolve import simulate_tests
from lgp import *
from plot import plot_results

if __name__ == '__main__':

    kwargs = {
        'name': 'debug',
        'seed': None,
        'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates, 3:

        # Size
        'num_runs': 2,
        'num_gens': 100,
        'pop_size': 100,

        # 'max_height': 10, # The maximum height
        'tape_dim': 2, # Dimensionality of the Turing tape
        'tm_timeout': 100, # Number of TM iterations before forcing a halting state

        # Initialization
        'init_individual_func': random_trans, # Function used to generate the initial population
        'init_min_len': 1,
        'init_max_len': 4,
        'states': ['start', 'halt'] + [str(i) for i in range(5)],
        'symbols': [0, 1],
        'moves': [-1,0,1],

        # Evaluation
        'fitness_func': fitness_box,
        'minimize_fitness': False,

        # Selection
        'keep_parents': 2, # Elitism, must be even
        'k': 2, # Number of randomly chosen parents for each tournament

        # Repopulation
        'p_c': 0.2, # Probability of crossover
        'crossover_func': one_point_crossover,
        # 'subgraph_max_height': 4,
        'mutate_funcs': [
            [point_mutation, 0.3],
        ],

        # Tests
        'test_kwargs': [
            ['Initial Population', 'p_c',],
            ['Low', 0.5,],
            ['High', 0.9,],
            # ['With Constants', ['x']+list(range(-5,6)),],
        ],
    }

    simulate_tests(**kwargs)
    # plot_results(**kwargs)