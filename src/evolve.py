from multiprocessing import Pool, cpu_count

from src.utils.save import *

"""
Core functions used in controlling evolution
All functions are independent of what is being evolved
"""


#
# Initialization
#

def init_pop(pop_size, init_individual_func, **kwargs):
    """Generate a random population"""
    return [init_individual_func(**kwargs) for _ in range(pop_size)]


#
# Selection
#

def tournament_selection(pop, fits, k, **kwargs):
    """Select a single parent from a tournament of k"""
    # Select the random tournament
    tourn_indices = kwargs['rng'].choice(len(pop), size=2, replace=False)
    tourn = [pop[i] for i in tourn_indices]
    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(k)]
    # Sort all parents by fitness
    parent = sorted(parent)
    if not kwargs['minimize_fitness']:
        parent = parent[::-1]
    # Get the chromosome of the first element
    parent, fit = tourn[parent[0][1]], parent[0][0]
    return parent, fit


#
# Simulation and Iteration
#

def next_pop(pop, **kwargs):

    # Truncate Selection
    # This is not used
    if 'lambda' in kwargs:
        pass
        # Pool starts with all current parents
        # pool = list(pop) if kwargs['keep_parents'] else []
        # # Create children for all parents and add to the pool
        # for parent in pop:
        #     for i in range(kwargs['lambda']):
        #         child = kwargs['mutate_func'](parent, **kwargs)
        #         pool.append(child)
        # # Evaluation
        # pool_fits = kwargs['fitness_func'](pop=pool, **kwargs)
        # # Sort and truncate the indices of the next generation
        # pool_indices = [(pool_fits[i], i) for i in range(len(pool))]
        # pool_indices = np.array(sorted(pool_indices))
        # pool_indices = pool_indices[:kwargs['pop_size'], 1]
        # pool_indices = list(pool_indices)
        # # Reduce reps
        # new_pop = np.array(pool)[pool_indices]
        # kwargs['fits'] = np.array(pool_fits)[pool_indices]
        # if kwargs['verbose'] > 1: print(pop)
        # return new_pop, kwargs['fits']

    # Crossover
    else:
        kwargs['pop'] = pop
        # Evaluation
        kwargs['fits'] = kwargs['fitness_func'](**kwargs)
        # Elitism
        pool = [(kwargs['fits'][i], i) for i in range(kwargs['pop_size'])]
        pool = sorted(pool)
        pool = pool if kwargs['minimize_fitness'] else pool[::-1]
        new_pop = [pop[pool[i][1]] for i in range(kwargs['keep_parents'])]
        # Repeat until the new population is the same reps as the old
        while len(new_pop) < len(pop):
            # Selection
            c0, f0 = tournament_selection(**kwargs)
            c1, f1 = tournament_selection(**kwargs)

            # Crossover
            c = kwargs['rng'].random()
            if c < kwargs['p_c']:
                c0, c1 = kwargs['crossover_func'](c0, c1, **kwargs)

            # Mutation
            a, p = zip(*kwargs['mutate_funcs'])
            mutate_func = kwargs['rng'].choice(a=a, p=p)
            if mutate_func is not None:
                c0 = mutate_func(c0, **kwargs)
                c1 = mutate_func(c1, **kwargs)

            c0 = c0.copy()
            c1 = c1.copy()
            # c0.prev_fit = (f0 + f1) / 2
            # c1.prev_fit = (f0 + f1) / 2
            new_pop.append(c0)
            new_pop.append(c1)

        return new_pop, kwargs['fits']


def simulate_run(**kwargs):
    """Run a single simulation of a full set of generations"""

    # Add no-operation as a possible mutation
    prob_noop = 1 - sum(list(zip(*kwargs['mutate_funcs']))[1])
    if prob_noop > 0:
        kwargs['mutate_funcs'].append([None, prob_noop])

    shape = (kwargs['num_gens'] + 1, kwargs['pop_size'])

    # Initialization
    all_pops = np.empty(shape, dtype=object)
    all_fits = np.empty(shape)
    prev_fit = np.empty(shape)

    pop = init_pop(**kwargs)
    # all_pops[0] = [n.to_lists() for n in pop]
    all_pops[0] = pop

    # for indiv in pop: indiv.prev_fit = 0

    # Loop level 2
    for generation in range(kwargs['num_gens']):
        if kwargs['verbose'] > 0 and generation % 1 == 0:
            print(f'\tGeneration {generation} of {kwargs["num_gens"]}')

        # Next generation and previous fitness
        pop, fit = next_pop(pop=pop, **kwargs)

        # Save results
        # all_pops[generation + 1] = [n.to_lists() for n in pop]
        all_pops[generation + 1] = pop
        all_fits[generation] = fit

    # Final fitness values
    all_fits[-1] = kwargs['fitness_func'](pop, is_final=True, **kwargs)

    return all_pops, all_fits


def _simulate_and_save_test_run(test_num, run_num, test_kwargs, base_kwargs):
    """Parallel worker for (test_num, run_num)."""

    # Extract test-specific kwargs
    test_keys = test_kwargs[0]
    test_values = test_kwargs[test_num + 1]
    kwargs = base_kwargs.copy()

    # Update with test-specific values
    for key, value in zip(test_keys, test_values):
        kwargs[key] = value

    # Set path and create directory (thread-safe)
    test_name = test_values[0]
    path = f'../saves/{kwargs["name"]}/data/{test_name}'
    os.makedirs(path, exist_ok=True)

    # Assign seed and RNG
    if kwargs['seed'] is None:
        kwargs['seed'] = np.random.randint(0, 2**64, dtype='uint64')
    kwargs['rng'] = np.random.default_rng(kwargs['seed'])

    # Run simulation and save
    pops, fits = simulate_run(**kwargs)
    save_run(path, pops, fits, **kwargs)

    return f"Test {test_num}, Run {run_num}, Seed {kwargs['seed']}"


def simulate_tests(num_runs, test_kwargs, **kwargs):
    """
    Simulate all runs for all tests with different hyperparameters.
    There are four levels: [test] [run/replicant] [generation/population] [individual]
    """

    # Save kwargs first in case of failure
    # save_kwargs(num_runs=num_runs, test_kwargs=test_kwargs, **kwargs)

    # Number of tests must be inferred and is only used within this function
    num_tests = len(test_kwargs) - 1

    # Build the job list: one job per (test, run)
    jobs_args = [
        (test_num, run_num, test_kwargs, kwargs.copy())
        for test_num in range(num_tests)
        for run_num in range(num_runs)
    ]

    # Parallelize processes
    if kwargs.get('parallelize', False):

        if kwargs.get('verbose', 0) > 0:
            print(f"Dispatching {len(jobs_args)} parallel jobs...")

        #MODIFY cpu_count() to adjust max core count use
        with Pool(processes=min(cpu_count(), len(jobs_args))) as pool:
            results = pool.starmap(_simulate_and_save_test_run, jobs_args)

        if kwargs.get('verbose', 0) > 0:
            print("All test runs completed.")
            for res in results:
                print(res)

    # Non parallelized
    else:
        for job_args in jobs_args:
            _simulate_and_save_test_run(*job_args)