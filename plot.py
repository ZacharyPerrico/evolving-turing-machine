import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from lgp import _run_maze_tm, gen_maze, _solve_maze, maze_fitness, _format_maze
from tm import TM
from save_utils import load_kwargs, load_runs

"""All functions relevant to plotting"""


def plot_nodes(nodes, result_fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    """Plot all given nodes and the fitness function"""

    # Only plot the first domain
    xs = np.linspace(*kwargs['domains'][0])

    # Plot target function if given
    if 'target_func' in kwargs and result_fitness_func is not None:
        label = 'Target Function'
        target_ys = [kwargs['target_func'](x) for x in xs]
        plt.scatter(xs, target_ys, label=label)
        plt.plot(xs, target_ys, lw=5)

    # Plot nodes
    for i, node in enumerate(nodes):
        # Determine label based on what info is known
        if labels is not None:
            label = labels[i]
        elif 'test_kwargs' in kwargs:
            label = kwargs['test_kwargs'][i + 1][0]
        else:
            label = ''
        if 'target_func' in kwargs and result_fitness_func is not None:
            label += f' Fitness = {result_fitness_func([node], **kwargs)[0]:f}'

        # Evaluate and plot real part and imaginary part if applicable
        node_ys = [node(i, eval_method=kwargs['eval_method']) for i in xs]
        plt.scatter(xs, np.real(node_ys), label=label)
        plt.plot(xs, np.real(node_ys))
        if np.iscomplex(node_ys).any():
            label = label.split('Fitness')[0] + 'Imaginary Part'
            plt.scatter(xs, np.imag(node_ys), label=label)
            plt.plot(xs, np.imag(node_ys), ':')

    plt.title(title)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    plt.show()


def plot_tape(trans, fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    """Plot the resulting Turing tape"""
    # tape = TM(trans)(kwargs['tm_timeout'])

    tape = _run_maze_tm(trans, **kwargs)
    fit = maze_fitness([trans], **kwargs)[0]

    plt.title(title)
    plt.scatter(1,1, label=fit)
    plt.imshow(tape)
    if 'test_kwargs' in kwargs:
        plt.legend(title=kwargs['test_kwargs'][0][0])
    if 'name' in kwargs:
        plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    plt.show()




def plot_min_fit(all_pops, all_fits, title=None, legend_title=None, **kwargs):
    """Plot the average of the runs' minimum fitness for each test"""
    fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))
    for test in range(all_fits.shape[0]):
        if kwargs['minimize_fitness']:
            y = np.mean(np.min(all_fits[test], axis=2), axis=0)
            plt.ylabel('Average Min Fitness Value')
        else:
            y = np.mean(np.max(all_fits[test], axis=2), axis=0)
            plt.ylabel('Average Max Fitness Value')
        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])
        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)
    plt.title(title)
    # ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/Fits.png')
    plt.show()


def plot_means(values, ylabel):
    """Plot the means of some values"""
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(values[test], axis=(0,2))
        xs = np.array(range(values.shape[2]))
        plt.plot(xs, ys, label=label)
        # ys_std = ys.std()
        # ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


def plot_medians(values, ylabel):
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        xs = np.array(range(values.shape[2]))
        ys = np.median(values[test], axis=(0,2))
        plt.plot(xs, ys, label=label)

        ys = np.mean(values[test], axis=(0,2))
        plt.plot(xs, ys, label=label)

        q1 = np.quantile(values[test], 0.25, axis=(0,2))
        q3 = np.quantile(values[test], 0.75, axis=(0,2))
        ax.fill_between(xs, q1, q3, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


def plot_hist(values, ylabel):
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        xs = values[test, :, -1].ravel()
        # ax.boxplot(xs,
        #     positions=[test],
        #     label=label,
        #     patch_artist=True,
        #     # showmeans=False,
        #     # showfliers=False,
        #     # medianprops={"color": "white", "linewidth": 0.5},
        #     # boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
        #     # whiskerprops={"color": "C0", "linewidth": 1.5},
        #     # capprops={"color": "C0", "linewidth": 1.5}
        # )

        label = kwargs['test_kwargs'][test + 1][0]
        xs = values[test, :, -1].ravel()
        ax.boxplot(xs,
                   positions=[test],
                   label=label,
                   patch_artist=True,
                   # showmeans=False,
                   # showfliers=False,
                   # medianprops={"color": "white", "linewidth": 0.5},
                   # boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                   # whiskerprops={"color": "C0", "linewidth": 1.5},
                   # capprops={"color": "C0", "linewidth": 1.5}
                   )
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


#
# Tables
#

def table_best(all_pops, all_fits, **kwargs):
    """Plot the best result of the given run and gen"""
    xs = [np.linspace(*domain) for domain in kwargs['domains']]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
    y_true = np.array([[kwargs['target_func'](*list(x))] for x in xs])
    table = np.concat((xs, y_true), axis=1)
    # Iterate over all runs
    for run in range(len(kwargs['test_kwargs']) - 1):
        i = all_fits[run, :, :, :].argmin()
        node = all_pops[run, :, :, :].flatten()[i]
        y_node = [[node(*x)] for x in xs]
        tab = np.concat((table, y_node), axis=1)
        print('\n', node, sep='')
        for row in tab:
            print(('f(' + ', '.join(['{}'] * len(kwargs['domains'])) + ') = {} | {}').format(*row))


#
# Graphs
#

def plot_graph(node, layout='topo', scale=1, title=None, **kwargs):
    """Plot the node as a graph"""

    # Remove duplicates
    node = list(set(tuple(t) for t in node))

    verts = []
    edges = []
    edge_labels = {}

    for t in node:
        print(t)
        state0, symbol0, state1, symbol1, *move = t
        if state0 not in verts: verts.append(state0)
        if state1 not in verts: verts.append(state1)
        index0 = verts.index(state0)
        index1 = verts.index(state1)
        edge = (index0,index1)
        if edge not in edges:
            edges.append(edge)
        edge_label = f'{symbol0} {symbol1} ' + ''.join(map(str,move))
        if edge in edge_labels:
            edge_labels[edge] += '\n' + edge_label
        else:
            edge_labels[edge] = edge_label

    # Create networkxs graph
    fig, ax = plt.subplots()
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G)
    connectionstyle = [f"arc3,rad={r}" for r in [.5, 1]]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=range(len(verts)),
        node_color='white',
        edgecolors='black',
        node_size=600 * scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={key: vert for key, vert in enumerate(verts)},
        font_color='black',
        font_size=10 * scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        edgelist=edges, # Specify edge order
        connectionstyle=connectionstyle,
        arrowsize=20 * scale,
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        # alpha=0.5,
        node_size=600 * scale,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        connectionstyle=connectionstyle,
        # edge_labels = {edges[key]: label for key,label in enumerate(edge_labels)},
        edge_labels=edge_labels,
        alpha=0.5,
        # label_pos=0.0,
        # node_size=24000 * scale,
        bbox=None,
    )
    # plt.suptitle(f'${node.latex()}$')
    plt.title(title)
    # if 'result_fitness_func' in kwargs:
    #     plt.legend(title=f'Fitness = {kwargs['result_fitness_func']([node], **kwargs)[0]}')
    # if 'name' in kwargs:
    #     plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    plt.show()


#
# Control
#

def get_best(all_pops, all_fits, gen=-1, **kwargs):
    """Get the best result of the given run and gen"""
    nodes = []
    # Iterate over all runs
    for run in range(all_pops.shape[0]):
        if kwargs['minimize_fitness']:
            i = all_fits[run, slice(None), gen, :].argmin()
        else:
            i = all_fits[run, slice(None), gen, :].argmax()
        node = all_pops[run, slice(None), gen, :].flatten()[i]
        # nodes.append(Node.from_lists(*node))
        nodes.append(node)
    return nodes


def plot_results(all_pops, all_fits, **kwargs):
    """Plot all standard plots"""
    path = f'saves/{kwargs["name"]}/plots/'
    os.makedirs(path, exist_ok=True)
    print('Plotting results')

    plot_min_fit(all_pops, all_fits, title='', **kwargs)

    # Plot best
    best = get_best(all_pops, all_fits, **kwargs)
    # if len(kwargs['domains']) == 1:
    #     plot_nodes(best, **kwargs)
    # else:
    #     table_best(all_pops, all_fits, title='Best Overall', **kwargs)

    # plot_means(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_medians(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_hist(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')

    # plot_size(all_pops, all_fits, **kwargs)
    # plot_quality_gain(all_pops, all_fits, **kwargs)
    # plot_success_rate(all_pops, all_fits, **kwargs)
    # plot_effective(all_pops, all_fits, **kwargs)
    # plot_noop_size(all_pops, all_fits, **kwargs)

    # # def plot_tape(trans, fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    # # """Plot the resulting Turing tape"""
    # tape = kwargs['target']
    # # plt.title(title)
    # plt.imshow(tape)
    # plt.legend(title=kwargs['test_kwargs'][0][0])
    # # plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    # plt.show()


    for i, tm in enumerate(best):
        title = 'Best TM (' + kwargs['test_kwargs'][i + 1][0] + ')'
        plot_graph(tm, title=title+'_0', **kwargs)
        plot_tape(tm, title=title, **kwargs)


if __name__ == '__main__':
    # kwargs = load_kwargs('maze_0')
    # kwargs['target'] = np.array(kwargs['target'])
    # kwargs['maze_sol'] = np.array(kwargs['maze_sol'])
    # # from main import kwargs
    # pops, fits = load_runs(**kwargs)
    # plot_results(pops, fits, **kwargs)



    maze = gen_maze((9,9))
    maze = _format_maze(maze)
    maze_sol = _solve_maze((maze!=0)*1, (3,3))
    trans = [
        ['start', 0, 'start', 1, +1, 0]
    ]
    tape = TM(trans)(100)
    plt.imshow(tape)
    plt.show()
    # tape = _run_maze_tm(trans, tm_timeout=100, states=['start'], target=maze)
    # fits = maze_fitness(pop, tm_timeout=100, states=['start'], target=maze, maze_sol=maze_sol)
    # t1 = time.time()
    # total = t1-t0
    # plot_tape(trans, tm_timeout=100, target=maze, maze_sol=maze_sol)
    # print(fits)








    # maze = gen_maze((71,71))

    # maze = [
    #     [
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1
    #     ],
    #     [
    #         1,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         1,
    #         0,
    #         1
    #     ],
    #     [
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         0,
    #         1,
    #         0,
    #         1
    #     ],
    #     [
    #         1,
    #         0,
    #         0,
    #         0,
    #         1,
    #         0,
    #         0,
    #         0,
    #         1
    #     ],
    #     [
    #         1,
    #         0,
    #         1,
    #         0,
    #         1,
    #         1,
    #         1,
    #         0,
    #         1
    #     ],
    #     [
    #         1,
    #         0,
    #         1,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         1
    #     ],
    #     [
    #         1,
    #         0,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1
    #     ],
    #     [
    #         1,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         1
    #     ],
    #     [
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1
    #     ]
    # ]
    #
    # maze = np.array(maze)
    # maze_sol = _solve_maze(maze)
    #
    # trans = [
    #     ('start', ((0, -1, 1), (-1, -1, -1), (0, 1, 1)), '0', ((1, 1, 1), (0, 0, 0), (1, 1, 1)), 1, -1),
    #     ('start', ((-1, 1, 0), (1, -1, 0), (1, 0, -1)), '0', ((1, 1, 1), (0, 0, 0), (0, 1, 0)), 0, -1),
    #     ('start', ((-1, 1, -1), (0, 1, 0), (1, 1, 1)), '0', ((1, 1, 1), (0, 1, 0), (0, 1, 0)), 0, 1),
    #     ('start', ((1, 0, 0), (1, 1, 1), (1, 0, 1)), 'start', ((1, 0, 0), (1, 0, 0), (1, 1, 0)), 1, 0),
    #     ('start', ((1, 0, 1), (0, -1, 0), (1, -1, 0)), '0', ((1, 0, 1), (1, 1, 0), (0, 1, 0)), 0, 1),
    #     ('0', ((0, 1, -1), (1, -1, 1), (1, 0, 0)), '0', ((0, 0, 0), (0, 1, 1), (1, 0, 0)), 0, 0),
    #     ('0', ((0, 1, 1), (-1, -1, -1), (1, -1, 0)), 'start', ((0, 1, 0), (0, 0, 1), (0, 0, 1)), 1, 0),
    #     ('start', ((0, 0, 1), (0, 1, 0), (1, 1, 1)), '0', ((0, 0, 0), (1, 1, 0), (0, 0, 0)), 1, 1),
    #     ('0', ((-1, -1, 0), (1, 0, 1), (0, -1, 0)), '0', ((1, 0, 1), (0, 1, 0), (0, 0, 1)), 1, 0),
    #     ('start', ((-1, 0, 1), (1, 1, 1), (0, 1, -1)), 'start', ((1, 1, 0), (1, 0, 0), (0, 0, 1)), 1, -1),
    #     ('start', ((-1, -1, 1), (1, 1, 0), (1, -1, 0)), 'start', ((0, 0, 1), (0, 1, 0), (0, 0, 0)), -1, 0),
    #     ('0', ((1, 0, 0), (0, -1, 0), (1, 1, -1)), 'start', ((0, 0, 0), (0, 0, 0), (1, 1, 0)), 0, -1),
    #     ('start', ((1, 1, 1), (1, 1, 0), (0, 1, 1)), 'start', ((1, 0, 1), (0, 1, 0), (1, 1, 0)), -1, -1),
    #     ('0', ((1, -1, 0), (0, 0, 0), (1, 1, -1)), 'start', ((0, 0, 1), (1, 0, 1), (1, 1, 0)), -1, -1),
    #     ('start', ((-1, 0, 0), (-1, 1, -1), (0, 1, -1)), 'start', ((1, 0, 1), (1, 1, 1), (0, 0, 1)), 0, -1),
    #     ('start', ((0, 1, 0), (1, 0, 0), (0, 0, 0)), '0', ((1, 1, 0), (0, 1, 0), (0, 1, 0)), -1, 1),
    #     ('0', ((1, 1, 1), (1, -1, 1), (-1, -1, -1)), 'start', ((0, 0, 1), (0, 1, 0), (0, 0, 0)), -1, -1),
    #     ('start', ((1, 0, 1), (0, 1, 1), (-1, 0, 0)), '0', ((1, 0, 1), (1, 0, 1), (0, 0, 1)), 0, 0),
    #     ('start', ((-1, -1, 0), (1, 0, -1), (-1, 1, -1)), '0', ((1, 0, 0), (0, 1, 1), (0, 1, 0)), 0, 0),
    #     ('start', ((1, -1, -1), (0, 1, 1), (-1, -1, -1)), 'start', ((1, 0, 1), (1, 1, 0), (1, 0, 1)), 1, 1),
    #     ('start', ((-1, -1, -1), (-1, -1, 1), (1, 0, 1)), 'start', ((0, 1, 1), (1, 0, 1), (0, 0, 1)), 0, 0),
    #     ('0', ((-1, 1, -1), (1, 1, 0), (-1, 1, 1)), 'start', ((1, 1, 1), (0, 0, 1), (0, 0, 0)), 1, 0),
    #     ('start', ((1, -1, 1), (-1, -1, 1), (1, 1, -1)), '0', ((1, 1, 1), (1, 1, 0), (0, 1, 1)), 1, -1),
    #     ('0', ((1, -1, 1), (1, -1, 0), (-1, 0, -1)), '0', ((1, 1, 0), (0, 1, 0), (1, 0, 0)), -1, 1),
    #     ('start', ((-1, -1, 0), (1, 1, 1), (-1, 0, 1)), '0', ((0, 0, 1), (1, 1, 1), (0, 1, 1)), 0, 1),
    #     ('start', ((1, 0, 0), (-1, -1, -1), (0, 0, -1)), '0', ((0, 1, 1), (1, 1, 0), (0, 1, 0)), 1, 0),
    #     ('start', ((-1, 0, 0), (-1, 0, 0), (-1, -1, 0)), '0', ((0, 1, 1), (1, 0, 1), (0, 1, 1)), 0, 1),
    #     ('0', ((-1, 1, 1), (0, 0, 1), (1, -1, -1)), '0', ((1, 1, 1), (1, 1, 0), (0, 0, 0)), 1, -1),
    #     ('0', ((1, 0, 0), (1, 0, 1), (0, 1, 0)), '0', ((0, 1, 0), (0, 1, 0), (1, 0, 0)), 0, 1),
    #     ('0', ((-1, 0, 0), (-1, 0, 1), (1, 1, 0)), 'start', ((1, 1, 0), (0, 0, 0), (0, 0, 0)), -1, -1),
    #     ('start', ((1, 1, 0), (0, 0, 1), (1, 0, 0)), 'start', ((0, 1, 0), (1, 1, 1), (1, 1, 1)), 1, 1),
    #     ('start', ((-1, 0, -1), (-1, 0, -1), (-1, -1, -1)), 'start', ((1, 1, 0), (1, 1, 0), (1, 1, 0)), 1, 0),
    #     ('start', ((0, -1, 1), (0, 1, -1), (0, -1, 0)), '0', ((1, 0, 1), (0, 0, 0), (1, 1, 0)), 0, -1),
    #     ('0', ((0, -1, 0), (1, -1, 1), (1, 0, 1)), 'start', ((1, 0, 0), (1, 0, 1), (1, 0, 1)), 1, 0),
    #     ('start', ((0, 0, 1), (1, 1, 1), (-1, 1, 1)), 'start', ((1, 1, 1), (0, 0, 1), (1, 1, 0)), -1, 1),
    #     ('0', ((-1, 0, -1), (0, 1, 0), (1, 1, 0)), 'start', ((1, 0, 0), (0, 1, 1), (0, 0, 0)), 0, 0),
    #     ('0', ((0, 0, 1), (-1, 1, 1), (-1, 0, 1)), 'start', ((0, 1, 1), (0, 0, 0), (1, 0, 1)), -1, 1),
    #     ('0', ((0, 1, 0), (1, 1, 1), (1, 0, 1)), 'start', ((0, 1, 1), (0, 0, 1), (1, 0, 1)), 0, -1),
    #     ('0', ((-1, -1, 0), (-1, -1, 1), (-1, -1, 0)), '0', ((1, 0, 1), (1, 0, 0), (1, 1, 0)), -1, 1),
    #     ('0', ((1, -1, -1), (-1, -1, -1), (1, 1, 1)), '0', ((1, 1, 1), (0, 0, 1), (1, 0, 0)), 1, 1),
    #     ('start', ((1, 0, 1), (0, 1, 1), (0, 0, 1)), 'start', ((1, 0, 0), (0, 0, 0), (0, 0, 0)), 1, 1),
    #     ('start', ((1, 0, -1), (-1, 0, 0), (-1, 1, 1)), '0', ((0, 1, 0), (1, 0, 1), (0, 0, 1)), 0, 0),
    #     ('0', ((0, 1, 0), (-1, -1, -1), (1, -1, -1)), '0', ((1, 0, 0), (1, 1, 1), (1, 0, 1)), 0, 1),
    #     ('start', ((1, 1, 0), (0, 1, 1), (0, 1, 0)), 'start', ((0, 1, 1), (1, 1, 0), (1, 1, 0)), 0, 0),
    #     ('0', ((0, -1, 1), (0, 0, 1), (0, 0, 0)), '0', ((1, 1, 1), (0, 0, 0), (0, 0, 0)), 0, 1),
    #     ('start', ((0, 0, 0), (0, -1, 0), (-1, 1, 0)), '0', ((1, 0, 0), (1, 0, 1), (0, 1, 1)), -1, -1),
    #     ('start', ((-1, 0, 0), (1, -1, 0), (-1, 0, -1)), 'start', ((1, 0, 1), (1, 1, 0), (0, 1, 0)), -1, 1),
    #     ('start', ((1, 1, 1), (1, 0, 0), (1, 0, 1)), 'start', ((0, 1, 1), (0, 0, 1), (1, 0, 0)), 0, 1),
    #     ('start', ((1, 0, 0), (-1, 1, 1), (0, 1, -1)), '0', ((1, 0, 1), (0, 0, 1), (1, 1, 1)), 0, 0),
    #     ('0', ((0, 0, -1), (1, 0, 1), (-1, 0, -1)), 'start', ((1, 1, 1), (1, 0, 1), (1, 0, 1)), 0, 1),
    #     ('0', ((1, 0, 1), (0, 0, 1), (0, 0, 0)), '0', ((1, 0, 1), (0, 1, 1), (1, 0, 1)), 0, 0),
    #     ('0', ((0, 0, 0), (0, 0, 1), (-1, 0, 1)), '0', ((0, 1, 0), (1, 1, 1), (0, 0, 0)), 1, 0),
    #     ('0', ((0, 1, 1), (-1, 0, 1), (0, 1, 0)), '0', ((1, 1, 0), (1, 0, 0), (0, 0, 0)), 1, 1),
    #     ('start', ((-1, 0, 0), (-1, 0, -1), (0, 1, -1)), 'start', ((1, 0, 0), (0, 1, 1), (1, 1, 1)), 1, 1),
    #     ('0', ((0, -1, 1), (0, -1, 0), (1, 1, -1)), 'start', ((1, 1, 1), (0, 0, 0), (0, 1, 0)), -1, 0),
    #     ('start', ((0, 0, 0), (1, 1, -1), (0, 1, 1)), 'start', ((1, 0, 0), (1, 1, 0), (1, 1, 1)), 1, -1),
    #     ('0', ((-1, 0, -1), (0, 1, -1), (0, 1, 0)), '0', ((1, 1, 0), (1, 1, 1), (0, 1, 1)), -1, 0),
    #     ('0', ((1, -1, 1), (1, 1, -1), (0, 1, -1)), '0', ((0, 1, 0), (0, 1, 1), (1, 1, 0)), 1, 0),
    #     ('0', ((0, 0, 0), (0, -1, -1), (1, 0, 1)), 'start', ((1, 0, 0), (0, 0, 0), (1, 0, 0)), 0, -1),
    #     ('start', ((-1, -1, 1), (-1, 0, 1), (0, 1, -1)), '0', ((1, 0, 0), (0, 1, 1), (0, 0, 0)), 0, 0),
    #     ('0', ((1, 1, -1), (0, 0, -1), (1, 0, 1)), 'start', ((1, 0, 1), (0, 1, 0), (0, 0, 1)), -1, 1),
    #     ('0', ((1, 1, 1), (1, 0, 1), (0, 1, 0)), '0', ((0, 1, 1), (1, 0, 1), (0, 0, 0)), -1, 1),
    #     ('0', ((1, 0, 1), (1, 0, 1), (1, 0, 0)), '0', ((0, 0, 0), (0, 1, 0), (1, 0, 0)), 1, 0),
    #     ('start', ((-1, 1, 0), (-1, 1, 1), (0, 0, 0)), 'start', ((1, 1, 0), (1, 0, 0), (1, 1, 1)), 1, 0),
    #     ('start', ((-1, 1, 1), (-1, 0, 1), (0, 1, 0)), 'start', ((1, 1, 0), (0, 1, 0), (0, 1, 0)), 1, -1),
    #     ('start', ((0, 1, 0), (1, 1, 1), (1, 1, -1)), 'start', ((1, 0, 1), (0, 1, 0), (0, 0, 1)), 1, -1),
    #     ('0', ((0, -1, 0), (-1, 0, 0), (1, 0, 1)), 'start', ((1, 0, 0), (0, 0, 0), (0, 0, 1)), -1, 0),
    #     ('start', ((0, 1, 1), (0, 1, -1), (-1, -1, 1)), '0', ((1, 0, 1), (1, 1, 0), (0, 1, 0)), -1, 0),
    #     ('start', ((1, -1, 1), (1, 0, 1), (0, 0, 0)), '0', ((0, 0, 1), (1, 0, 1), (1, 1, 1)), -1, 1),
    #     ('0', ((0, -1, -1), (-1, 0, -1), (0, 0, 1)), 'start', ((1, 1, 0), (0, 1, 1), (0, 0, 1)), 0, -1),
    #     ('0', ((1, 0, -1), (-1, -1, -1), (1, 0, 1)), 'start', ((1, 0, 0), (1, 0, 1), (0, 1, 0)), 0, 1),
    #     ('0', ((1, -1, 1), (0, 1, 1), (0, 0, 1)), '0', ((0, 0, 0), (1, 0, 0), (1, 0, 1)), 1, 1),
    #     ('0', ((-1, 1, -1), (-1, -1, 0), (-1, -1, -1)), 'start', ((0, 1, 0), (1, 1, 1), (0, 0, 0)), 1, 0),
    #     ('start', ((0, -1, 1), (1, 0, 0), (1, -1, -1)), '0', ((0, 1, 0), (1, 1, 1), (1, 1, 0)), -1, 1),
    #     ('0', ((-1, -1, 0), (0, 0, 1), (1, 1, 0)), 'start', ((0, 1, 0), (0, 1, 0), (1, 0, 0)), 0, -1),
    #     ('start', ((-1, 1, -1), (-1, 1, -1), (1, 1, 1)), '0', ((1, 0, 0), (1, 0, 0), (0, 0, 1)), 1, 0),
    #     ('start', ((-1, -1, -1), (-1, 0, -1), (-1, 0, -1)), 'start', ((0, 0, 1), (1, 1, 1), (1, 1, 0)), 1, 1),
    #     ('0', ((0, 1, 1), (0, 0, 1), (0, 0, 0)), 'start', ((0, 0, 1), (1, 0, 0), (1, 1, 0)), 1, 0),
    #     ('0', ((1, 1, 1), (0, 0, 1), (1, 0, 0)), '0', ((1, 1, 0), (0, 0, 0), (0, 0, 1)), 0, 0),
    #     ('0', ((1, 0, 1), (0, -1, 0), (-1, 0, 0)), 'start', ((0, 1, 0), (1, 1, 0), (1, 1, 1)), -1, -1),
    #     ('0', ((1, 0, 0), (1, 1, 0), (1, 1, 1)), '0', ((0, 0, 1), (1, 1, 1), (1, 0, 0)), 0, -1),
    #     ('start', ((1, 1, 1), (0, 1, 1), (0, 0, 0)), '0', ((0, 0, 1), (0, 0, 1), (1, 0, 0)), 0, 0),
    #     ('start', ((0, 1, -1), (0, -1, 0), (-1, 1, -1)), 'start', ((1, 0, 0), (0, 1, 1), (1, 0, 0)), -1, 0),
    #     ('0', ((0, 0, -1), (-1, 1, 0), (0, -1, 0)), 'start', ((0, 0, 0), (0, 1, 0), (0, 1, 1)), -1, -1),
    #     ('start', ((1, 1, 0), (0, 0, 1), (0, -1, 0)), 'start', ((0, 0, 0), (1, 0, 0), (0, 0, 0)), 1, 0),
    #     ('0', ((1, 1, 0), (1, 0, 1), (-1, 0, 0)), '0', ((1, 0, 0), (1, 1, 1), (1, 0, 0)), 0, 0),
    #     ('start', ((1, 1, 1), (0, 1, 0), (0, 1, 0)), '0', ((1, 0, 0), (1, 1, 1), (0, 0, 1)), 0, -1),
    #     ('0', ((0, 1, -1), (1, 0, -1), (0, -1, 1)), 'start', ((0, 1, 0), (1, 0, 0), (0, 0, 0)), -1, -1),
    #     ('start', ((1, 1, -1), (1, 0, 0), (1, 0, 1)), 'start', ((1, 0, 0), (0, 1, 1), (0, 1, 1)), 0, 0),
    #     ('0', ((0, 1, 1), (0, 1, 0), (1, -1, 1)), 'start', ((1, 1, 1), (0, 1, 1), (1, 1, 1)), 1, 1),
    #     ('0', ((-1, 1, 1), (-1, -1, 1), (-1, 1, -1)), 'start', ((0, 1, 1), (0, 0, 0), (0, 0, 0)), 1, -1),
    #     ('0', ((0, 1, 1), (1, 0, 0), (0, 1, 0)), '0', ((1, 0, 0), (1, 0, 0), (1, 0, 0)), 0, 1),
    #     ('start', ((1, 0, 1), (0, -1, -1), (1, -1, 0)), 'start', ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 1, 0),
    #     ('start', ((-1, 0, 0), (-1, 0, 0), (0, 1, 1)), '0', ((0, 1, 0), (0, 0, 0), (1, 1, 1)), 0, -1),
    #     ('0', ((1, -1, 0), (0, 0, -1), (0, 1, 1)), '0', ((1, 0, 0), (0, 0, 1), (1, 1, 0)), -1, 0),
    #     ('start', ((1, -1, 1), (-1, 0, 1), (1, -1, 0)), '0', ((1, 0, 0), (0, 1, 0), (0, 0, 0)), -1, 0),
    #     ('start', ((1, 0, -1), (-1, -1, 0), (1, 1, 1)), '0', ((1, 1, 1), (0, 1, 1), (0, 0, 0)), 0, -1),
    #     ('start', ((-1, 1, 1), (0, -1, 0), (0, 1, -1)), '0', ((1, 0, 0), (1, 1, 0), (0, 0, 1)), 0, 0),
    #     ('start', ((-1, 1, 0), (-1, 1, 1), (0, -1, 1)), 'start', ((1, 0, 1), (0, 0, 0), (0, 1, 1)), 0, -1),
    #     ('start', ((0, 1, 1), (1, 1, -1), (1, -1, -1)), '0', ((0, 0, 0), (0, 0, 0), (0, 0, 0)), -1, 0),
    #     ('0', ((1, -1, 1), (1, -1, 0), (1, -1, -1)), 'start', ((0, 0, 1), (0, 0, 1), (1, 1, 1)), 0, 1),
    #     ('start', ((1, 1, 1), (1, 1, 1), (1, 0, 0)), 'start', ((0, 0, 1), (0, 1, 1), (1, 0, 0)), 1, 0),
    #     ('start', ((0, 0, 1), (1, -1, 0), (0, 1, 0)), '0', ((0, 1, 0), (1, 0, 1), (1, 1, 1)), 0, -1),
    #     ('start', ((1, 1, 0), (-1, 0, 1), (1, 1, -1)), '0', ((0, 0, 1), (0, 1, 1), (1, 1, 0)), 1, -1),
    #     ('0', ((1, 0, 1), (1, -1, 0), (0, -1, -1)), 'start', ((0, 1, 1), (0, 0, 0), (0, 1, 0)), -1, 1),
    #     ('start', ((0, 1, 0), (0, 1, 0), (0, 1, 1)), '0', ((0, 0, 0), (1, 1, 0), (0, 0, 0)), -1, 1),
    #     ('0', ((1, -1, 0), (-1, 1, 0), (0, 0, 1)), 'start', ((0, 0, 0), (1, 0, 0), (1, 0, 0)), 0, -1),
    #     ('0', ((-1, -1, -1), (0, 1, 1), (1, -1, 1)), 'start', ((0, 0, 1), (1, 1, 1), (1, 0, 1)), -1, -1),
    #     ('start', ((1, -1, 1), (-1, 0, -1), (-1, 1, 1)), 'start', ((1, 0, 1), (0, 1, 1), (0, 1, 0)), 0, -1),
    #     ('0', ((0, 1, 0), (-1, -1, 1), (1, 1, 1)), '0', ((0, 1, 0), (0, 1, 0), (1, 1, 1)), 1, 1),
    #     ('start', ((-1, -1, 1), (0, 0, -1), (0, 0, 0)), 'start', ((0, 1, 0), (0, 0, 0), (0, 0, 1)), -1, 1),
    #     ('0', ((-1, -1, 1), (0, 0, -1), (-1, 1, 1)), '0', ((0, 0, 0), (1, 0, 0), (0, 0, 0)), 0, 1),
    #     ('0', ((0, 1, 0), (-1, 0, 0), (0, 1, 0)), '0', ((1, 0, 0), (0, 1, 1), (0, 1, 0)), -1, -1),
    # ]
    #
    # # tape = _run_maze_tm(trans, tm_timeout=100, states=['start'], target=maze)
    #
    # plot_tape(trans, tm_timeout=10, states=['start','0','1'], target=maze, maze_sol=maze_sol, test_kwargs=[['Sample']])