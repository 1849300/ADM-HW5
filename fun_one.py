import trees as t
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def direc(graph):
    ''' Return "yes" if all the couples of nodes are an edge for each direction, "no" oterwhise '''

    for el in graph.edges:
        if (el[1], el[0]) not in graph.edges:
            return 'yes'
    return 'no'


def fun1(graph):
    ''' This function implements the functionality 1 '''

    # Number of users
    numusers = len(graph.nodes)
    # Number of answers to question or comments
    numanswers = len(graph.edges)
    # Average number of links for each node
    average = numanswers/numusers
    # Density degree of the graph
    tot = sum([graph.get_edge_data(el[0], el[1])['weight']
              for el in graph.edges])
    density = tot / ((numusers)*(numusers)-1)
    density = round(density, 4)
    # The graph is sparse or dense?
    if density < 0.5:
        result = 'yes'
    else:
        result = 'no'

    l = [direc(graph), numusers, numanswers, average, density, result]
    table = pd.DataFrame(l, columns=['result'])
    table.index = ['directed?', 'N° of users',
                   'N° of answers', 'average', 'density', 'sparse?']
    return table


def plot_degree_dist(graph):
    ''' Plot degree density function '''

    degrees = [graph.degree(n) for n in graph.nodes()]
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees)))
    plt.show()
