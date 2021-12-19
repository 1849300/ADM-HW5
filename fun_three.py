import trees as t
import math
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


def shortest_path(graph, s, t):
    ''' It computes the shortest path between nodes s and t in graph '''

    if s == t:
        return "There is no path, source and target nodes are the same", -1

    unvisited, shortest_path, predecessor = list(), dict(), dict()

    # We set the distances between the source node and all other nodes to infinity, except for the distance between source
    # and itself, which we set to 0.
    for node in graph.nodes():
        shortest_path[node] = math.inf
        unvisited.append(node)
    shortest_path[s] = 0

    # We loop until we visit all the nodes in the graph
    while unvisited:
        # We choose the node with the smallest value as the “current node”
        current_node = None
        for node in unvisited:
            if current_node == None:
                current_node = node
            elif shortest_path[node] < shortest_path[current_node]:
                current_node = node
        # Visit all the  neighbour of current_node. As we visit each neighbor, we update their tentative distance
        # from the starting node
        for neighbor in graph.neighbors(current_node):
            value = shortest_path[current_node] + \
                graph[current_node][neighbor]['weight']
            if value < shortest_path[neighbor]:
                shortest_path[neighbor] = value
                predecessor[neighbor] = current_node

        unvisited.remove(current_node)

    # Now we have to return the path using predecessor dictionary
    if t not in predecessor:
        return "Not possible, there is no path between target and source", -1
    last = t
    path = list([last])
    while last != s:
        path.append(predecessor[last])
        last = predecessor[last]

    return path


def selectRandom(nodes):
    ''' Select at random one node '''
    return random.choice(list(nodes))


def selectRandomSeq(nodes):
    ''' Select at random a certain number of nodes (in this case 5) '''
    return np.random.choice(nodes, 5)


def fun3(start, stop, p, p1, pn):
    ''' It computes the shortest ordered path '''

    graph = t.build('a', start, stop)

    # Create the path between the starting node and the first of the sequence
    path0 = list(shortest_path(graph, p1, p[0])[::-1][:-1])

    # Create a path between a node and his following in the sequence of nodes
    path = []
    for i in range(len(p)-1):
        percorso = list(shortest_path(graph, p[i], p[i+1]))
        if percorso[1] == -1:
            return "Not possible"
        percorso = percorso[::-1]
        path.extend(percorso[:-1])

    # Create the path between the last node of the sequence and the last node of the path (pn)
    pathn = list(shortest_path(graph, p[-1], pn)[::-1])

    path.extend(pathn)
    path0.extend(path)

    return path0


def vis3(start, stop, p, p1, pn):
    lista = fun3(start, stop, p, p1, pn)
    if lista == 'Not possible':
        return lista
    edges = []
    for i in range(len(lista)-1):
        edge = (lista[i], lista[i+1])
        edges.append(edge)
    print(edges)
    grafo = nx.DiGraph(edges)
    color = []
    for node in grafo:
        if node == p1 or node == pn:
            color.append('green')
        elif node in p:
            color.append('yellow')
        else:
            color.append('blue')

    plt.clf()
    nx.draw_networkx(grafo, node_color=color)
    plt.show()
