import trees as t
import math
from tqdm import tqdm
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
            if shortest_path[node] <= shortest_path[current_node]:
                current_node = node

        if current_node == t:
            break
        # Visit all the  neighbour of current_node. As we visit each neighbor, we update their tentative distance
        # from the starting node
        for neighbor in graph.neighbors(current_node):
            value = shortest_path[current_node] + \
                graph[current_node][neighbor]['weight']
            if value < shortest_path[neighbor]:
                shortest_path[neighbor] = value
                predecessor[neighbor] = current_node
        unvisited.remove(current_node)

    if t not in predecessor:
        return "Not possible, there is no path between target and source", -1

    # Now we have to return the path using predecessor dictionary
    last = t
    path = list([last])
    while last != s:
        path.append(predecessor[last])
        last = predecessor[last]

    return path, shortest_path[t]


def betweeness(v, graph):
    ''' It computes betweeness '''
    num, den = 0, 0

    # For each node we compute the shortest path with all the other nodes in the graph
    for source in tqdm(graph.nodes()):
        for target in graph.nodes():
            if source != target and source != v and target != v:
                path, dist = shortest_path(graph, source, target)
                # If exist a shortest path between the source and the target, we update the denominator
                if dist != -1:
                    den += 1
                    # If the node in input is in the shortest path then we update the numerator
                    if v in path:
                        num += 1
    # We change the value of the denominator if it's equal to 0, to avoid dividing by zero
    if den == 0:
        den = 1
    betweenness_centrality = num/den

    return betweenness_centrality


def closeness(v, graph):
    ''' It computes closeness '''

    N = len(graph.nodes())
    count = 0
    # For each node in the graph we calculate the distance between this node and the node in input
    for node in tqdm(graph.nodes()):
        path, dist = shortest_path(graph, v, node)
        # If exist a path between the node in input v and "node" then we update the count of the distances
        if dist != -1:
            count += dist
    if count == 0:
        return "There is no path between the node in input and all other nodes in the graph!"

    closeness_centrality = (N - 1) / count

    return closeness_centrality


def degree_centrality(v, graph):
    ''' It computes degree centrality '''

    N = len(graph.nodes())
    out_degree = 0
    in_degree = 0
    for (i, j) in graph.edges():
        # For each edge in the graph we collect the number of outgoing edges from v
        if i == v:
            out_degree += 1
        # For each edge in the graph we collect the number of ingoing edges in v
        if j == v:
            in_degree += 1
    # The degree of a node is given by the sum of the in-degree and out-degree of the node
    degree = in_degree + out_degree
    degree_centrality = degree / (N-1)

    return degree_centrality


def create_matrix_P(graph, alpha):
    ''' It builds the P matrix for PageRank algorithm '''

    N = len(graph.nodes())
    # To make the graph stochastic
    graph = nx.stochastic_graph(graph, weight='weight')

    # Initialize P_rw and M matrix
    P_rw = np.zeros([N, N])
    M = np.full([N, N], 1/N)

    # For each node in the graph, we define and memorize its position number in order to build the P_rw matrix
    pos = dict()
    count_pos = 0
    for node in graph.nodes():
        pos[node] = count_pos
        count_pos += 1

    # For each node we see the edges that exit from this node
    for node in pos:
        col = list()
        for (i, j) in graph.edges():
            # For each node we collect the node in which each edge goes
            if node == i:
                col.append(j)
        # Then we update the component relative to "node" and "elem" of the P_rw matrix
        for elem in col:
            P_rw[pos[node]][pos[elem]] = graph[node][elem]["weight"]

    # If the outdegree of a node is equal to 0, so the node is an "autorithy", the relative components of P_rw are equal to 1/N
    for node in pos:
        if np.sum(P_rw[pos[node]]) == 0:
            P_rw[pos[node]] = np.full([1, N], 1/N)

    # Finally we build the matrix P
    P = ((1 - alpha) * M) + (alpha * P_rw)

    return P, pos


def pagerank_score(v, graph, alpha, max_iter, tol):
    ''' To compute the pagerank algorithm:
        - We start from select at random a starting node;
        - Then we initialize the q_0 vector, such that all of his components are equal to 0, except for the component relative to the starting node, which is equal to 1;
        - for each step t of the algorithm we compute: q_t = q_(t-1) * P and we check if the algorithm converges;
        - if the algorithm have converged we return the number of iterations needed and the page rank value relative to the node $v$ in input.
    '''

    N = len(graph.nodes())
    P, pos = create_matrix_P(graph, alpha)

    if v not in graph.nodes():
        return "the node in input should be in the time interval graph"

    # Picking the start position at random
    random.seed(123)
    start_pos = random.randint(0, N)

    # Find the position of the node in input using pos dictionary, we will use this position after when returning the pr
    for key, value in pos.items():
        if key == v:
            input_node_pos = value

    # Initialize the vector q such that there are all zeros except a one in the start_position
    q_start = np.zeros(N)
    q_start[start_pos] = 1

    # Keep track of the convergence of the algorithm
    convergence = False

    for step in tqdm(range(max_iter)):
        # For every step we compute q_t = q_(t-1) * P
        q_t = np.dot(q_start, P)
        # Calculate the error in order to check the convergence
        err = sum([abs(q_t[i] - q_start[i]) for i in range(len(q_t))])
        if err < N * tol:
            print("the algorithm Page rank converges in ", step, "iterations")
            return q_t[input_node_pos]
        # Update the vector q_start
        q_start = q_t

    if convergence == False:
        print("The algorithm did not converge for ",
              max_iter, "number of iterations")

    return q_t[input_node_pos]


def fun2(v, start_interval, end_interval, metric, alpha=None, max_iter=None, tol=None):
    ''' Depending on input params, it call the correct function to compute the requested metric '''

    graph = t.build("a", start_interval, end_interval)
    if metric == "Betweeness":
        return betweeness(v, graph)
    elif metric == "PageRank" and alpha != None and max_iter != None and tol != None:
        return pagerank_score(v, graph, alpha, max_iter, tol)
    elif metric == "ClosenessCentrality":
        return closeness(v, graph)
    elif metric == "DegreeCentrality":
        return degree_centrality(v, graph)


def vis2(v, start_interval, end_interval):
    ''' It makes possible the visualization of the functionality 2 '''

    graph = t.build("a", start_interval, end_interval)
    g = nx.DiGraph()

    # Building a graph with only the neighbors of the node in input
    for neighbor in graph.neighbors(v):
        g.add_edge(v, neighbor, weight=graph[v][neighbor]["weight"])
    nx.draw(g, node_color="green", edge_color="gray",
            node_size=1000, width=1.5, with_labels=True)

    return plt.show(g)
