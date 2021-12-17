import trees as t
import networkx as nx
import matplotlib.pyplot as plt


def find_unique_nodes(start_1, end_1, start_2, end_2):
    ''' It is a function that we only need to find the two unique nodes for the two time intervals (nodes on which we then call in functionality 4) '''

    # Build two graphs for the two time intervals
    graph_1 = t.build("a", start_1, end_1)
    graph_2 = t.build("a", start_2, end_2)
    u_1 = list()
    u_2 = list()
    # Collecting the nodes in the two graphs
    nodes_1 = list(graph_1.nodes())
    nodes_2 = list(graph_2.nodes())
    # Collecting the nodes that are unique for interval 1
    for n in nodes_1:
        if n not in nodes_2:
            u_1.append(n)
    # Collecting the nodes that are unique for interval 2
    for m in nodes_2:
        if m not in nodes_1:
            u_2.append(m)
    return u_1, u_2


def check_unique_nodes(start_1, end_1, start_2, end_2, user_1, user_2):
    ''' It is a function to check whether the two nodes in input are unique for each time interval '''

    check = True

    # Build the two graphs for the intervals time in input
    graph_1 = t.build("a", start_1, end_1)
    graph_2 = t.build("a", start_2, end_2)

    # Collecting the nodes in the two graphs
    nodes_1 = list(graph_1.nodes())
    nodes_2 = list(graph_2.nodes())

    # Check wether the user_1 node in input is only in the graph_1
    # Check wether the user_2 node in input is only in the graph_2
    if (user_1 in nodes_2) or (user_2 in nodes_1):
        check = False

    return check


def mincut(graph, graph_copy, source, target):
    ''' Performs mincut (with FordFulkerson algorithm) '''

    # Initialize that the path exist as true
    path_exist = True
    edges = set()
    # Loop until exist a path between the source and the target
    while path_exist:
        # Initialize visited to keep track of the visited nodes, predecessor to keep track of the path
        visited, predecessor, queue = list(), dict(), list()
        visited.append(source)
        queue.append(source)
        # Loop on the queue
        while queue:
            # Select the first node from the queue
            v = queue.pop(0)
            for u in graph.neighbors(v):
                # For each adjacent node to v we consider it if the node has not been visisted yet and if its weight is bigger than 0
                if (u not in visited) and (graph[v][u]["weight"] > 0):
                    # We add the node u in queue and visisted list
                    queue.append(u)
                    visited.append(u)
                    predecessor[u] = v
                    # If we find a path between the source and the target we compute the path_flow and we update the edges of the path
                    if u == target:
                        last = target
                        while last != source:
                            edges.add((predecessor[last], last))
                            last = predecessor[last]
                        path_flow = min(graph[j][i]["weight"]
                                        for i, j in predecessor.items())
                        for i, j in predecessor.items():
                            graph[j][i]["weight"] -= path_flow

        # If it doesn't exist a path between source and target we update the boolan to stop the loop
        if target not in predecessor:
            path_exist = False

    # Counting the number of edges we need to delect to disconnect source and target and collecting the edges we remove
    min_edges = 0
    to_remove = list()
    for (i, j) in edges:
        if graph[i][j]["weight"] == 0 and graph_copy[i][j]["weight"] > 0:
            min_edges += 1
            to_remove.append([i, j])

    return min_edges, to_remove, edges


def fun4(start_1, end_1, start_2, end_2, user_1, user_2):
    ''' It "wraps" all functions to accomplish the task '''

    # Check if the the nodes are unique for the two separated graphs
    check = check_unique_nodes(start_1, end_1, start_2, end_2, user_1, user_2)
    if check == False:
        return "The nodes in input should be unique for each interval"

    # Create the graph considering two different intervals of time
    graph = t.build("a", start_1, end_1, start_2, end_2)
    graph_copy = graph.copy()
    # Using FordFulkerson function to find the min_edges
    min_edges, to_remove, edges = mincut(graph, graph_copy, user_1, user_2)

    return min_edges


def vis4(start_1, end_1, start_2, end_2, user_1, user_2):
    ''' It makes possible the visualization for functionality 4 '''

    # Check if the the nodes are unique for the two separated graphs
    check = check_unique_nodes(start_1, end_1, start_2, end_2, user_1, user_2)
    if check == False:
        return "The nodes in input should be unique for each interval"
    # Create the graph considering two different intervals of time
    graph = t.build("a", start_1, end_1, start_2, end_2)
    graph_copy = graph.copy()
    # We use mincut to compute the number of edges to remove, the (u,v) edge that we remove and the (u,v) edge that is in the path between u1 and u2
    min_edges, edge_to_remove, edges = mincut(
        graph, graph_copy, user_1, user_2)
    print(edge_to_remove, edges)

    # Initialize the graph
    g = nx.DiGraph()

    # For each edge in the path between the user_1 and user_2
    for (i, j) in edges:
        # If the edge is in the list of edges we have to remove we add it to g and we color it in red
        if list((i, j)) in edge_to_remove:
            g.add_edge(i, j, color="r")
        # If the edge is not in the list of edges we have to remove, we add it to g and we color it in black
        else:
            g.add_edge(i, j, color="black")

    # We do a copy of the graph in order to be able to compute the point 3
    g_copy = g.copy()
    # If the len of edges in the path is equal to th edges we remove, we add the neighbor edges of the nodes in g to give a more general idea of the graph
    if len(edge_to_remove) == len(edges):
        for node in g_copy:
            count = 0
            for (u, v) in graph_copy.out_edges(node):
                # We add the neighnor edge to the graph if it's not already in the graph and if we did not add already more than 3 edges to g
                if not g.has_edge(u, v) and count < 3:
                    g.add_edge(u, v, color="black")
                    count += 1

    colors = [g[u][v]['color'] for u, v in g.edges]
    nx.draw(g, node_color="green", edge_color=colors,
            node_size=1000, width=1.5, with_labels=True)

    return plt.show(g)
