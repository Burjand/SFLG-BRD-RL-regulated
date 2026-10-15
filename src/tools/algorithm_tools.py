import networkx as nx

def calculate_distance_matrix(G):

    """
    
    Calculate the distance matrix using Dijkstra's algorithm

    Example result:
    {
      0: {0: 0, 1: 1, 2: 2},
      1: {0: 1, 1: 0, 2: 1},
      2: {0: 2, 1: 1, 2: 0}
    }
    So, the distance from node 0 to node 2 is 2 """

    return dict(nx.all_pairs_dijkstra_path_length(G)) # (dictionary of dictionaries)

def calculate_D(G, distance_matrix):

    """ Calculate maximum shortest path in graph """

    n_nodes = len(distance_matrix)

    if n_nodes <= 3000:
        D = max(max(distances.values()) for distances in distance_matrix.values())
        
    else:
        D = nx.algorithms.distance_measures.approximation_diameter(G)

    return D