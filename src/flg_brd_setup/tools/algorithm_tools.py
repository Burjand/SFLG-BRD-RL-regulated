import networkx as nx

class Tools():

    def calculate_distance_matrix(self, G):

        # Calculate the distance matrix using Dijkstra's algorithm
        distance_matrix = dict(nx.all_pairs_dijkstra_path_length(G))
        return distance_matrix