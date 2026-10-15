import networkx as nx
import numpy as np

from src.tools.algorithm_tools import calculate_distance_matrix, calculate_D

class FLG_graph:

    def __init__(self, network_topology, n_nodes, seed=42, demand_distribution=('uniform',1,1), weight_distribution=('uniform',1,1), graph_density=0.3):

        self.network_topology = network_topology
        self.n_nodes = n_nodes
        self.graph_density = graph_density # Not used by the current generators (mean degree is fixed at 6)
        self.seed = seed
        self.demand_distribution = demand_distribution
        self.weight_distribution = weight_distribution
        self.rng = np.random.default_rng(seed=self.seed)

        # Generate the whole FLG environment
        self.graph, self.adj_matrix = self._generate_graph()
        assert all(isinstance(node, int) for node in self.graph.nodes()) # Check that nodes were correctly generated (Just in case)
        self._generate_demand_distribution()

        # Calculate important metrics
        self.distances = calculate_distance_matrix(self.graph) # Distances from all nodes to all nodes with Dijkstra
        self.D = calculate_D(self.graph, self.distances) # Graph diameter


    def _generate_base_topology(self, current_seed):
        # TARGET: Average Degree ~ 6 neighbors per node
        target_k = 6

        if self.network_topology == "barabasi-albert":
            # m = Number of edges to attach from a new node to existing nodes
            # Avg degree of BA is approx 2*m
            m = int(target_k / 2)  # m = 3
            return nx.barabasi_albert_graph(n=self.n_nodes, m=m, seed=current_seed)

        elif self.network_topology == "watts-strogatz":
            # k = Nearest neighbors in ring topology
            k = target_k # k = 6
            p_rewire = 0.3
            return nx.watts_strogatz_graph(n=self.n_nodes, k=k, p=p_rewire, seed=current_seed)

        else: # Erdős-Rényi
            if self.n_nodes > 1:
                p = target_k / (self.n_nodes - 1)
            else:
                p = 0

            # Safety clamp for small N to ensure p <= 1.0
            p = min(p, 1.0)

            return nx.erdos_renyi_graph(n=self.n_nodes, p=p, seed=current_seed)


    def _generate_graph(self):

        # Generate connected graph
        current_seed = self.seed
        connected = False
        while not connected:
            G = self._generate_base_topology(current_seed)
            connected = nx.is_connected(G)
            if not connected:
                current_seed += 1 # Change seed locally until graph is connected

        G = nx.convert_node_labels_to_integers(G)

        # Assign edge weights according to the chosen distribution
        for u, v in G.edges():
            if self.weight_distribution[0] == 'normal':
                weight = self.rng.normal(loc=self.weight_distribution[1], scale=self.weight_distribution[2])

            elif self.weight_distribution[0] == 'uniform':
                weight = self.rng.uniform(low=self.weight_distribution[1], high=self.weight_distribution[2])

            weight = np.abs(np.clip(np.round(weight).astype(int), 1, None))
            G.edges[u, v]['weight'] = weight

        # Create adjacency matrix
        G_adj_matrix = nx.adjacency_matrix(G, weight='weight').todense()

        return G, G_adj_matrix


    def _generate_demand_distribution(self):

        """ Generates a normalized list of demands per node """
        # Generate demand for each node according to the specified distribution

        if self.demand_distribution[0] == 'normal':
            demand = self.rng.normal(loc=self.demand_distribution[1], scale=self.demand_distribution[2], size=self.n_nodes)

        elif self.demand_distribution[0] == 'uniform':
            demand = self.rng.uniform(low=self.demand_distribution[1], high=self.demand_distribution[2], size=self.n_nodes)

        else:
            raise ValueError("Unsupported demand distribution type.")

        # Make sure demand was correctly generated
        demand = np.maximum(demand.astype(np.float32), 0.0)
        if demand.sum() == 0.0:
            demand = np.ones(self.n_nodes, dtype=np.float32)

        # Normalize demand
        total = demand.sum()
        node_weight = demand / total

        self.nodes_demands_raw = {i: demand[i] for i in range(self.n_nodes)} # Absolute node demands
        self.nodes_demands = {i: node_weight[i] for i in range(self.n_nodes)} # Normalized node demands to represent market share
