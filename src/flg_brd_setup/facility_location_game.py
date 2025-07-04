import networkx as nx
import numpy as np

class FLG_environment:

    def __init__(self, n_nodes, potential_facilities, seed=42, demand_distribution=('normal',20,5), weight_distribution=('normal',5,1)):
        
        self.n_nodes = n_nodes
        self.seed = seed
        self.demand_distribution = demand_distribution
        self.weight_distribution = weight_distribution
        self.potential_facilities = potential_facilities
        self.rng = np.random.default_rng(seed=self.seed)        

        self.check_potential_facilities()
        self.generate_flg_env()


    def check_potential_facilities(self):

        # Make sure that the number of potential facilities is a valid number

        if not isinstance(self.potential_facilities, int):
            raise ValueError("The number of potential facilities must be an integer.")
        if self.potential_facilities > self.n_nodes:
            raise ValueError("The number of potential facilities cannot exceed number of nodes.")
        if self.potential_facilities < 1:
            raise ValueError("The number of potential facilities must be at least 1.")
        return 0
    

    def generate_flg_env(self):

        # Generate the whole FLG environment

        self.graph, self.adj_matrix = self.generate_tree()
        self.node_demand = self.generate_demand_distribution()
        self.potential_facilities_mask = self.select_potential_facilities()

    
    def generate_tree(self):

        # Generate tree graph
        G = nx.random_labeled_tree(n=self.n_nodes, seed=self.seed)
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
    

    def generate_demand_distribution(self):

        # Generate demand for each node according to the specified distribution

        if self.demand_distribution[0] == 'normal':
            demand = self.rng.normal(loc=self.demand_distribution[1], scale=self.demand_distribution[2], size=self.n_nodes)

        elif self.demand_distribution[0] == 'uniform':
            demand = self.rng.uniform(low=self.demand_distribution[1], high=self.demand_distribution[2], size=self.n_nodes)

        else:
            raise ValueError("Unsupported demand distribution type.")
        
        demand = np.round(np.abs(demand)).astype(int)
        demand = np.clip(demand, 1, None) # Ensure demand >= 1
        return {i: demand[i] for i in range(self.n_nodes)}
    

    def select_potential_facilities(self):

        # Select nodes to become potential facilities

        facilities = self.rng.choice(self.n_nodes, size=self.potential_facilities, replace=False).tolist()
        return [1 if i in facilities else 0 for i in range(self.n_nodes)]
        
        return selected_nodes