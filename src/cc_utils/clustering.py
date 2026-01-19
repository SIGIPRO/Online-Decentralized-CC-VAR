from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, Any, Set, TypeVar, List
from collections.abc import Hashable
from itertools import combinations
import itertools
import numpy as np
from numpy.linalg import matrix_power
import networkx as nx
from copy import copy
import matplotlib.pyplot as plt
from scipy.io import loadmat

@dataclass
class ModularityBasedClustering:
    """Cellular complex clustering with d-hop ad hoc algorithms. The class simulates the ad-hoc algorithm. However, this is not an exactly distributed algorithm. The class will save and return clustering results given Hodge-Laplacian and boundary matrices.
    """

    cellularComplex : Dict[int, np.ndarray]
    clusteringParameters: Dict[str, int]
    
    Nin : Dict[int, Any] = field(default_factory=dict)
    Nout : Dict[Tuple[int,int], Any] = field(default_factory=dict)
    interface: Dict[Tuple[int,int], Any] = field(default_factory = dict)
    clustered_complexes : Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    global_to_local_idx : Dict[int, Dict[int, List[int]]] = field(default_factory=dict)
    agent_graph : Dict[int, Set] = field(default_factory=dict)
    upper_lower_adjacency: Optional[list[Dict[int, list[int]]]] = None
    adjacency_graph: nx.Graph = field(default_factory=nx.Graph)
    adjacency_trees: Dict[int, Dict[int, Dict[int, List[int]]]] = field(default_factory = dict)

    ## Explanation for adjacency_trees: [dim_orig, num_cell, dim_target] dim_orig is the dimension we look at the cell. num_cell is the id of the original cell we seek adjacency tree. dim_target is the dimension we seek adjacency to. The result is a list of cells that are adjacent to num_cell at dimension dim_target.

    def __generate_graph(self, laplacian):
        adjacency = copy(laplacian)
        np.fill_diagonal(adjacency, 0)  # Ensure diagonal is computed
        adjacency = (adjacency != 0).astype(int)
        G = nx.from_numpy_array(adjacency)
        return G
    
    def __plot_graphs(self, head, coords):
        head_nodes = list(self.Nin[head][0])
        incidence_map = self.cellularComplex[1][head_nodes, :]
        adjacency = incidence_map @ incidence_map.T

       
        np.fill_diagonal(adjacency, 0)
        G = nx.from_numpy_array(adjacency)
        graph_coords = self.__sample_coords(head_nodes=head_nodes, coords=coords)
        
        nx.draw_networkx_nodes(G, graph_coords, nodelist = list(G.nodes()))
        nx.draw_networkx_edges(G, graph_coords, edgelist = list(G.edges()))


    def __sample_coords(self, head_nodes, coords):
        i = 0
        new_coords = dict()
        for node in head_nodes:
            new_coords[i] =  (float(coords[node][0]), float(coords[node][1]))
            i += 1
        return new_coords



    def plot_clusters(self): ## TODO: Continue to the plotter

        assert bool(self.Nin) and bool(self.interface) and bool(self.adjacency_trees), "Run plot_clusters after clusters are handled!!!"
        assert max(self.cellularComplex.keys()) < 3, "Only cellular complexes with maximum dimension 2 can be drawn!!!"
     
        path = self.clusteringParameters.get('position_path', None)
        

        
        try:
            positions = loadmat(path)
            coords = positions.get(self.clusteringParameters['position_name'], np.empty(shape = (), dtype = np.bool))
            coords = {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(coords.shape[0])}
        except:
            coords = nx.spring_layout
        
        heads = list(self.Nin.keys())
        cmap = plt.get_cmap("tab20" if len(heads) > 9 else "tab10")
        colors = [cmap(i) for i in np.linspace(0, 1, max(len(heads) + 1, 1))]
        # plt.figure()
        # for head in heads:
        #     self.__plot_graphs(head = head, coords = coords)

        
        G = nx.Graph()
        num_edges = self.cellularComplex[1].shape[1]
        for edge in range(num_edges):
            current_edge = tuple(self.adjacency_trees[1][edge][0])
            G.add_edge(*current_edge)
        plt.figure()
        ax = plt.gca()
        for head in heads:
            edge_list = []
            node_list = set()
            for edge in self.Nin[head][1]:
                nodes = self.adjacency_trees[1][edge][0]
                current_edge = tuple(nodes)
                edge_list.append(current_edge)
                node_list.update(set(nodes))
            
            for h in self.interface:
                if head not in h: continue
                node_list -= set(self.interface[h][0])
            # import pdb; pdb.set_trace()

            node_list = list(node_list)
            edge_list = list(set(edge_list))

            color = colors[head]
            
            nx.draw_networkx_nodes(G, coords, nodelist = node_list, node_color = [color], ax = ax)
            nx.draw_networkx_edges(G, coords, edgelist = edge_list, edge_color = [color], ax = ax)

        interface_nodes = set()
        for h in self.interface:
            interface_nodes |= set(self.interface[h][0])
        
        nx.draw_networkx_nodes(G, coords, nodelist = interface_nodes, node_color = [colors[-1]], ax = ax)
        plt.show()




        
        # # B_up = self.cellularComplex.get(int(self.clusteringParameters.get('dim', 0)), None)
        # # if B_up is not None:
        # #     laplacian_node = B_up @ B_up.T

        # # G_lower = self.__generate_graph(laplacian=laplacian_node)

        # # adjacency_node = copy(laplacian_node)
        # # np.fill_diagonal(adjacency_node, 0)  # Ensure diagonal is computed
        # # adjacency_node = (adjacency_node != 0).astype(int)
        # # G_node = nx.from_numpy_array(adjacency_node)

            
        # # plt.figure()
    
        # # import pdb; pdb.set_trace()

        # plt.figure()
        # cmap = plt.get_cmap("tab20" if len(communities) > 10 else "tab10")
        # colors = [cmap(i) for i in np.linspace(0, 1, max(len(communities), 1))]
        # # node_to_cluster = {
        # #     node: cluster_idx
        # #     for cluster_idx, subnodes in enumerate(communities)
        # #     for node in subnodes
        # # }
        # # node_colors = [colors[node_to_cluster[node]] for node in G.nodes()] 
        
        # ax = plt.gca()
        # for idx, subedges in enumerate(communities):
            

            
        #     B_down_new = B_down[:, list(subedges)]
        #     laplacian_new = B_down_new @ B_down_new.T
        #     lap_diag = laplacian_new.diagonal()
            
        #     indices = list(np.nonzero(lap_diag)[0])
        #     indices = [int(i) for i in indices]
        #     # import pdb; pdb.set_trace()

        #     B_down_new = B_down_new[indices, :]
        #     laplacian_new = B_down_new @ B_down_new.T
        #     adjacency_new = copy(laplacian_new)
        #     np.fill_diagonal(adjacency_new, 0)  # Ensure diagonal is computed
        #     adjacency_new = (adjacency_new != 0).astype(int)
        #     G_new = nx.from_numpy_array(adjacency_new) 

            
        #     pos = dict()
        
        #     for i in range(len(indices)):
        #         pos[i] = positions[indices[i]]

        #     # import pdb; pdb.set_trace()
            
        #     # G_sub = G.subgraph(subnodes)
        #     color = colors[idx]
        #     nx.draw_networkx_nodes(G, pos, nodelist=list(G_new.nodes()), node_color=[color], ax=ax)
        #     nx.draw_networkx_edges(G, pos, edgelist=list(G_new.edges()), edge_color=[color], ax=ax)
        # # nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color="lightgray", alpha=0.5, ax=ax)
        # # nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_colors, ax=ax)
        # ax.set_axis_off()
        # plt.show()
        # import pdb; pdb.set_trace()
    
    def __post_init__(self):
        dimensions = list(self.cellularComplex.keys())
        max_dim = max(dimensions)
        
        if self.clusteringParameters['dim'] < 0 or self.clusteringParameters['dim'] > max_dim or self.clusteringParameters['dim'] is None: raise ValueError("Invalid dimension for clustering.")

        
        requested_dim = int(self.clusteringParameters.get('dim', 0))
        self.dim = requested_dim
        B_down = self.cellularComplex.get(requested_dim, None)
        # B_up = self.cellularComplex.get(requested_dim + 1, None)

        L_lower = 0
        # L_upper = 0 # Do not consider L_upper as it creates irregulatiry 

        if B_down is not None:
            L_lower = B_down.T @ B_down
        # if B_up is not None:
        #     L_upper = B_up @ B_up.T
        
        # laplacian = L_lower + L_upper
        laplacian = L_lower

        # Construct loopless graph from L
        G = self.__generate_graph(laplacian=laplacian)

        Q = int(self.clusteringParameters['Q-hop'])
        assert Q>=0, "Q should be greater or equal to 0!!!"
        
        laplacian_Q = matrix_power(a = laplacian, n = Q)

        adjacency_self = laplacian_Q
        np.fill_diagonal(a = adjacency_self, val = 0)
        adjacency_self = adjacency_self != 0

       

        self.__generate_adj_trees()
        
        
        
        communities = nx.community.greedy_modularity_communities(G = G, best_n = self.clusteringParameters['best_n'], resolution=self.clusteringParameters['resolution'])
        ## NIN BLOCK
        for cluster_idx, cells in enumerate(communities):
            # 
            self.Nin[cluster_idx] = dict()
            self.Nin[cluster_idx][requested_dim] = list(cells)

            for dim in range(0, max_dim + 1):
                if dim == requested_dim: continue

                self.Nin[cluster_idx][dim] = set()

                for cell in cells:
                    self.Nin[cluster_idx][dim] |= set(self.adjacency_trees[requested_dim][cell][dim])

        cluster_id = list(self.Nin.keys())
        # import pdb; pdb.set_trace()

        for c1,c2 in combinations(cluster_id, 2):
            ott = (c1, c2)
            tto = (c2, c1)

            
            adjacents_c1 = self.__get_self_adjacents(c = c1, dim = requested_dim, adjacency_self = adjacency_self)
            adjacents_c2 = self.__get_self_adjacents(c = c2, dim = requested_dim, adjacency_self = adjacency_self)
            try: self.Nout[ott]
            except: self.Nout[ott] = dict()
            try: self.Nout[tto]
            except: self.Nout[tto] = dict()
            try: self.interface[ott]
            except: self.interface[ott] = dict()


            ## NOUT BLOCK
            self.Nout[ott][requested_dim] = (set(adjacents_c1) & set(self.Nin[c2][requested_dim])) - set(self.Nin[c1][requested_dim])

            self.Nout[tto][requested_dim] = (set(adjacents_c2) & set(self.Nin[c1][requested_dim])) - set(self.Nin[c2][requested_dim])

            ott_connection = False

            for dim in range(0, max_dim + 1):
                if dim == requested_dim: continue

                self.Nout[ott][dim] = set()
                self.Nout[tto][dim] = set()
                self.interface[ott][dim] = set()

                for cell in adjacents_c1:
                    self.Nout[ott][dim] |= set(self.adjacency_trees[requested_dim][cell][dim])

                for cell in adjacents_c2:
                    self.Nout[tto][dim] |= set(self.adjacency_trees[requested_dim][cell][dim])
                
                self.Nout[ott][dim] -= self.Nin[c1][dim]
                self.Nout[tto][dim] -= self.Nin[c2][dim]

                ## INTERFACE BLOCK

                self.interface[ott][dim] = set(self.Nin[c1][dim] & self.Nin[c2][dim])

                ott_connection = ott_connection or bool(self.interface[ott][dim])

                ## CORRECT WITH RESPECT TO INTERFACE
                self.Nout[ott][dim] -= self.interface[ott][dim]
                self.Nout[tto][dim] -= self.interface[ott][dim]

                self.Nin[c1][dim] -= self.interface[ott][dim]
                self.Nin[c2][dim] -= self.interface[ott][dim]


                ## TODO: Implement clustered_complexes, and global_to_local_idx
            if ott_connection:
                try: self.agent_graph[c1]
                except: self.agent_graph[c1] = set()

                self.agent_graph[c1].update({c2})

                try: self.agent_graph[c2]
                except: self.agent_graph[c2] = set()

                self.agent_graph[c2].update({c1})

        
        for dim in set(self.cellularComplex.keys()) | {0}:
            for h1 in self.Nin:
                dim_adjacencies = set()
                lower_adjacencies = set()

                dim_adjacencies.update(self.Nin.get(h1, {}).get(dim, set()))
                lower_adjacencies.update(self.Nin.get(h1, {}).get(dim - 1, set()))

                for h2 in self.agent_graph.get(h1, set()):
                    key = (h1, h2)
                    dim_adjacencies.update(self.interface.get(key, {}).get(dim, set()))
                    lower_adjacencies.update(self.interface.get(key, {}).get(dim - 1, set()))
                    dim_adjacencies.update(self.Nout.get(key, {}).get(dim, set()))
                    lower_adjacencies.update(self.Nout.get(key, {}).get(dim - 1, set()))
                
                if h1 not in self.clustered_complexes:
                    self.clustered_complexes[h1] = dict()
                    self.global_to_local_idx[h1] = dict()
                if dim not in self.global_to_local_idx[h1]:
                    self.global_to_local_idx[h1][dim] = list()

                # global_idx = list(lower_adjacencies)
                global_idx = list(dim_adjacencies)
                # local_idx = [i for i in range(len(global_idx))]
                if dim in self.cellularComplex:
                    self.clustered_complexes[h1][dim] = self.cellularComplex[dim][np.ix_(list(lower_adjacencies), list(dim_adjacencies))]
            

                self.global_to_local_idx[h1][dim] = global_idx
                

            

              # convert sets to lists for portability
        for h in self.Nin:
            for dim in self.Nin[h]:
                if isinstance(self.Nin[h][dim], set):
                    self.Nin[h][dim] = list(self.Nin[h][dim])
        for key in self.Nout:
            for dim in self.Nout[key]:
                if isinstance(self.Nout[key][dim], set):
                    self.Nout[key][dim] = list(self.Nout[key][dim])
        for key in self.interface:
            for dim in self.interface[key]:
                if isinstance(self.interface[key][dim], set):
                    self.interface[key][dim] = list(self.interface[key][dim])



        
        # path = '../../data/Input/noaa_coastwatch_cellular/longlat.mat'

    def __get_self_adjacents(self, c, dim, adjacency_self):
        laplacian_part = adjacency_self[self.Nin[c][dim], :]
        adjacents = np.any(laplacian_part, axis = 0)
        adjacents = np.where(adjacents)

        return adjacents[0]
    
    def __generate_adj_trees(self):
        self.adjacency_trees = dict()
        all_dim = set(self.cellularComplex.keys()) | {0}

        

        for dim_orig in all_dim:
            self.adjacency_trees[dim_orig] = dict()

            
            

            if dim_orig == 0:
                num_cells = self.cellularComplex[1].shape[0]
            else:
                num_cells = self.cellularComplex[dim_orig].shape[1]

            
            dim_boundaries = self.___generate_dim_boundaries(dim=dim_orig, all_dim=all_dim, num_cells=num_cells)


            for cell in range(0, num_cells):
                self.adjacency_trees[dim_orig][cell] = dict()

                for dim_target in all_dim:
                    self.adjacency_trees[dim_orig][cell][dim_target] = list(np.where(dim_boundaries[dim_target][cell, :])[0])
    
    def ___generate_dim_boundaries(self, dim, all_dim, num_cells):
        dim_boundaries = dict()

        boundary_lower = np.eye(num_cells, dtype = np.bool)

        ## LOWER BOUNDARIES
        for dim_target in range(dim - 1, -1, -1):
            boundary_lower = self.cellularComplex[dim_target + 1].astype(np.bool) @ boundary_lower
            dim_boundaries[dim_target] = boundary_lower.T
        
        boundary_upper = np.eye(num_cells, dtype = np.bool)

        ## UPPER BOUNDARIES
        for dim_target in range(dim + 1, max(all_dim) + 1):
            boundary_upper = boundary_upper @ self.cellularComplex[dim_target].astype(np.bool)
            dim_boundaries[dim_target] = boundary_upper

        B_down = self.cellularComplex.get(dim, None)
        B_up = self.cellularComplex.get(dim + 1, None)
        L_lower = 0
        L_upper = 0

        if B_down is not None:
            L_lower = B_down.T @ B_down
        if B_up is not None:
            L_upper = B_up @ B_up.T

        laplacian = L_lower + L_upper
        laplacian = laplacian.astype(np.bool)
        # np.fill_diagonal(a = laplacian, val = False)

        dim_boundaries[dim] = laplacian

        return dim_boundaries





        
            
## Continue from here.

@dataclass
class CellularComplexFakeClustering:
    """Cellular complex clustering with d-hop ad hoc algorithms. The class simulates the ad-hoc algorithm. However, this is not an exactly distributed algorithm. The class will save and return clustering results given Hodge-Laplacian and boundary matrices.
    """

    cellularComplex : Dict[int, np.ndarray]
    clusteringParameters: Dict[str, int]
    
    Nin : Dict[int, Any] = field(default_factory=dict)
    Nout : Dict[Tuple[int,int], Any] = field(default_factory=dict)
    interface: Dict[Tuple[int,int], Any] = field(default_factory = dict)
    clustered_complexes : Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    global_to_local_idx : Dict[int, Dict[int, List[int]]] = field(default_factory=dict)
    agent_graph : Dict[int, Set] = field(default_factory=dict)
    upper_lower_adjacency: Optional[list[Dict[int, list[int]]]] = None
    

    def __post_init__(self):
        """Initialize the clustering model."""

        dimensions = list(self.cellularComplex.keys())
        max_dim = max(dimensions)
        
        if self.clusteringParameters['dim'] < 0 or self.clusteringParameters['dim'] > max_dim or self.clusteringParameters['dim'] is None: raise ValueError("Invalid dimension for clustering.")

        
        requested_dim = int(self.clusteringParameters.get('dim', 0))
        B_down = self.cellularComplex.get(requested_dim, None)
        B_up = self.cellularComplex.get(requested_dim + 1, None)

        L_lower = 0
        L_upper = 0

        if B_down is not None:
            L_lower = B_down.T @ B_down
        if B_up is not None:
            L_upper = B_up @ B_up.T
        
        laplacian = L_lower + L_upper

        # Construct loopless graph from L
        adjacency = laplacian
        np.fill_diagonal(adjacency, 0)  # Ensure diagonal is computed
        adjacency = (adjacency != 0).astype(int)
        G = nx.from_numpy_array(adjacency)
        d = int(self.clusteringParameters.get('d', 1))

        heads = select_clusterheads_maxmin(G = G, d = d)
        head_of, parent, depth = form_clusters_tree_bfs(G = G, clusterheads = heads, d = d)


        Q = int(self.clusteringParameters['Q-hop'])
        assert Q>=0, "Q should be greater or equal to 0!!!"
        
        laplacian_Q = matrix_power(a = laplacian, n = Q)

        adjacency_self = laplacian_Q
        np.fill_diagonal(a = adjacency_self, val = 0)
        adjacency_self = adjacency_self != 0

        # Build per-cell adjacency dictionaries across all other dimensions
        self.upper_lower_adjacency = self._compute_upper_lower_adjacency(
            base_dim=requested_dim,
            max_dim=max_dim,
        )


        for cell, h in head_of.items():
            if h not in self.Nin:
                self.Nin[h] = dict()
                self.Nin[h][requested_dim] = set()
         

            self.Nin[h][requested_dim].add(cell)

            adjacents = self.upper_lower_adjacency[cell]

            for dim in adjacents.keys():
                if dim not in self.Nin[h]:
                    self.Nin[h][dim] = set()
                self.Nin[h][dim].update(adjacents[dim])


        for h1, h2 in combinations(heads, 2):
                key = (h1, h2)
                key_rev = (h2, h1)

                if h1 not in self.agent_graph.keys():
                    self.agent_graph[h1] = set()
                if h2 not in self.agent_graph.keys():
                    self.agent_graph[h2] = set()


                if key not in self.Nout:
                    self.Nout[key] = dict()
                    self.Nout[key][requested_dim] = set()
                if key_rev not in self.Nout:
                    self.Nout[key_rev] = dict()
                    self.Nout[key_rev][requested_dim] = set()

                
                self.interface[key] = dict()
                dims_h1 = self.Nin.get(h1, {})
                dims_h2 = self.Nin.get(h2, {})

                for dim in (set(dims_h1.keys()) & set(dims_h2.keys()) - {requested_dim}):
                    inter = set(dims_h1[dim]) & set(dims_h2[dim])
                    if inter:
                        self.interface[key][dim] = inter
                        self.agent_graph[h1].add(h2)
                        self.agent_graph[h2].add(h1)

                     

                for cell in self.Nin[h1][requested_dim]:
                    requested_adj = self.Nin[h2][requested_dim] & set(np.flatnonzero(adjacency_self[cell]))
                    self.Nout[key][requested_dim].update(requested_adj)
                for cell in self.Nin[h2][requested_dim]:
                    requested_adj = self.Nin[h1][requested_dim] & set(np.flatnonzero(adjacency_self[cell]))
                    self.Nout[key_rev][requested_dim].update(requested_adj)

                for dim in (set(dims_h1.keys()) & set(dims_h2.keys()) - {requested_dim}):
                    if dim not in self.Nout[key]:
                        self.Nout[key][dim] = set()
                    for cell in self.Nout[key][requested_dim]:
                        adjacents = set(self.upper_lower_adjacency[cell].get(dim, []))
                        adjacents -= self.interface.get(key, {}).get(dim, set())
                        self.Nout[key][dim].update(adjacents)
                    if dim not in self.Nout[key_rev]:
                        self.Nout[key_rev][dim] = set()
                    for cell in self.Nout[key_rev][requested_dim]:
                        adjacents = set(self.upper_lower_adjacency[cell].get(dim, []))
                        adjacents -= self.interface.get(key, {}).get(dim, set())
                        self.Nout[key_rev][dim].update(adjacents)

        for h1 in heads:
            for h2 in self.agent_graph.get(h1, set()):
                key = (h1, h2)
                for dim, inter in self.interface.get(key, {}).items():
                    if dim in self.Nin.get(h1, {}):
                        self.Nin[h1][dim] -= inter
                    if dim in self.Nin.get(h2, {}):
                        self.Nin[h2][dim] -= inter
            
        for dim in set(self.cellularComplex.keys()) | {0}:
            for h1 in heads:
                dim_adjacencies = set()
                lower_adjacencies = set()

                dim_adjacencies.update(self.Nin.get(h1, {}).get(dim, set()))
                lower_adjacencies.update(self.Nin.get(h1, {}).get(dim - 1, set()))

                for h2 in self.agent_graph.get(h1, set()):
                    key = (h1, h2)
                    dim_adjacencies.update(self.interface.get(key, {}).get(dim, set()))
                    lower_adjacencies.update(self.interface.get(key, {}).get(dim - 1, set()))
                    dim_adjacencies.update(self.Nout.get(key, {}).get(dim, set()))
                    lower_adjacencies.update(self.Nout.get(key, {}).get(dim - 1, set()))
                
                if h1 not in self.clustered_complexes:
                    self.clustered_complexes[h1] = dict()
                    self.global_to_local_idx[h1] = dict()
                if dim not in self.global_to_local_idx[h1]:
                    self.global_to_local_idx[h1][dim] = list()

                # global_idx = list(lower_adjacencies)
                global_idx = list(dim_adjacencies)
                # local_idx = [i for i in range(len(global_idx))]
                if dim in self.cellularComplex:
                    self.clustered_complexes[h1][dim] = self.cellularComplex[dim][np.ix_(list(lower_adjacencies), list(dim_adjacencies))]
            

                self.global_to_local_idx[h1][dim] = global_idx

        # convert sets to lists for portability
        for h in self.Nin:
            for dim in self.Nin[h]:
                if isinstance(self.Nin[h][dim], set):
                    self.Nin[h][dim] = list(self.Nin[h][dim])
        for key in self.Nout:
            for dim in self.Nout[key]:
                if isinstance(self.Nout[key][dim], set):
                    self.Nout[key][dim] = list(self.Nout[key][dim])
        for key in self.interface:
            for dim in self.interface[key]:
                if isinstance(self.interface[key][dim], set):
                    self.interface[key][dim] = list(self.interface[key][dim])
    

    def _infer_cell_counts(self) -> Dict[int, int]:
        """Infer number of cells per dimension from boundary maps."""
        counts: Dict[int, int] = {}
        for dim, B in self.cellularComplex.items():
            counts[dim] = B.shape[1]
            counts[dim - 1] = max(counts.get(dim - 1, 0), B.shape[0])
        return counts

    def _compute_upper_lower_adjacency(
        self,
        base_dim: int,
        max_dim: int,
    ) -> list[Dict[int, list[int]]]:
        """
        For each base_dim cell i, build a dictionary mapping target_dim -> list of adjacent cells.
        Adjacency is derived by chaining boundary maps downward (faces) and upward (cofaces).
        """
        counts = self._infer_cell_counts()
        if base_dim not in counts:
            raise ValueError(f"Cannot infer number of cells in dimension {base_dim}")
        n_base = counts[base_dim]

        adjacency: list[Dict[int, list[int]]] = [dict() for _ in range(n_base)]

        # Downward (faces): base_dim -> base_dim-1 -> ... -> 0
        if base_dim >= 1 and base_dim in self.cellularComplex:
            down_reach = (self.cellularComplex[base_dim] != 0).T  # (n_base x n_{base-1})
            for dim in range(base_dim - 1, -1, -1):
                for i in range(n_base):
                    faces = np.nonzero(down_reach[i])[0]
                    if faces.size > 0:
                        adjacency[i][dim] = faces.tolist()
                if dim > 0:
                    B_next = self.cellularComplex.get(dim, None)
                    if B_next is None:
                        break
                    down_reach = (down_reach @ (B_next != 0).T) != 0

        # Upward (cofaces): base_dim -> base_dim+1 -> ... -> max_dim
        if (base_dim + 1) in self.cellularComplex:
            up_reach = (self.cellularComplex[base_dim + 1] != 0)  # (n_base x n_{base+1})
            for dim in range(base_dim + 1, max_dim + 1):
                for i in range(n_base):
                    cofaces = np.nonzero(up_reach[i])[0]
                    if cofaces.size > 0:
                        adjacency[i][dim] = cofaces.tolist()
                B_next = self.cellularComplex.get(dim + 1, None)
                if B_next is None:
                    break
                up_reach = (up_reach @ (B_next != 0)) != 0

        return adjacency




## A ChatGPT implementation of the correction of the ad-hoc clustering method for d-hop clustering.
Node = TypeVar("Node", bound=Hashable)


def _injective_value(score: Dict[Node, Any], node: Node) -> Tuple[Any, Node]:
    """Make values totally ordered + injective by tie-breaking with node id."""
    return (score[node], node)


def select_clusterheads_maxmin(
    G: nx.Graph,
    d: int,
    score: Optional[Dict[Node, Any]] = None,
) -> set[Node]:
    """
    Generalized Max–Min d-clusterhead selection (2d rounds) from Mazieux et al. (2007).
    Clusterheads S = { x : W_{2d}(x) == v(x) }.  [oai_citation:2‡dl.ifip.org](https://dl.ifip.org/db/conf/networking/networking2007/MazieuxMB07.pdf)

    score: per-node criterion (e.g., energy/capacity). If None, uses node id.
    Values are made injective via (score, node) tie-break.
    """
    if d <= 0:
        raise ValueError("d must be >= 1")
    if score is None:
        score = {v: v for v in G.nodes()}

    # v(x)
    v_val = {x: _injective_value(score, x) for x in G.nodes()}

    # Winner/Sender lists, but we only need the current W_k, S_k.
    W = v_val.copy()
    S = {x: x for x in G.nodes()}

    def one_hop_plus_self(x: Node) -> Iterable[Node]:
        yield x
        yield from G.neighbors(x)

    # Max phase: k = 1..d, W_k(x) = max_{y in N1+(x)} W_{k-1}(y), S_k(x)=argmax
    for _k in range(1, d + 1):
        W_new, S_new = {}, {}
        for x in G.nodes():
            best_y = max(one_hop_plus_self(x), key=lambda y: W[y])
            W_new[x] = W[best_y]
            S_new[x] = best_y
        W, S = W_new, S_new

    # Min phase: k = d+1..2d, W_k(x) = min_{y in N1+(x)} W_{k-1}(y), S_k(x)=argmin
    for _k in range(d + 1, 2 * d + 1):
        W_new, S_new = {}, {}
        for x in G.nodes():
            best_y = min(one_hop_plus_self(x), key=lambda y: W[y])
            W_new[x] = W[best_y]
            S_new[x] = best_y
        W, S = W_new, S_new

    # Definition 1: S = {x : W_{2d}(x) = v(x)}  [oai_citation:3‡dl.ifip.org](https://dl.ifip.org/db/conf/networking/networking2007/MazieuxMB07.pdf)
    heads = {x for x in G.nodes() if W[x] == v_val[x]}
    return heads


def form_clusters_tree_bfs(
    G: nx.Graph,
    clusterheads: Iterable[Node],
    d: int,
    score: Optional[Dict[Node, Any]] = None,
) -> Tuple[Dict[Node, Node], Dict[Node, Optional[Node]], Dict[Node, int]]:
    """
    Corrected cluster formation from Mazieux et al. (2007), Section 2.2:
    heads announce themselves; nodes join at 1 hop, then 2 hops, ... up to d,
    producing tree clusters with a clusterhead root (no loops).  [oai_citation:4‡dl.ifip.org](https://dl.ifip.org/db/conf/networking/networking2007/MazieuxMB07.pdf)

    Tie-break when multiple head-frontiers reach a node at same depth:
      pick head with larger v(head) (criterion), then head id.
    """
    if d <= 0:
        raise ValueError("d must be >= 1")
    heads = list(clusterheads)
    if not heads:
        raise ValueError("No clusterheads provided (head election returned empty set).")

    if score is None:
        score = {v: v for v in G.nodes()}
    v_head = {h: _injective_value(score, h) for h in heads}

    # state
    head_of: Dict[Node, Node] = {h: h for h in heads}
    parent: Dict[Node, Optional[Node]] = {h: None for h in heads}
    depth: Dict[Node, int] = {h: 0 for h in heads}

    # frontier contains nodes assigned at previous depth
    frontier = set(heads)

    for dist in range(1, d + 1):
        next_frontier = set()
        for u in frontier:
            hu = head_of[u]
            for v in G.neighbors(u):
                # candidate assignment for v: head hu at distance dist via parent u
                cand = (dist, v_head[hu], hu, u)  # distance first, then "best head"
                if v not in depth:
                    # unclustered -> take it
                    head_of[v] = hu
                    parent[v] = u
                    depth[v] = dist
                    next_frontier.add(v)
                else:
                    # already assigned: only replace if strictly better
                    cur_h = head_of[v]
                    cur_parent = parent[v]
                    cur = (depth[v], v_head[cur_h], cur_h, cur_parent)
                    if cand < cur:
                        head_of[v] = hu
                        parent[v] = u
                        depth[v] = dist
                        next_frontier.add(v)
        frontier = next_frontier

    # nodes not reached within d hops remain unassigned (should not happen if heads form a d-dominating set)
    return head_of, parent, depth
