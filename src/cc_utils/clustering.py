from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, Any, Set
from itertools import combinations
import numpy as np
from numpy.linalg import matrix_power
import networkx as nx


## Continue from here.

@dataclass
class CellularComplexFakeClustering:
    """Cellular complex clustering with d-hop ad hoc algorithms. The class simulates the ad-hoc algorithm. However, this is not an exactly distributed algorithm. The class will save and return clustering results given Hodge-Laplacian and boundary matrices.
    """

    cellularComplex : Dict[int, np.ndarray]
    clusteringParameters: Dict[str, float]
    
    Nin : Dict[int, Any] = field(default_factory=dict)
    Nout : Dict[Tuple[int,int], Any] = field(default_factory=dict)
    interface: Dict[Tuple[int,int], Any] = field(default_factory = dict)
    clustered_complexes : Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
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


        # for cell, h in head_of.items():
        #     self.Nin[h].append(cell)
        #     self.Nout[h].append(h) ## Check this line later
        Q = int(self.clusteringParameters['Q-hop'])
        assert Q>=0, "Q should be greater or equal to 0!!!"
        
        laplacian_Q = matrix_power(a = laplacian, n = Q)

        adjacency_self = laplacian_Q
        np.fill_diagonal(a = adjacency_self, val = 0)
        adjacency_self = adjacency_self != 0

        # if B_up is not None:
        #     adjacency_upper = laplacian_Q @ B_up
        #     adjacency_upper = adjacency_upper != 0
        # if B_down is not None:
        #     adjacency_lower = B_down @ laplacian_Q 
        #     adjacency_lower = adjacency_lower != 0

        # Build per-cell adjacency dictionaries across all other dimensions
        self.upper_lower_adjacency = self._compute_upper_lower_adjacency(
            base_dim=requested_dim,
            max_dim=max_dim,
        )

        # self.Nin[requested_dim] = dict()
        # self.Nout[requested_dim] = dict()
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


        # self.agent_graph = dict()
        for h1, h2 in combinations(heads, 2):
                key = (h1, h2)

                if h1 not in self.agent_graph.keys():
                    self.agent_graph[h1] = set()

                # try:
                #     self.agent_graph[heads_iterator._i]
                # except:
                #     self.agent_graph[heads_iterator._i].append([])

                if key not in self.Nout:
                    self.Nout[key] = dict()
                    self.Nout[key][requested_dim] = set()

                
                self.interface[key] = dict()
                dims_h1 = self.Nin.get(h1, {})
                dims_h2 = self.Nin.get(h2, {})

                for dim in (set(dims_h1.keys()) & set(dims_h2.keys()) - {requested_dim}):
                    inter = set(dims_h1[dim]) & set(dims_h2[dim])
                    if inter:
                        self.interface[key][dim] = inter
                        self.agent_graph[h1].add(h2)

                     

                for cell in self.Nin[h1][requested_dim]:
                    requested_adj = self.Nin[h2][requested_dim] & set(np.flatnonzero(adjacency_self[cell]))
                    self.Nout[key][requested_dim].update(requested_adj)

                for dim in (set(dims_h1.keys()) & set(dims_h2.keys()) - {requested_dim}):
                    if dim not in self.Nout[key]:
                        self.Nout[key][dim] = set()
                    for cell in self.Nout[key][requested_dim]:
                        adjacents = set(self.upper_lower_adjacency[cell].get(dim, []))
                        adjacents -= self.interface.get(key, {}).get(dim, set())
                        self.Nout[key][dim].update(adjacents)

        for h1 in heads:
            for h2 in self.agent_graph.get(h1, set()):
                key = (h1, h2)
                for dim, inter in self.interface.get(key, {}).items():
                    if dim in self.Nin.get(h1, {}):
                        self.Nin[h1][dim] -= inter
                    if dim in self.Nin.get(h2, {}):
                        self.Nin[h2][dim] -= inter
            
        for dim in self.cellularComplex:
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
                self.clustered_complexes[h1][dim] = self.cellularComplex[dim][list(lower_adjacencies), list(dim_adjacencies)]

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


                


                
                
                    

                
                
        




            






        # for cluster_id in self.Nin.keys():
        #     pass
        # for h in heads:

        #     self.Nin[i].append(h)
        #     for node, head in parent.items():
        #         if head == h:
        #             self.Nin[i].append(node)

        #     i += 1

        # self.Nin = {i : head_of[h] for h in head_of.keys()}
    

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



Node = int

@dataclass
class MaxMinDClusterResult:
    clusterheads: set[Node]
    head_of: Dict[Node, Node]     # node -> clusterhead label
    parent: Dict[Node, Optional[Node]]  # node -> parent in cluster tree (None for head/unassigned)
    depth: Dict[Node, int]        # hop distance to head (0 for head)


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


# def maxmin_d_cluster(
#     G: nx.Graph,
#     d: int,
#     score: Optional[Dict[Node, Any]] = None,
# ) -> MaxMinDClusterResult:
#     """
#     Full corrected pipeline: head election (2d max-min) + tree cluster formation (<= d hops).  [oai_citation:5‡dl.ifip.org](https://dl.ifip.org/db/conf/networking/networking2007/MazieuxMB07.pdf)
#     """
#     heads = select_clusterheads_maxmin(G, d=d, score=score)
#     head_of, parent, depth = form_clusters_tree_bfs(G, heads, d=d, score=score)
#     return MaxMinDClusterResult(clusterheads=heads, head_of=head_of, parent=parent, depth=depth)


# def clusters_are_connected(G: nx.Graph, head_of: Dict[Node, Node]) -> bool:
#     # each cluster should induce a connected subgraph (tree construction ensures it for assigned nodes)
#     by_h: Dict[Node, list[Node]] = {}
#     for v, h in head_of.items():
#         by_h.setdefault(h, []).append(v)
#     return all(nx.is_connected(G.subgraph(vs)) for vs in by_h.values() if len(vs) > 1)
       

    

    # def fit(self):
    #     """Fit the clustering model using multi-source shortest path Voronoi method."""
    #     self.clustering_results = []
    #     for d in range(self.dim + 1):
    #         laplacian = self.laplacians[d]
    #         boundary = self.boundaries[d]
    #         if self.source_simplices is not None:
    #             sources = self.source_simplices[d]
    #         else:
    #             sources = np.random.choice(laplacian.shape[0], self.num_clusters, replace=False)
    #         clustering = self._multi_source_shortest_path_voronoi(laplacian, boundary, sources)
    #         self.clustering_results.append(clustering)

    # def _multi_source_shortest_path_voronoi(self, laplacian: sp.csr_matrix, boundary: sp.csr_matrix, sources: np.ndarray) -> np.ndarray:
    #     """Perform multi-source shortest path Voronoi clustering.

    #     Args:
    #         laplacian (sp.csr_matrix): Hodge-Laplacian matrix.
    #         boundary (sp.csr_matrix): Boundary matrix.
    #         sources (np.ndarray): Indices of source simplices.
    #     """
    #     return np.zeros()
