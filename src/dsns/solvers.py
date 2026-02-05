from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List, Set, Tuple
from collections import defaultdict

from dsns.helpers import SatID
from dsns.multiconstellation import MultiConstellation
from dsns.bmssp_solver import BmsspSolver as LibBmsspSolver
# from dsns.bmssp_solver import BmsspSolverV2 as LibBmsspSolverV2
from dsns.comparison_solvers import dijkstra_sssp, dijkstra
from dsns.graph import Graph as LibGraph


class GraphSolver(ABC):
    # Graph is an adjacency dict: u -> {v -> weight}
    graph: Dict[SatID, Dict[SatID, float]]
    # Optional instance of the library's Graph object (lazily built)
    lib_graph: Optional[LibGraph]
    # Cache stores distance results: source -> distances
    cache: Dict[SatID, Tuple[List[float], List[Optional[int]]]]
    n: int

    def __init__(self) -> None:
        super().__init__()
        self.graph = defaultdict(dict)
        self.lib_graph = None
        self.cache = {}
        self.n = 0

    def update(
        self,
        data: Union[MultiConstellation, int],
        costs: Optional[Dict[Tuple[SatID, SatID], float]] = None,
    ) -> None:
        self.cache.clear()
        self.graph = defaultdict(dict)
        self.lib_graph = None

        if isinstance(data, MultiConstellation):
            mobility = data
            self.n = len(mobility.satellites)
            for u, v in mobility.links:
                w_uv = mobility.get_delay(u, v)
                w_vu = mobility.get_delay(v, u)
                self.graph[u][v] = w_uv
                self.graph[v][u] = w_vu

        elif isinstance(data, int):
            if costs is None:
                raise ValueError("If 'n' is provided, 'costs' must also be provided.")
            self.n = data
            for (u, v), c in costs.items():
                self.graph[u][v] = c
                self.graph[v][u] = c
        else:
            raise TypeError(f"Unexpected type for data: {type(data)}")

    def _ensure_lib_graph(self) -> None:
        if self.lib_graph is None:
            self.lib_graph = LibGraph(self.n)
            for u, neighbors in self.graph.items():
                for v, w in neighbors.items():
                    if 0 <= u < self.n and 0 <= v < self.n:
                        self.lib_graph.add_edge(u, v, w)

    @abstractmethod
    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        pass

    @abstractmethod
    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        pass

    # FOR BENCHMARKING ONLY, BYPASSES CACHE
    @abstractmethod
    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        pass

    def remove_edges(self, edges: Set[Tuple[SatID, SatID]]) -> None:
        for u, v in edges:
            if u in self.graph:
                self.graph[u].pop(v, None)
            if v in self.graph:
                self.graph[v].pop(u, None)
        self.lib_graph = None
        self.cache.clear()

    def _reconstruct_path(self, predecessors: List[Optional[int]], source: int, destination: int) -> List[int]:
        """Helper to reconstruct path from predecessor array."""
        if not (0 <= destination < len(predecessors)):
            return []
            
        path = []
        curr = destination

        while curr is not None:
            path.append(curr)
            if curr == source:
                break

            curr = predecessors[curr]

        if not path or path[-1] != source:
            return []

        return path[::-1]


class BmsspSolver(GraphSolver):
    def __init__(self) -> None:
        super().__init__()


    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            solver = LibBmsspSolver(self.lib_graph)
            # Returns (distances, predecessors)
            res = solver.solve_sssp(source)
            self.cache[source] = res

        distances, _ = res
        if 0 <= destination < len(distances):
            val = distances[destination]
            return val
        return float("inf")

    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            solver = LibBmsspSolver(self.lib_graph)
            res = solver.solve_sssp(source)
            self.cache[source] = res
            
        distances, predecessors = res
        
        if 0 <= destination < len(distances):
            if distances[destination] == float('inf'):
                return []
        else:
            return []

        return self._reconstruct_path(predecessors, source, destination)

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        self._ensure_lib_graph()
        solver = LibBmsspSolver(self.lib_graph)
        result = solver.solve(source, destination)
        if result:
            return result[0]
        return float("inf")


class DijkstraSolver(GraphSolver):
    def __init__(self) -> None:
        super().__init__()

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            res = dijkstra_sssp(self.lib_graph, source)
            self.cache[source] = res

        distances, _ = res
        if 0 <= destination < len(distances):
            val = distances[destination]
            return val
        return float("inf")

    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            res = dijkstra_sssp(self.lib_graph, source)
            self.cache[source] = res
            
        distances, predecessors = res
        
        if 0 <= destination < len(distances):
            if distances[destination] == float('inf'):
                return []
        else:
            return []

        return self._reconstruct_path(predecessors, source, destination)

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        self._ensure_lib_graph()
        result = dijkstra(self.lib_graph, source, destination)
        if result:
            return result[0]
        return float("inf")

