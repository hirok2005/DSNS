from abc import ABC, abstractmethod
from typing import Union, Optional, overload
import ssspx
from dsns.helpers import SatID
from dsns.multiconstellation import MultiConstellation


class GraphSolver(ABC):
    graph: ssspx.Graph
    cache: dict[SatID, ssspx.SSSPResult] = {}
    

    def __init__(self) -> None:
        super().__init__()

    def update(self, mobility: MultiConstellation) -> None:
        pass

    def update(self, n: int, costs: dict[tuple[SatID, SatID], float]) -> None:
        pass

    def update(
        self,
        data: Union[MultiConstellation, int],
        costs: Optional[dict[tuple[SatID, SatID], float]] = None,
    ) -> None:
        self.cache.clear()
        if isinstance(data, MultiConstellation):
            mobility = data
            n = len(mobility.satellites)
            self.graph = ssspx.Graph(n)
            for u, v in mobility.links:
                self.graph.add_edge(u, v, mobility.get_delay(u, v))
        elif isinstance(data, int):
            if costs is None:
                raise ValueError("If 'n' is provided, 'costs' must also be provided.")
            n = data
            self.graph = ssspx.Graph(n)
            for (u, v), c in costs.items():
                self.graph.add_edge(u, v, c)
                self.graph.add_edge(v, u, c)
        else:
            raise TypeError(f"Unexpected type for data: {type(data)}")

    @abstractmethod
    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        pass

    @abstractmethod
    def get_path(self, source: SatID, destination: SatID) -> list[SatID]:
        pass

    # FOR BENCHMARKING ONLY, BYPASSES CACHE
    @abstractmethod
    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        pass

    def remove_edges(self, edges: set[tuple[SatID, SatID]]) -> None:
        edges_by_u = {}
        for u, v in edges:
            edges_by_u.setdefault(u, set()).add(v)
            edges_by_u.setdefault(v, set()).add(u)

        for u, targets in edges_by_u.items():
            self.graph.adj[u] = [
                (v, w) for v, w in self.graph.adj[u] if v not in targets
            ]


class BmsspSolver(GraphSolver):
    config = ssspx.SolverConfig(use_transform=False)

    def __init__(self) -> None:
        super().__init__()

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = self.cache.get(source, None) 
        if not res:
            solver = ssspx.SSSPSolver(self.graph, source, BmsspSolver.config)
            res = solver.solve()
            self.cache[source] = res
        return res.distances[destination]

    def get_path(self, source: SatID, destination: SatID) -> list[SatID]:
        res = self.cache.get(source, None) 
        if not res:
            solver = ssspx.SSSPSolver(self.graph, source, BmsspSolver.config)
            res = solver.solve()
            self.cache[source] = res
        return ssspx.reconstruct_path_basic(res.predecessors, source, destination)

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        solver = ssspx.SSSPSolver(self.graph, source, BmsspSolver.config)
        res = solver.solve()
        return res.distances[destination]



class DijkstraSolver(GraphSolver):
    def __init__(self) -> None:
        super().__init__()

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = self.cache.get(source, None) 
        if not res:
            res = ssspx.dijkstra_reference(self.graph, [source])
            self.cache[source] = res
        return res.distances[destination]

    def get_path(self, source: SatID, destination: SatID) -> list[SatID]:
        res = self.cache.get(source, None) 
        if not res:
            res = ssspx.dijkstra_reference(self.graph, [source])
            self.cache[source] = res
        return ssspx.reconstruct_path_basic(res.predecessors, source, destination)

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        res = ssspx.dijkstra_reference(self.graph, [source])
        return res.distances[destination]

