from abc import ABC, abstractmethod, overload
from typing import Union, Optional
import ssspx
from dsns.helpers import SatID
from dsns.multiconstellation import MultiConstellation


class GraphSolver(ABC):
    def __init__(self) -> None:
        super().__init__()

    @overload
    def update(self, mobility: MultiConstellation) -> None:
        pass

    @overload
    def update(self, n: int, costs: dict[tuple[SatID, SatID], float]) -> None:
        pass

    def update(self, data: Union[MultiConstellation, int], costs: Optional[dict[tuple[SatID, SatID], float]] = None) -> None:        
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


class BmsspSolver(GraphSolver):
    config = ssspx.SolverConfig(use_transform=False)

    def __init__(self) -> None:
        super().__init__()

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        solver = ssspx.SSSPSolver(self.graph, source, BmsspSolver.config)
        res = solver.solve()
        return res.distances[destination]

    def get_path(self, source: SatID, destination: SatID) -> list[SatID]:
        solver = ssspx.SSSPSolver(self.graph, source, BmsspSolver.config)
        _ = solver.solve()
        return solver.path(destination)


class DijkstraSolver(GraphSolver):
    def __init__(self) -> None:
        super().__init__()

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = ssspx.dijkstra_reference(self.graph, [source])
        return res.distances[destination]

    def get_path(self, source: SatID, destination: SatID) -> list[SatID]:
        res = ssspx.dijkstra_reference(self.graph, [source])
        return ssspx.reconstruct_path_basic(res.predecessors, source, destination)
