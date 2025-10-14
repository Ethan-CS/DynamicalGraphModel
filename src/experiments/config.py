from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single runtime measurement experiment."""

    graph_type: str
    num_vertices: int = 100
    graph_params: Dict[str, Any] = field(default_factory=dict)
    method: str = "both"
    iterations: int = 5
    timeout: int = 600
    t_max: int = 15
    use_closures: bool = True
    solve_equations: bool = False
    num_initial_infected: int = 1
    tolerance: float = 1e-2
    target_average: Optional[Dict[str, float]] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "graph_type": self.graph_type,
            "num_vertices": self.num_vertices,
            "graph_params": dict(self.graph_params),
            "method": self.method,
            "iterations": self.iterations,
            "timeout": self.timeout,
            "t_max": self.t_max,
            "use_closures": self.use_closures,
            "solve_equations": self.solve_equations,
            "num_initial_infected": self.num_initial_infected,
            "tolerance": self.tolerance,
            "target_average": self.target_average,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(raw)

        graph_type = data.get("graph_type")
        if graph_type is None:
            raise KeyError("'graph_type' must be provided in experiment config")

        graph_params = data.get("graph_params")
        if graph_params is None:
            graph_params = {}
        elif not isinstance(graph_params, dict):
            raise TypeError("'graph_params' must be a dictionary when provided")

        target_average = data.get("target_average")
        if target_average is not None:
            if not isinstance(target_average, dict):
                raise TypeError("'target_average' must be a dictionary of floats when provided")
            target_average = {str(key): float(value) for key, value in target_average.items()}

        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)

        return cls(
            graph_type=str(graph_type),
            num_vertices=int(data.get("num_vertices", 100)),
            graph_params=dict(graph_params),
            method=str(data.get("method", "both")),
            iterations=int(data.get("iterations", 5)),
            timeout=int(data.get("timeout", 600)),
            t_max=int(data.get("t_max", 15)),
            use_closures=bool(data.get("use_closures", True)),
            solve_equations=bool(data.get("solve_equations", False)),
            num_initial_infected=int(data.get("num_initial_infected", 1)),
            tolerance=float(data.get("tolerance", 1e-2)),
            target_average=target_average,
            seed=seed,
        )

