from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from experiments.config import ExperimentConfig


DEFAULT_CAP_FRACTIONS: Sequence[float] = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class GraphSpec:
    graph_type: str
    num_vertices: int
    params: Dict[str, Any] = field(default_factory=dict)
    allow_closures: bool = True
    tag_prefix: Optional[str] = None
    seed: Optional[int] = None

    def resolved_tag_prefix(self) -> str:
        base = self.tag_prefix or f"{self.graph_type}_n{self.num_vertices}"
        return base.replace(" ", "_")


DEFAULT_GRAPH_SPECS: Sequence[GraphSpec] = (
    GraphSpec("path", 16, allow_closures=False, tag_prefix="path16"),
    GraphSpec("cycle", 16, allow_closures=False, tag_prefix="cycle16"),
    GraphSpec("erdos_renyi", 20, params={"p": 0.12}, allow_closures=True, tag_prefix="erdos20_p012", seed=21),
    GraphSpec("erdos_renyi", 20, params={"p": 0.2}, allow_closures=True, tag_prefix="erdos20_p020", seed=22),
    GraphSpec("barabasi_albert", 20, params={"m": 2}, allow_closures=True, tag_prefix="ba20_m2", seed=33),
    GraphSpec("barabasi_albert", 20, params={"m": 3}, allow_closures=True, tag_prefix="ba20_m3", seed=34),
)


def _variant_tag(prefix: str, label: str) -> str:
    return f"{prefix}_{label}"


def _copy_params(params: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    copy = dict(params)
    if seed is not None:
        copy.setdefault("seed", seed)
    return copy


def _make_config(
    spec: GraphSpec,
    label: str,
    *,
    use_closures: bool,
    term_cap: Optional[int],
    iterations: int,
    timeout: int,
    t_max: int,
    num_initial_infected: int,
) -> ExperimentConfig:
    params = _copy_params(spec.params, spec.seed)
    tag = _variant_tag(spec.resolved_tag_prefix(), label)
    return ExperimentConfig(
        graph_type=spec.graph_type,
        num_vertices=spec.num_vertices,
        graph_params=params,
        method="equations",
        iterations=iterations,
        timeout=timeout,
        t_max=t_max,
        use_closures=use_closures,
        term_length_cap=term_cap,
        solve_equations=True,
        num_initial_infected=num_initial_infected,
        tolerance=1e-2,
        seed=spec.seed,
        tag=tag,
    )


def build_truncation_experiments(
    graph_specs: Sequence[GraphSpec] = DEFAULT_GRAPH_SPECS,
    *,
    cap_fractions: Sequence[float] = DEFAULT_CAP_FRACTIONS,
    iterations: int = 3,
    timeout: int = 900,
    t_max: int = 20,
    num_initial_infected: int = 1,
) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = []
    for spec in graph_specs:
        experiments.append(
            _make_config(
                spec,
                "full",
                use_closures=False,
                term_cap=None,
                iterations=iterations,
                timeout=timeout,
                t_max=t_max,
                num_initial_infected=num_initial_infected,
            )
        )
        if spec.allow_closures:
            experiments.append(
                _make_config(
                    spec,
                    "closed",
                    use_closures=True,
                    term_cap=None,
                    iterations=iterations,
                    timeout=timeout,
                    t_max=t_max,
                    num_initial_infected=num_initial_infected,
                )
            )
        for frac in cap_fractions:
            cap = max(1, ceil(spec.num_vertices * float(frac)))
            label = f"cap_{int(round(frac * 100))}"
            experiments.append(
                _make_config(
                    spec,
                    label,
                    use_closures=False,
                    term_cap=cap,
                    iterations=iterations,
                    timeout=timeout,
                    t_max=t_max,
                    num_initial_infected=num_initial_infected,
                )
            )
    return experiments


def write_experiment_config(path: Path, experiments: Iterable[ExperimentConfig]) -> None:
    payload = {"experiments": [config.to_dict() for config in experiments]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "GraphSpec",
    "DEFAULT_GRAPH_SPECS",
    "DEFAULT_CAP_FRACTIONS",
    "build_truncation_experiments",
    "write_experiment_config",
]
