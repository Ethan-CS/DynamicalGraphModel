"""Tests for the truncation-focused experiment runner."""

from model_params.cmodel import get_SIR

from experiments import fatanode_runner as runner


def test_compute_caps_rounds_and_deduplicates():
    caps = runner.compute_caps(4, [0.2, 0.26, 0.5])
    assert caps == [1, 2], f"Expected rounded caps [1, 2], got {caps}"


def test_variant_plan_includes_full_closed_and_truncations():
    configs = runner.variant_plan(4, (0.25, 0.5, 0.75))
    kinds = [cfg.kind for cfg in configs]
    assert kinds.count("full") == 1
    assert kinds.count("closed") == 1
    trunc_caps = sorted(cfg.cap for cfg in configs if cfg.kind == "truncated")
    assert trunc_caps == [1, 2, 3]


def test_run_for_graph_solves_and_computes_errors():
    spec = runner.GRAPH_SPECS["lollipop"]
    model = get_SIR()
    results = runner.run_for_graph(
        spec=spec,
        cap_fractions=(0.25, 0.5),
        model=model,
        num_initial_infected=1,
        t_max=2.0,
    )
    assert len(results) == 4

    full = next(r for r in results if r.config.kind == "full")
    closed = next(r for r in results if r.config.kind == "closed")
    assert full.solve_success, "Full system should solve"
    assert closed.solve_success, "Closed system should solve"

    for res in results:
        assert res.equation_count > 0
        if res.config.kind == "truncated":
            assert res.solve_success, "Truncated systems should solve"
            assert res.mean_abs_error_full is not None
            assert res.max_abs_error_full is not None
            # Closed comparison only defined when the baseline exists
            assert res.mean_abs_error_closed is not None
            assert res.max_abs_error_closed is not None
        elif res.config.kind == "closed":
            # Closed variant should not report closed-vs-closed error
            assert res.mean_abs_error_closed is None
            assert res.max_abs_error_closed is None
