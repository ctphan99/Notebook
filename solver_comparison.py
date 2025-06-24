"""Compare Algorithm 1 (F2CSA) with an implicit gradient baseline.

This script runs small bilevel problems and measures the execution time
of both the F2CSA hypergradient estimator and an implicit differentiation
approach. It automatically searches over a few ``alpha`` values and reports
which configuration yields the largest speedup.
"""

import numpy as np
from paper_f2csa import (
    PaperCompliantBilevelProblem,
    measure_performance,
)


def run_comparison():
    # Problem sizes to test
    configs = [
        {"x_dim": 30, "y_dim": 60, "m": 30},
        {"x_dim": 40, "y_dim": 80, "m": 40},
    ]
    alpha_values = [0.3, 0.4, 0.5, 0.6]

    for cfg in configs:
        print(f"\nProblem: x={cfg['x_dim']} y={cfg['y_dim']} m={cfg['m']}")
        problem = PaperCompliantBilevelProblem(
            x_dim=cfg["x_dim"], y_dim=cfg["y_dim"], m_constraints=cfg["m"]
        )
        x = problem.x0

        best_perf = None
        for alpha in alpha_values:
            perf = measure_performance(problem, x, alpha, N_g=1, num_runs=3)
            print(
                f"  alpha={alpha}: F2CSA={perf['f2csa_time']:.4f}s, "
                f"Implicit={perf['implicit_time']:.4f}s, speedup={perf['speedup']:.2f}x"
            )
            if best_perf is None or perf["speedup"] > best_perf["speedup"]:
                best_perf = {"alpha": alpha, **perf}

        print(
            f"  -> best alpha {best_perf['alpha']} with speedup {best_perf['speedup']:.2f}x"
        )


if __name__ == "__main__":
    run_comparison()
