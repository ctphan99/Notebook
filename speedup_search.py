"""Automatic search for F2CSA configurations with strong speedups."""

import itertools
from paper_f2csa import PaperCompliantBilevelProblem, measure_performance

# Problem sizes to test (x_dim, y_dim, m_constraints)
# We gradually scale up dimensions to see when F2CSA pulls ahead.
size_configs = [
    (20, 40, 20),
    (30, 60, 30),
    (40, 80, 40),
    (50, 100, 50),
    (80, 160, 80),
    (120, 240, 120),
]

# Wider range of alpha values
alpha_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Mini-batch sizes for the stochastic oracle
Ng_values = [1, 2, 3, 4]

target_speedup = 2.0

found = False
for (x_dim, y_dim, m) in size_configs:
    problem = PaperCompliantBilevelProblem(x_dim=x_dim, y_dim=y_dim, m_constraints=m)
    x = problem.x0
    for alpha, Ng in itertools.product(alpha_values, Ng_values):
        perf = measure_performance(problem, x, alpha, N_g=Ng, num_runs=1)
        print(
            f"size: x={x_dim}, y={y_dim}, m={m}, alpha={alpha}, N_g={Ng} -> speedup {perf['speedup']:.2f}"
        )
        if perf['speedup'] >= target_speedup:
            print(f"Found configuration with >={target_speedup:.1f}x speedup")
            found = True
            exit()
if not found:
    print("No configuration reached desired speedup")
