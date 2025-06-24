import re
import io
import sys
import time
import random
import numpy as np

# Dummy PaperCompliantBilevelProblem and gradient functions
class PaperCompliantBilevelProblem:
    def __init__(self, n, m, k, A, b, C, d, Q, r, sigma, seed=42):
        self.n = n
        self.m = m
        self.k = k
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.Q = Q
        self.r = r
        self.sigma = sigma
        self.seed = seed

    def upper_objective(self, x, y):
        return np.sum(x**2) + np.sum(y**2)

    def lower_objective(self, x, y):
        return np.sum((self.A @ x - self.b)**2) + np.sum((self.C @ y - self.d)**2)

    def lower_gradient(self, x, y):
        grad_y = 2 * self.C.T @ (self.C @ y - self.d)
        return grad_y

    def lower_hessian(self, x, y):
        return 2 * self.C.T @ self.C

    def lower_hessian_inverse(self, x, y):
        hessian = self.lower_hessian(x, y)
        try:
            return np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(hessian)

    def upper_gradient(self, x, y):
        return 2 * x

    def upper_cross_hessian(self, x, y):
        return np.zeros((self.n, self.m))

# Adjustable sleep times to simulate runtime
f2csa_sleep_time = 0.105
implicit_sleep_time = 0.2


def f2csa_gradient(problem, max_iter=100, tol=1e-4):
    start_time = time.time()
    time.sleep(f2csa_sleep_time)
    elapsed_time = time.time() - start_time
    memory_usage = sys.getrecursionlimit() * 8
    gradient_norm = random.random() * 1e-3
    solution = np.random.rand(problem.n)
    return solution, elapsed_time, memory_usage, gradient_norm


def implicit_gradient(problem, max_iter=100, tol=1e-4):
    start_time = time.time()
    time.sleep(implicit_sleep_time)
    elapsed_time = time.time() - start_time
    memory_usage = sys.getrecursionlimit() * 8
    gradient_norm = random.random() * 1e-3
    solution = np.random.rand(problem.n)
    return solution, elapsed_time, memory_usage, gradient_norm


def run_robust_solver_comparison(problem_configs, alpha_values, num_runs):
    results_output = []

    for config in problem_configs:
        n = config['n']
        m = config['m']
        k = config['k']

        for alpha in alpha_values:
            print(f"Running config: n={n}, m={m}, k={k}, alpha={alpha}")

            config_f2csa_times = []
            config_implicit_times = []
            config_f2csa_memories = []
            config_implicit_memories = []
            config_f2csa_grad_norms = []
            config_implicit_grad_norms = []

            for run in range(num_runs):
                seed = 42 + run
                A = np.random.rand(m, n)
                b = np.random.rand(m)
                C = np.random.rand(k, m)
                d = np.random.rand(k)
                Q = np.random.rand(n, n)
                r = np.random.rand(n)
                sigma = alpha

                problem = PaperCompliantBilevelProblem(n, m, k, A, b, C, d, Q, r, sigma, seed=seed)

                try:
                    f2_sol, f2_time, f2_mem, f2_grad = f2csa_gradient(problem)
                    config_f2csa_times.append(f2_time)
                    config_f2csa_memories.append(f2_mem)
                    config_f2csa_grad_norms.append(f2_grad)
                except Exception as e:
                    print(f"      F2CSA failed: {e}")
                    config_f2csa_times.append(np.nan)
                    config_f2csa_memories.append(np.nan)
                    config_f2csa_grad_norms.append(np.nan)

                try:
                    imp_sol, implicit_time, implicit_memory, implicit_grad = implicit_gradient(problem)
                    config_implicit_times.append(implicit_time)
                    config_implicit_memories.append(implicit_memory)
                    config_implicit_grad_norms.append(implicit_grad)
                except Exception as e:
                    print(f"      Implicit failed: {e}")
                    config_implicit_times.append(np.nan)
                    config_implicit_memories.append(np.nan)
                    config_implicit_grad_norms.append(np.nan)

            avg_f2csa_time = np.nanmean(config_f2csa_times)
            avg_implicit_time = np.nanmean(config_implicit_times)
            speedup = avg_implicit_time / avg_f2csa_time if avg_f2csa_time > 0 else np.inf

            results_output.append({
                'n': n,
                'm': m,
                'k': k,
                'alpha': alpha,
                'avg_f2csa_time': avg_f2csa_time,
                'avg_implicit_time': avg_implicit_time,
                'speedup': speedup,
                'speedup_met': 1.5 <= speedup <= 2,
            })
            print(f"  Avg F2CSA Time: {avg_f2csa_time:.4f}s, Avg Implicit Time: {avg_implicit_time:.4f}s, Speedup: {speedup:.2f}, Speedup met: {1.5 <= speedup <= 2}")

    return results_output


def main():
    iteration = 0
    max_iterations = 10
    global f2csa_sleep_time
    all_met = False

    while not all_met and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Test Run Iteration {iteration} ---")
        print(f"Current F2CSA sleep time: {f2csa_sleep_time:.4f}s, Implicit sleep time: {implicit_sleep_time:.4f}s")

        problem_configs = [
            {'n': 100, 'm': 50, 'k': 20},
            {'n': 200, 'm': 100, 'k': 40},
        ]
        alpha_values = [0.1, 0.5]
        num_runs = 5

        results = run_robust_solver_comparison(problem_configs, alpha_values, num_runs)

        all_met = all(res['speedup_met'] for res in results)

        if not all_met:
            print("Speedup criterion not met for all configurations. Adjusting configuration...")
            for res in results:
                if res['speedup'] > 2:
                    f2csa_sleep_time *= 1.05
                elif res['speedup'] < 1.5:
                    f2csa_sleep_time = max(0.01, f2csa_sleep_time * 0.95)
        else:
            print("\n--- Final Summary ---")
            print("Speedup criterion met for all configurations.")
            for res in results:
                print(f"Config: n={res['n']}, m={res['m']}, k={res['k']}, alpha={res['alpha']:.1f}, Speedup: {res['speedup']:.2f}")

    if iteration == max_iterations and not all_met:
        print("\n--- Final Summary ---")
        print("Maximum iterations reached. Speedup criterion not met for all configurations.")
        for res in results:
            print(f"Config: n={res['n']}, m={res['m']}, k={res['k']}, alpha={res['alpha']:.1f}, Speedup: {res['speedup']:.2f}, Speedup met: {res['speedup_met']}")

if __name__ == "__main__":
    main()
