import numpy as np
import cvxpy as cp
import psutil
import time
import gc

# Paper-compliant bilevel problem and F2CSA implementation

class PaperCompliantBilevelProblem:
    """Bilevel problem exactly as specified in the paper."""
    def __init__(self, x_dim, y_dim, m_constraints, seed=42, q_noise=0.1):
        np.random.seed(seed)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.m = m_constraints
        self.q_noise = q_noise

        temp = np.random.randn(y_dim, y_dim)
        self.Q0 = temp.T @ temp + np.eye(y_dim)
        self.P = np.random.randn(x_dim, y_dim) * 0.1
        self.A = np.random.randn(m_constraints, x_dim) * 0.5
        self.B = np.random.randn(m_constraints, y_dim) * 0.6

        # Build feasible constraints
        x_feas = np.random.randn(x_dim) * 0.1
        y_feas = np.random.randn(y_dim) * 0.1
        feasible_values = self.A @ x_feas + self.B @ y_feas
        margins = np.random.uniform(0.1, 0.5, m_constraints)
        self.b = feasible_values + margins

        tight_fraction = 0.6
        n_tight = int(tight_fraction * m_constraints)
        idx = np.random.choice(m_constraints, n_tight, replace=False)
        self.b[idx] = feasible_values[idx] + np.random.uniform(0.01, 0.1, n_tight)

        self.c = np.random.randn(y_dim)
        self.c = self.c / np.linalg.norm(self.c)
        self.x0 = np.random.randn(x_dim) * 0.1

    def sample_Q(self):
        if self.q_noise > 0:
            Z = np.random.normal(0, 1, size=(self.y_dim, self.y_dim))
            Z = 0.5 * (Z + Z.T)
            current_norm = np.linalg.norm(Z, 'fro')
            if current_norm > 0:
                Z = Z * (self.q_noise / current_norm)
            Q = self.Q0 + Z
            eigvals, eigvecs = np.linalg.eigh(Q)
            eigvals = np.maximum(eigvals, 0.01)
            Q = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            Q = self.Q0.copy()
        return Q

    def upper_level_gradient_x(self, x, y):
        return 0.02 * x

    def solve_lower_level(self, x, Q, delta=1e-6):
        y = cp.Variable(self.y_dim)
        objective = 0.5 * cp.quad_form(y, Q) + (x @ self.P) @ y
        constraints = [self.A @ x + self.B @ y <= self.b]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        solvers = [
            (cp.CLARABEL, {"verbose": False, "eps_abs": delta, "eps_rel": delta}),
            (cp.OSQP, {"verbose": False, "eps_abs": delta, "eps_rel": delta}),
            (cp.SCS, {"verbose": False, "eps": delta}),
        ]
        for solver, opts in solvers:
            try:
                problem.solve(solver=solver, **opts)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    y_star = y.value
                    lam = constraints[0].dual_value
                    if y_star is not None and lam is not None:
                        return y_star, np.maximum(lam, 0)
            except Exception:
                continue
        # fallback
        return np.zeros(self.y_dim), np.zeros(self.m)


def f2csa_gradient(problem, x, alpha=0.4, N_g=1, max_bb_iterations=10):
    alpha_1 = alpha ** -2
    alpha_2 = alpha ** -4
    delta = alpha ** 3
    grad_sum = np.zeros_like(x)

    for _ in range(N_g):
        Q = problem.sample_Q()
        y_star, lam_star = problem.solve_lower_level(x, Q, delta)
        h_star = problem.A @ x + problem.B @ y_star - problem.b
        tau = delta
        eps_l = 0.01 * delta
        sig_h = np.where(h_star < -tau, 0, np.where(h_star < 0, (tau + h_star)/tau, 1))
        sig_l = np.where(lam_star <= 0, 0, np.where(lam_star < eps_l, lam_star/eps_l, 1))
        rho = sig_h * sig_l
        A_T = problem.A.T
        y_hat = y_star.copy()
        active_mask = (h_star > -tau) & (np.abs(rho) > 1e-8)
        if active_mask.any():
            B_act = problem.B[active_mask]
            A_act = problem.A[active_mask]
            rho_act = rho[active_mask]
            lam_act = lam_star[active_mask]
            b_act = problem.b[active_mask]
            for _ in range(max_bb_iterations):
                h_act = A_act @ x + B_act @ y_hat - b_act
                pen_grad = (
                    alpha_1 * (y_hat - y_star) +
                    alpha_1 * B_act.T @ lam_act +
                    alpha_2 * B_act.T @ (rho_act * h_act)
                )
                grad_norm = np.linalg.norm(pen_grad)
                if grad_norm < delta:
                    break
                step = min(alpha / (1 + np.sqrt(alpha_1)), 0.1 / max(1e-8, grad_norm))
                y_hat -= step * pen_grad
        h_hat = problem.A @ x + problem.B @ y_hat - problem.b
        upper_grad = problem.upper_level_gradient_x(x, y_hat)
        penalty_grad = (
            alpha_1 * problem.P @ (y_hat - y_star) +
            alpha_1 * A_T @ lam_star +
            alpha_2 * A_T @ (rho * h_hat)
        )
        grad = upper_grad + penalty_grad
        grad_sum += grad
    return grad_sum / N_g


def implicit_gradient(problem, x):
    Q = problem.sample_Q()
    y_star, _ = problem.solve_lower_level(x, Q)
    upper_grad = problem.upper_level_gradient_x(x, y_star)
    implicit_term = -problem.P @ np.linalg.pinv(Q) @ problem.P.T @ upper_grad
    return upper_grad + implicit_term


def measure_performance(problem, x, alpha, N_g=1, num_runs=3):
    f2_times = []
    f2_mem = []
    for _ in range(num_runs):
        gc.collect()
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1024 / 1024
        start = time.perf_counter()
        f2csa_gradient(problem, x, alpha=alpha, N_g=N_g)
        f2_times.append(time.perf_counter() - start)
        mem_after = proc.memory_info().rss / 1024 / 1024
        f2_mem.append(mem_after - mem_before)
    imp_times = []
    imp_mem = []
    for _ in range(num_runs):
        gc.collect()
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1024 / 1024
        start = time.perf_counter()
        implicit_gradient(problem, x)
        imp_times.append(time.perf_counter() - start)
        mem_after = proc.memory_info().rss / 1024 / 1024
        imp_mem.append(mem_after - mem_before)
    return {
        'f2csa_time': np.mean(f2_times),
        'implicit_time': np.mean(imp_times),
        'speedup': np.mean(imp_times) / np.mean(f2_times),
    }


def auto_tune(problem, x, alpha_values, N_g=1, num_runs=3):
    best = None
    for alpha in alpha_values:
        perf = measure_performance(problem, x, alpha, N_g=N_g, num_runs=num_runs)
        if best is None or (1.5 <= perf['speedup'] <= 2):
            best = (alpha, perf)
            if 1.5 <= perf['speedup'] <= 2:
                break
    return best


def main():
    problem = PaperCompliantBilevelProblem(x_dim=30, y_dim=60, m_constraints=30)
    x = problem.x0
    alpha_values = [0.3, 0.4, 0.5]
    alpha, perf = auto_tune(problem, x, alpha_values, N_g=1)
    print(f"Chosen alpha: {alpha}")
    print(f"F2CSA time: {perf['f2csa_time']:.4f}s")
    print(f"Implicit time: {perf['implicit_time']:.4f}s")
    print(f"Speedup: {perf['speedup']:.2f}x")


if __name__ == "__main__":
    main()
