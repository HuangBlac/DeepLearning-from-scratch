'''
SGD step-size comparison experiment — reproducing Figure 5.2 / Exercise 5.34
=============================================================================

Problem: d=40 binary classification, linear predictor, hinge loss + (mu/2)||theta||^2
Data:   X ~ Gaussian, clipped so ||x_i|| <= R, y = sign(X @ w_true + noise)
Samples: n = 400
Metric: excess error of training objective  f(theta_t) - f(theta*)

Three step-size schedules:
  gamma1_t = 1/(mu*t)                  — strongly-convex step, O(1/t)
  gamma2_t = R^2 / sqrt(t)             — general-convex step, O(1/sqrt(t))
  gamma3_t = 1/(R^2*sqrt(t) + mu*t)    — combined step (Exercise 5.34)

Compared under large mu vs small mu.

Key insight (from the text):
- When mu is large: 1/(mu*t) works well (strong convexity dominates)
- When mu is small: 1/(mu*t) has huge initial step gamma_1 = 1/mu,
  causing severe oscillations. R^2/sqrt(t) is more robust initially.
- The combined step 1/(R^2*sqrt(t) + mu*t) adapts: it behaves like
  1/(R^2*sqrt(t)) early on, then like 1/(mu*t) later when mu*t >> R^2*sqrt(t).
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Global settings ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
})

SEED = 42
rng = np.random.default_rng(SEED)

# ── Data generation ─────────────────────────────────────────────
D = 40                  # feature dimension
N = 400                 # number of samples
NOISE_STD = 0.1         # label noise std

# True weight vector (sparse, 10 non-zero components)
w_true = np.zeros(D)
nonzero_idx = rng.choice(D, size=10, replace=False)
w_true[nonzero_idx] = rng.normal(0, 1, size=10)
w_true = w_true / np.linalg.norm(w_true) * 3.0    # control margin scale

# Generate features X ~ N(0, I)
X_raw = rng.normal(0, 1, size=(N, D))

# Normalise so that max ||x_i|| = R = 1 (standard scaling for clean step sizes)
norms = np.linalg.norm(X_raw, axis=1)
X = X_raw / np.max(norms)          # all ||x_i|| <= 1
R_data = np.max(np.linalg.norm(X, axis=1))
assert np.isclose(R_data, 1.0), f"Expected R=1, got {R_data}"

# Generate labels: y = sign(X @ w_true + noise)
logits = X @ w_true + NOISE_STD * rng.normal(0, 1, size=N)
y = np.sign(logits)
y = np.where(y == 0, 1, y)         # avoid sign(0) = 0

print(f"Data: d={D}, n={N}, R={R_data:.3f}")
print(f"Label balance: positive={np.mean(y == 1):.2%}")


# ── Optimal solution theta* ─────────────────────────────────────
def svm_obj(theta: np.ndarray, X: np.ndarray, y: np.ndarray, mu: float) -> float:
    """f(theta) = mean(max(0, 1 - y_i * theta^T x_i)) + (mu/2)||theta||^2"""
    margins = y * (X @ theta)
    hinge = np.maximum(0.0, 1.0 - margins)
    return float(np.mean(hinge) + 0.5 * mu * np.dot(theta, theta))


def compute_optimum(X: np.ndarray, y: np.ndarray, mu: float) -> tuple[np.ndarray, float]:
    """Find the global minimizer of the strongly-convex SVM objective via L-BFGS-B."""
    result = minimize(
        fun=lambda w: svm_obj(w, X, y, mu),
        x0=np.zeros(D),
        method='L-BFGS-B',
        options={'gtol': 1e-12, 'maxiter': 10000},
    )
    return result.x, result.fun


# ── SGD solver ──────────────────────────────────────────────────
class SGDSVM:
    """SVM trained via SGD. Supports multiple step-size schedules and records
    per-iteration excess error."""

    def __init__(self, mu: float, R: float, max_epochs: int = 200):
        self.mu = mu
        self.R = R
        self.max_epochs = max_epochs

    def fit(self, X: np.ndarray, y: np.ndarray, step_type: str,
            w_opt: np.ndarray | None = None) -> dict:
        """
        step_type: 'mu_inv' | 'R2_sqrt' | 'combined'

        Returns dict with keys: iterations, losses, excess_errors.
        Records more frequently early on to capture oscillations.
        """
        n_samples = X.shape[0]
        total_iters = self.max_epochs * n_samples
        w = np.zeros(X.shape[1])

        f_opt = svm_obj(w_opt, X, y, self.mu) if w_opt is not None else None

        # Dense recording early, sparser later (log-spaced)
        record_iters: set[int] = {1}
        record_iters.update(np.unique(np.logspace(
            0, np.log10(total_iters), num=600, dtype=int)))
        record_iters = {k for k in record_iters if 1 <= k <= total_iters}

        iterations: list[int] = []
        losses: list[float] = []
        excess_errors: list[float] = []

        idx_rng = np.random.default_rng(SEED)

        for epoch in range(self.max_epochs):
            perm = idx_rng.permutation(n_samples)

            for local_t, i in enumerate(perm):
                k = epoch * n_samples + local_t + 1    # global iteration (1-based)

                x_i, y_i = X[i], y[i]
                margin = y_i * float(np.dot(x_i, w))

                # Subgradient of hinge loss + L2 regularizer
                if margin <= 1.0:
                    grad = -y_i * x_i + self.mu * w
                else:
                    grad = self.mu * w

                # ── Step-size schedule ──
                if step_type == 'mu_inv':
                    gamma = 1.0 / (self.mu * k)
                elif step_type == 'R2_sqrt':
                    gamma = self.R ** 2 / np.sqrt(k)
                elif step_type == 'combined':
                    gamma = 1.0 / (self.R ** 2 * np.sqrt(k) + self.mu * k)
                else:
                    raise ValueError(f"Unknown step_type: {step_type}")

                w -= gamma * grad

                if k in record_iters:
                    f_val = svm_obj(w, X, y, self.mu)
                    iterations.append(k)
                    losses.append(f_val)
                    if f_opt is not None:
                        excess_errors.append(max(f_val - f_opt, 1e-16))

        return {
            'iterations': iterations,
            'losses': losses,
            'excess_errors': excess_errors,
        }


# ── Run experiment ──────────────────────────────────────────────
# mu values chosen to clearly show the two regimes:
#   large mu (1.0): strong convexity dominates from the start
#   small mu (1e-4): nearly general-convex, 1/(mu*t) step starts at gamma_1 = 10000 (!)
MU_LARGE = 1.0
MU_SMALL = 1e-6

STEP_LABELS = {
    'mu_inv':    r'$\gamma_t = 1/(\mu t)$',
    'R2_sqrt':   r'$\gamma_t = R^2/\sqrt{t}$',
    'combined':  r'$\gamma_t = 1/(R^2\sqrt{t} + \mu t)$',
}

STEP_COLORS = {
    'mu_inv':   '#2196F3',    # blue
    'R2_sqrt':  '#FF5722',    # orange
    'combined': '#4CAF50',    # green
}

all_results: dict[str, dict] = {}

for mu_label, mu_val in [('large', MU_LARGE), ('small', MU_SMALL)]:
    print(f"\n{'='*60}")
    print(f"mu = {mu_val} ({'large — strong convexity' if mu_val == MU_LARGE else 'small — nearly convex'})")
    print(f"  gamma_1 (1/(mu*t)) = {1.0/mu_val:.1f}")
    print(f"  gamma_1 (R^2/sqrt(t)) = {R_data**2:.1f}")
    print(f"{'='*60}")

    w_opt, f_opt = compute_optimum(X, y, mu_val)
    print(f"f(theta*) = {f_opt:.6f}, ||theta*|| = {np.linalg.norm(w_opt):.4f}")

    all_results[mu_label] = {}

    for step_key in STEP_LABELS:
        model = SGDSVM(mu=mu_val, R=R_data, max_epochs=200)
        res = model.fit(X, y, step_type=step_key, w_opt=w_opt)
        all_results[mu_label][step_key] = res
        final_excess = res['excess_errors'][-1] if res['excess_errors'] else float('nan')
        print(f"  {step_key:10s}: final f={res['losses'][-1]:.6f}, "
              f"excess={final_excess:.2e}")


# ── Figure 1: excess error trajectories (log-log) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

mu_titles = {
    'large': rf'$\mu = {MU_LARGE}$  —  strong convexity dominates',
    'small': rf'$\mu = {MU_SMALL}$  —  nearly general-convex',
}

for ax_idx, (mu_label, mu_val) in enumerate([('large', MU_LARGE), ('small', MU_SMALL)]):
    ax = axes[ax_idx]

    for step_key in ['mu_inv', 'R2_sqrt', 'combined']:
        res = all_results[mu_label][step_key]
        ax.loglog(
            res['iterations'], res['excess_errors'],
            linewidth=1.2, alpha=0.9,
            color=STEP_COLORS[step_key],
            label=STEP_LABELS[step_key],
        )

    ax.set_xlabel('Iteration k')
    ax.set_ylabel(r'Excess error  $f(\theta_k) - f(\theta^*)$')
    ax.set_title(mu_titles[mu_label])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Annotate the small-mu case to highlight initial instability of 1/(mu*t)
    if mu_val == MU_SMALL:
        res_mu = all_results[mu_label]['mu_inv']
        # Find the peak excess error in the first 50 recorded iterations
        early_n = min(50, len(res_mu['excess_errors']))
        peak_idx = np.argmax(res_mu['excess_errors'][:early_n])
        ax.annotate(
            rf'$\gamma_1 = 1/\mu = {1.0/mu_val:.0f}$'
            '\nHuge initial step'
            '\ncauses wild oscillation',
            xy=(res_mu['iterations'][peak_idx], res_mu['excess_errors'][peak_idx]),
            xytext=(res_mu['iterations'][peak_idx] * 5,
                    res_mu['excess_errors'][peak_idx] * 0.5),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
            fontsize=8, color='gray',
        )

fig.suptitle('SGD Step-Size Comparison  (log-log scale)', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# ── Figure 2: step-size magnitude vs iteration ──────────────────
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
k_vals = np.arange(1, 5001)

# 1/(mu*t) for both mu values
for mu_val, ls, label in [
    (MU_LARGE, '--', rf'$1/(\mu t)$,  $\mu={MU_LARGE}$'),
    (MU_SMALL, '-.', rf'$1/(\mu t)$,  $\mu={MU_SMALL}$'),
]:
    ax2.loglog(k_vals, 1.0 / (mu_val * k_vals),
               linewidth=1.5, linestyle=ls, label=label)

# R^2 / sqrt(t)
ax2.loglog(k_vals, R_data**2 / np.sqrt(k_vals),
           linewidth=1.8, color='#FF5722',
           label=rf'$R^2/\sqrt{{t}}  (R={R_data})$')

# Combined step for both mu
for mu_val, ls, label in [
    (MU_LARGE, '--', rf'combined, $\mu={MU_LARGE}$'),
    (MU_SMALL, '-.', rf'combined, $\mu={MU_SMALL}$'),
]:
    ax2.loglog(k_vals, 1.0 / (R_data**2 * np.sqrt(k_vals) + mu_val * k_vals),
               linewidth=1.0, linestyle=ls, alpha=0.6, label=label)

# Mark the crossover point for small mu: R^2*sqrt(t) = mu*t => t = (R^2/mu)^2
t_cross = int((R_data**2 / MU_SMALL) ** 2)
if t_cross <= k_vals[-1]:
    ax2.axvline(x=t_cross, color='gray', linewidth=0.6, linestyle=':')
    ax2.annotate(
        rf'$t = (R^2/\mu)^2 = {t_cross}$',
        xy=(t_cross, R_data**2 / np.sqrt(t_cross)),
        xytext=(t_cross * 2, R_data**2 / np.sqrt(t_cross) * 10),
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6),
        fontsize=7, color='gray',
    )

ax2.set_xlabel('Iteration k')
ax2.set_ylabel(r'Step size  $\gamma_k$')
ax2.set_title('Step-size schedules  (log-log scale)')
ax2.legend(fontsize=7, ncol=2, loc='lower left')
ax2.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.show()
