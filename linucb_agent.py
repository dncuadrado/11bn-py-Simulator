import numpy as np


# ============================================================
#   LINUCB AGENT (masked selection)
# ============================================================
class LinUCBAgent:
    def __init__(self, n_arms, context_dim, alpha=1.0, mab_coeff=0.4, mab_penalty_weight=0.5):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.mab_coeff = mab_coeff
        self.mab_penalty_weight = mab_penalty_weight
        self.A = [np.identity(context_dim) for _ in range(n_arms)]   # design matrix
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_arms)]  # reward-weighted context sum

    def select_arm(self, context, candidate_indices: np.ndarray):
        """
        Select an arm among candidate_indices using LinUCB.
        candidate_indices: 1D array of ints (valid arm indices).
        """
        if len(candidate_indices) == 0:
            raise RuntimeError("No candidate arms provided to select_arm.")

        context = context.reshape(-1, 1)
        D = len(candidate_indices)
        ucb_values = np.full(D, -np.inf, dtype=float)

        for i, a in enumerate(candidate_indices):
            A_inv = np.linalg.inv(self.A[a]) # inverse covariance (uncertainty) matrix for arm a
            theta = A_inv @ self.b[a]        # estimated parameter vector for arm a
            # extract scalars explicitly to avoid numpy deprecation
            mean_est = float((theta.T @ context).item())   
            conf = self.alpha * np.sqrt((context.T @ A_inv @ context).item())
            ucb_values[i] = mean_est + conf

        # pick the best candidate with random tie-breaking
        max_val = np.max(ucb_values)
        best_local = np.flatnonzero(np.isclose(ucb_values, max_val))
        chosen_local = int(np.random.choice(best_local))
        chosen_arm = int(candidate_indices[chosen_local])
        return chosen_arm

    def update(self, arm, reward, context):
        context = context.reshape(-1, 1)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context
