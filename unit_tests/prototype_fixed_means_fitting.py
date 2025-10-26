#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype: Fit GMM covariances with fixed means.

Tests several approaches to find best method before implementing in production.

Created: 2025-10-22
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix


def generate_synthetic_2dye_data(
    true_means,
    true_covariances,
    true_weights,
    n_samples=1000,
    random_state=42
):
    """Generate synthetic data from known GMM parameters."""
    np.random.seed(random_state)

    # Sample component assignments
    components = np.random.choice(
        len(true_means),
        size=n_samples,
        p=true_weights
    )

    # Generate samples
    X = np.zeros((n_samples, true_means.shape[1]))
    for i, comp in enumerate(components):
        X[i] = np.random.multivariate_normal(
            true_means[comp],
            true_covariances[comp]
        )

    return X, components


def method_hard_assignment(X, fixed_means):
    """
    Method A: Hard assignment to nearest mean, then direct covariance estimation.

    Pros: Simple, fast, stable
    Cons: Ignores uncertainty near boundaries
    """
    # Assign each point to nearest mean (Voronoi partition)
    distances = cdist(X, fixed_means, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    # Compute covariance for each component
    n_components = len(fixed_means)
    covariances = []
    weights = []

    for k in range(n_components):
        X_k = X[labels == k]

        if len(X_k) < 2:
            # Not enough points - use identity
            covariances.append(np.eye(X.shape[1]))
            weights.append(0.0)
            print(f"    Warning: Component {k} has <2 points, using identity covariance")
        else:
            # Center around fixed mean and compute covariance
            centered = X_k - fixed_means[k]
            cov_k = np.cov(centered.T, bias=False)

            # Ensure positive definite (add small regularization if needed)
            min_eig = np.linalg.eigvalsh(cov_k)[0]
            if min_eig < 1e-6:
                cov_k += (1e-6 - min_eig) * np.eye(cov_k.shape[0])

            covariances.append(cov_k)
            weights.append(len(X_k) / len(X))

    return np.array(covariances), np.array(weights)


def method_soft_em(X, fixed_means, max_iter=100, tol=1e-6):
    """
    Method B: Custom EM algorithm with fixed means.

    E-step: Calculate responsibilities using current parameters
    M-step: Update covariances and weights (NOT means)

    Pros: Probabilistic, considers uncertainty
    Cons: More complex, needs convergence monitoring
    """
    n_samples, n_features = X.shape
    n_components = len(fixed_means)

    # Initialize
    covariances = [np.eye(n_features) for _ in range(n_components)]
    weights = np.ones(n_components) / n_components

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        # E-step: Calculate responsibilities
        log_probs = np.zeros((n_samples, n_components))

        for k in range(n_components):
            try:
                mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])
            except np.linalg.LinAlgError:
                # Singular covariance - add regularization
                cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                mvn = multivariate_normal(mean=fixed_means[k], cov=cov_reg)
                log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

        # Normalize to get responsibilities
        log_probs_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        responsibilities = probs / probs.sum(axis=1, keepdims=True)

        # Calculate log-likelihood
        log_likelihood = np.sum(np.log(probs.sum(axis=1)))

        # Check convergence
        if abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

        # M-step: Update weights and covariances (NOT means - they're fixed!)
        weights = responsibilities.mean(axis=0)

        for k in range(n_components):
            # Weighted covariance around fixed mean
            centered = X - fixed_means[k]
            weighted = responsibilities[:, k:k+1] * centered

            cov_k = (weighted.T @ centered) / responsibilities[:, k].sum()

            # Regularize to ensure positive definite
            min_eig = np.linalg.eigvalsh(cov_k)[0]
            if min_eig < 1e-6:
                cov_k += (1e-6 - min_eig) * np.eye(n_features)

            covariances[k] = cov_k

    return np.array(covariances), weights


def method_mle_scipy(X, fixed_means):
    """
    Method C: Maximum Likelihood Estimation using scipy.optimize.

    Parameterizes covariance matrices and uses numerical optimization to maximize
    likelihood with fixed means.

    Pros: Statistically principled, standard MLE approach
    Cons: More complex parameterization, slower optimization
    """
    n_samples, n_features = X.shape
    n_components = len(fixed_means)

    # Parameterize: [cov00_comp0, cov01_comp0, cov11_comp0, ..., weight_0, ...]
    # For 2D full covariance: 3 params per component (cov00, cov01, cov11)
    # Plus (n_components - 1) weight params (last weight is 1 - sum(others))

    def pack_params(covariances, weights):
        """Pack covariances and weights into flat parameter vector."""
        params = []
        for k in range(n_components):
            cov = covariances[k]
            # Store as [var_0, cov_01, var_1]
            params.extend([cov[0, 0], cov[0, 1], cov[1, 1]])
        # Store n_components-1 weights (last is 1 - sum)
        params.extend(weights[:-1])
        return np.array(params)

    def unpack_params(params):
        """Unpack flat parameter vector into covariances and weights."""
        covariances = []
        idx = 0
        for k in range(n_components):
            var_0 = params[idx]
            cov_01 = params[idx + 1]
            var_1 = params[idx + 2]
            cov = np.array([[var_0, cov_01], [cov_01, var_1]])
            covariances.append(cov)
            idx += 3

        # Reconstruct weights
        weights = np.zeros(n_components)
        weights[:-1] = params[idx:]
        weights[-1] = 1.0 - weights[:-1].sum()

        return covariances, weights

    def negative_log_likelihood(params):
        """Negative log-likelihood for minimization."""
        try:
            covariances, weights = unpack_params(params)

            # Check validity
            for k in range(n_components):
                # Check positive definite
                eigvals = np.linalg.eigvalsh(covariances[k])
                if np.any(eigvals <= 0):
                    return 1e10  # Invalid

            # Check weights valid
            if np.any(weights <= 0) or np.any(weights >= 1):
                return 1e10

            # Calculate log-likelihood
            log_probs = np.zeros((n_samples, n_components))
            for k in range(n_components):
                mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

            # Log-sum-exp trick for numerical stability
            log_probs_max = log_probs.max(axis=1, keepdims=True)
            log_likelihood = np.sum(log_probs_max + np.log(np.exp(log_probs - log_probs_max).sum(axis=1)))

            return -log_likelihood

        except (np.linalg.LinAlgError, ValueError):
            return 1e10

    # Initialize with hard assignment estimates
    cov_init, weights_init = method_hard_assignment(X, fixed_means)
    params_init = pack_params(cov_init, weights_init)

    # Optimize
    result = minimize(
        negative_log_likelihood,
        params_init,
        method='Nelder-Mead',  # Doesn't require gradients
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
    )

    covariances, weights = unpack_params(result.x)

    # Ensure positive definite (regularize if needed)
    for k in range(n_components):
        min_eig = np.linalg.eigvalsh(covariances[k])[0]
        if min_eig < 1e-6:
            covariances[k] += (1e-6 - min_eig) * np.eye(n_features)

    return np.array(covariances), weights


def compare_methods():
    """Compare hard assignment vs soft EM for fixed-means covariance fitting."""

    print("=" * 60)
    print("PROTOTYPE: Fixed-Means Covariance Fitting")
    print("=" * 60)

    # Ground truth parameters
    true_means = np.array([[0.7, 0.2], [0.2, 0.7]])
    true_covariances = np.array([
        [[0.01, 0.003], [0.003, 0.01]],
        [[0.015, 0.005], [0.005, 0.012]]
    ])
    true_weights = np.array([0.6, 0.4])

    print("\nGround Truth Parameters:")
    print(f"  Means:\n{true_means}")
    print(f"  Covariances[0]:\n{true_covariances[0]}")
    print(f"  Covariances[1]:\n{true_covariances[1]}")
    print(f"  Weights: {true_weights}")

    # Generate synthetic data
    n_samples = 1000
    X, true_labels = generate_synthetic_2dye_data(
        true_means, true_covariances, true_weights, n_samples
    )

    print(f"\nGenerated {n_samples} synthetic samples")

    # Test Method A: Hard Assignment
    print("\n" + "-" * 60)
    print("Method A: Hard Assignment")
    print("-" * 60)

    cov_hard, weights_hard = method_hard_assignment(X, true_means)

    print("Fitted covariances (hard assignment):")
    for k in range(len(true_means)):
        print(f"  Component {k}:")
        print(f"    True cov:\n{true_covariances[k]}")
        print(f"    Fitted cov:\n{cov_hard[k]}")
        print(f"    Frobenius error: {np.linalg.norm(cov_hard[k] - true_covariances[k]):.6f}")
        print(f"    Weight: true={true_weights[k]:.3f}, fitted={weights_hard[k]:.3f}")

    # Test Method B: Soft EM
    print("\n" + "-" * 60)
    print("Method B: Soft EM")
    print("-" * 60)

    cov_em, weights_em = method_soft_em(X, true_means, max_iter=100)

    print("Fitted covariances (soft EM):")
    for k in range(len(true_means)):
        print(f"  Component {k}:")
        print(f"    True cov:\n{true_covariances[k]}")
        print(f"    Fitted cov:\n{cov_em[k]}")
        print(f"    Frobenius error: {np.linalg.norm(cov_em[k] - true_covariances[k]):.6f}")
        print(f"    Weight: true={true_weights[k]:.3f}, fitted={weights_em[k]:.3f}")

    # Test Method C: MLE with scipy.optimize
    print("\n" + "-" * 60)
    print("Method C: MLE (scipy.optimize)")
    print("-" * 60)

    cov_mle, weights_mle = method_mle_scipy(X, true_means)

    print("Fitted covariances (MLE):")
    for k in range(len(true_means)):
        print(f"  Component {k}:")
        print(f"    True cov:\n{true_covariances[k]}")
        print(f"    Fitted cov:\n{cov_mle[k]}")
        print(f"    Frobenius error: {np.linalg.norm(cov_mle[k] - true_covariances[k]):.6f}")
        print(f"    Weight: true={true_weights[k]:.3f}, fitted={weights_mle[k]:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Data with true assignments
    ax = axes[0]
    for k in range(len(true_means)):
        mask = true_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=20, label=f'Component {k}')
    ax.scatter(true_means[:, 0], true_means[:, 1], c='black', s=200, marker='X',
               edgecolors='white', linewidths=2, label='True means', zorder=10)
    ax.set_xlabel('A_R')
    ax.set_ylabel('A_G')
    ax.set_title('True Component Assignments')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Hard assignment
    ax = axes[1]
    distances = cdist(X, true_means, metric='euclidean')
    labels_hard = np.argmin(distances, axis=1)
    for k in range(len(true_means)):
        mask = labels_hard == k
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=20, label=f'Component {k}')
    ax.scatter(true_means[:, 0], true_means[:, 1], c='black', s=200, marker='X',
               edgecolors='white', linewidths=2, label='Fixed means', zorder=10)
    ax.set_xlabel('A_R')
    ax.set_ylabel('A_G')
    ax.set_title('Hard Assignment (Voronoi)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Covariance comparison
    ax = axes[2]
    methods = ['True', 'Hard', 'Soft EM', 'MLE']
    comp0_errors = [0,
                   np.linalg.norm(cov_hard[0] - true_covariances[0]),
                   np.linalg.norm(cov_em[0] - true_covariances[0]),
                   np.linalg.norm(cov_mle[0] - true_covariances[0])]
    comp1_errors = [0,
                   np.linalg.norm(cov_hard[1] - true_covariances[1]),
                   np.linalg.norm(cov_em[1] - true_covariances[1]),
                   np.linalg.norm(cov_mle[1] - true_covariances[1])]

    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, comp0_errors, width, label='Component 0', alpha=0.7)
    ax.bar(x + width/2, comp1_errors, width, label='Component 1', alpha=0.7)
    ax.set_xlabel('Method')
    ax.set_ylabel('Frobenius Error')
    ax.set_title('Covariance Estimation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/fixed_means_prototype.png', dpi=150)
    print(f"\nSaved visualization to /tmp/fixed_means_prototype.png")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    hard_error_avg = np.mean([
        np.linalg.norm(cov_hard[k] - true_covariances[k])
        for k in range(len(true_means))
    ])
    em_error_avg = np.mean([
        np.linalg.norm(cov_em[k] - true_covariances[k])
        for k in range(len(true_means))
    ])
    mle_error_avg = np.mean([
        np.linalg.norm(cov_mle[k] - true_covariances[k])
        for k in range(len(true_means))
    ])

    print(f"Average Frobenius error:")
    print(f"  Hard assignment: {hard_error_avg:.6f}")
    print(f"  Soft EM:         {em_error_avg:.6f}")
    print(f"  MLE (scipy):     {mle_error_avg:.6f}")

    # Find best method
    errors = {
        'Hard assignment': hard_error_avg,
        'Soft EM': em_error_avg,
        'MLE (scipy)': mle_error_avg
    }
    best_method = min(errors, key=errors.get)
    best_error = errors[best_method]

    print(f"\n✓ Best method: {best_method} (error: {best_error:.6f})")

    # Compare to others
    for method, error in errors.items():
        if method != best_method:
            ratio = error / best_error
            print(f"  vs {method}: {ratio:.2f}x worse")

    # Recommendations
    print("\nRecommendations:")
    if best_method == 'Hard assignment':
        print("  → Use hard assignment: simplest, fastest, and most accurate")
    elif best_method == 'Soft EM':
        print("  → Use soft EM: good balance of accuracy and interpretability")
    else:
        print("  → Use MLE: most statistically principled and accurate")

    return {
        'hard': {'covariances': cov_hard, 'weights': weights_hard, 'error': hard_error_avg},
        'em': {'covariances': cov_em, 'weights': weights_em, 'error': em_error_avg},
        'mle': {'covariances': cov_mle, 'weights': weights_mle, 'error': mle_error_avg},
    }


if __name__ == "__main__":
    results = compare_methods()
