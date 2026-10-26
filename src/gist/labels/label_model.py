"""Generative label model over labeling-function votes (Dawid & Skene 1979, Snorkel-style).

Each labeling function j has a class-conditional confusion matrix ``pi[j, y, v]`` =
P(LF j votes v | true class y), where v ranges over the K classes plus ABSTAIN. EM alternates
between posterior class probabilities per item and re-estimating ``pi`` and the class prior.
No Snorkel dependency; NumPy only, vectorised so millions of windows fit in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMAL, JAMMED, SPOOFED = 0, 1, 2
CLASSES = ("normal", "jammed", "spoofed")
ABSTAIN = -1


@dataclass
class LabelModel:
    n_classes: int = 3
    n_iter: int = 50
    smoothing: float = 1.0  # Dirichlet pseudo-count on every confusion cell
    tol: float = 1e-5

    def fit(self, votes: np.ndarray, prior_accuracy: float = 0.8) -> "LabelModel":
        """votes: int array (n_items, n_lfs) with values in {ABSTAIN, 0..K-1}."""
        n, m = votes.shape
        k = self.n_classes
        onehot = self._onehot(votes)  # (n, m, k+1); last slot = abstain
        # Initialise from a majority vote so EM starts near a sensible solution.
        post = self._majority(votes)
        self.prior_ = np.full(k, 1.0 / k)
        prev = None
        for _ in range(self.n_iter):
            # M-step: confusion matrices pi[j, y, v]
            counts = np.einsum("ny,nmv->myv", post, onehot) + self.smoothing
            # Keep LFs better than random at the start: add extra mass on the diagonal.
            if prev is None:
                counts[:, np.arange(k), np.arange(k)] += prior_accuracy * n / k
            self.pi_ = counts / counts.sum(-1, keepdims=True)
            self.prior_ = (post.sum(0) + self.smoothing) / (n + k * self.smoothing)
            # E-step
            post = self._posterior(onehot)
            ll = float(np.log(np.clip(post.max(1), 1e-12, None)).mean())
            if prev is not None and abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict_proba(self, votes: np.ndarray) -> np.ndarray:
        return self._posterior(self._onehot(votes))

    def lf_accuracy(self) -> np.ndarray:
        """Estimated P(vote correct | LF votes non-abstain), per LF."""
        k = self.n_classes
        diag = self.pi_[:, np.arange(k), np.arange(k)]           # (m, k)
        voted = 1 - self.pi_[:, :, k]                              # P(non-abstain | y)
        return (diag * self.prior_).sum(1) / np.clip((voted * self.prior_).sum(1), 1e-9, None)

    # -- helpers -----------------------------------------------------------------------
    def _onehot(self, votes: np.ndarray) -> np.ndarray:
        k = self.n_classes
        idx = np.where(votes == ABSTAIN, k, votes)
        return np.eye(k + 1, dtype=np.float32)[idx]

    def _posterior(self, onehot: np.ndarray) -> np.ndarray:
        logpi = np.log(self.pi_)                                     # (m, k, k+1)
        ll = np.einsum("nmv,myv->ny", onehot, logpi) + np.log(self.prior_)
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(1, keepdims=True)

    def _majority(self, votes: np.ndarray) -> np.ndarray:
        k = self.n_classes
        counts = np.stack([(votes == c).sum(1) for c in range(k)], 1).astype(float) + 1e-3
        return counts / counts.sum(1, keepdims=True)


def majority_vote(votes: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Baseline for the 'label model → majority vote' ablation (soft, ties split)."""
    counts = np.stack([(votes == c).sum(1) for c in range(n_classes)], 1).astype(float)
    counts[counts.sum(1) == 0] = 1.0
    return counts / counts.sum(1, keepdims=True)
