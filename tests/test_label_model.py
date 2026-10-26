import numpy as np

from gist.labels.label_model import ABSTAIN, LabelModel, majority_vote


def _synthetic(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.8, 0.15, 0.05])
    # (accuracy, coverage, class it can fire on)
    lfs = [(0.95, 0.6, None), (0.7, 0.9, None), (0.9, 0.5, 1), (0.85, 0.7, 2), (0.6, 0.8, None)]
    votes = np.full((n, len(lfs)), ABSTAIN)
    for j, (acc, cov, only) in enumerate(lfs):
        fire = rng.random(n) < cov
        if only is not None:
            fire &= (y == only) | (rng.random(n) < 0.03)
        correct = rng.random(n) < acc
        wrong = (y + rng.integers(1, 3, n)) % 3
        votes[fire, j] = np.where(correct, y, wrong)[fire]
    return y, votes


def test_label_model_beats_majority_vote():
    y, votes = _synthetic()
    covered = (votes != ABSTAIN).any(1)
    lm = LabelModel().fit(votes)
    acc_lm = (lm.predict_proba(votes).argmax(1) == y)[covered].mean()
    acc_mv = (majority_vote(votes).argmax(1) == y)[covered].mean()
    assert acc_lm >= acc_mv - 0.005
    assert acc_lm > 0.85


def test_lf_accuracy_ranks_good_lf_above_bad():
    _, votes = _synthetic()
    acc = LabelModel().fit(votes).lf_accuracy()
    assert acc[0] > acc[4]
