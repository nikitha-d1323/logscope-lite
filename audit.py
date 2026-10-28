"""LogScope Auditor: checks how 'easy' (leaky) a train/test split is."""
import numpy as np
from sklearn.metrics import f1_score


def dup_rate(seqs):
    """Share of sessions that are exact copies of another session."""
    return 1 - len({tuple(s) for s in seqs}) / len(seqs)


def overlap(train, test):
    """Share of TEST sessions whose exact sequence also appears in TRAIN (memorisation risk)."""
    seen = {tuple(s) for s in train}
    return float(np.mean([tuple(s) in seen for s in test])) if test else 0.0


def novelty(train, test):
    """Share of TEST sessions containing an event never seen in TRAIN."""
    known = {e for s in train for e in s}
    return float(np.mean([any(e not in known for e in s) for s in test])) if test else 0.0


def best_single_event(seqs, y):
    """Trivial Signal Baseline: the one event whose mere presence best predicts an anomaly."""
    events = sorted({e for s in seqs for e in s})
    idx = {e: i for i, e in enumerate(events)}
    present = np.zeros((len(seqs), len(events)), dtype=bool)
    for i, s in enumerate(seqs):
        for e in set(s):
            present[i, idx[e]] = True
    pos = (y == 1)[:, None]
    tp = (present & pos).sum(0)
    fp = (present & ~pos).sum(0)
    fn = pos.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = int(f1.argmax())
    return events[best], float(f1[best])


def audit_card(name, train, test, y_train, y_test):
    """Print the audit card for one protocol."""
    event, _ = best_single_event(train, y_train)  # pick the event on TRAIN only
    pred = [int(event in s) for s in test]
    tsb = f1_score(y_test, pred, zero_division=0)
    print(f"\n=== LogScope Audit: {name} ===")
    print(f"train sessions        {len(train):>9,}   test sessions {len(test):>9,}")
    print(f"duplication (train)   {dup_rate(train):>8.1%}")
    print(f"train/test overlap    {overlap(train, test):>8.1%}")
    print(f"unseen-event rate     {novelty(train, test):>8.1%}")
    print(f"anomaly share (test)  {np.mean(y_test):>8.1%}")
    print(f"Trivial Signal F1     {tsb:>8.3f}   (rule: 'anomaly iff {event} appears')")
