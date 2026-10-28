"""Run every model under two protocols and print the honest-evaluation summary.

  P1  random 80/20 split              (what most papers do)
  P3  chronological 80/20 + test sequences already seen in training removed
      (+ drift stress: test logs aged at 10/20/30%)

  python run.py --demo
  python run.py --traces data/Event_traces.csv --templates data/HDFS.log_templates.csv --max_sessions 200000
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support

from audit import audit_card
from data import load_hdfs, make_demo
from drift import drift
from models import CountLR, DeepLog, Halo, TrivialRule

RATES = [0.0, 0.1, 0.2, 0.3]


def metrics(y, score, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    ap = average_precision_score(y, score) if y.sum() else float("nan")
    return dict(P=p, R=r, F1=f, PR_AUC=ap, MCC=matthews_corrcoef(y, pred))


def split_p1(n, rng):
    idx = rng.permutation(n)
    return idx[: int(0.8 * n)], idx[int(0.8 * n):]


def split_p3(seqs):
    cut = int(0.8 * len(seqs))
    seen = {tuple(s) for s in seqs[:cut]}
    return np.arange(cut), np.array([i for i in range(cut, len(seqs)) if tuple(seqs[i]) not in seen], dtype=int)


def run_protocol(name, tr, te, seqs, y, templates, rates, args, rows):
    Xtr, ytr, Xte, yte = [seqs[i] for i in tr], y[tr], [seqs[i] for i in te], y[te]
    if len(Xte) == 0 or yte.sum() == 0:
        print(f"[{name}] test set has no anomalies after deduplication; skipping (use more data)")
        return
    audit_card(name, Xtr, Xte, ytr, yte)
    models = [TrivialRule(), CountLR(), DeepLog(epochs=args.epochs, seed=args.seed),
              Halo(open_set=False, epochs=args.epochs, seed=args.seed),
              Halo(open_set=True, epochs=args.epochs, seed=args.seed)]
    for m in models:
        t0 = time.time()
        m.fit(Xtr, ytr, templates)
        train_s = time.time() - t0
        for r in rates:
            dx, dt = drift(Xte, templates, r, args.seed)
            sc = m.scores(dx, dt)
            row = dict(protocol=name, model=m.name, drift=r, train_s=round(train_s, 1),
                       **metrics(yte, sc, (sc > 0.5).astype(int)))
            rows.append(row)
            print(f"[{name}] {m.name:<12} drift={r:.1f}  P={row['P']:.3f} R={row['R']:.3f} F1={row['F1']:.3f}  ({train_s:.1f}s)")


def summarise(res):
    p1 = res[(res.protocol == "P1") & (res.drift == 0)].set_index("model")
    p3 = res[(res.protocol == "P3") & (res.drift == 0)].set_index("model")
    d = res[(res.protocol == "P3") & (res.drift > 0)]
    drs = d.assign(rel=d.F1 / d.model.map(p3.F1).clip(lower=1e-9)).groupby("model").rel.mean()
    return pd.DataFrame({
        "F1_P1": p1.F1, "F1_P3": p3.F1,
        "LIF": p1.F1 / p3.F1.clip(lower=1e-9),            # Leakage Inflation Factor
        "LiftOverTSB": p3.F1 - p3.loc["TrivialRule", "F1"],  # gain over the one-event rule
        "DRS": drs,                                        # Drift Resilience Score
        "train_s": p3.train_s,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--traces")
    ap.add_argument("--templates")
    ap.add_argument("--max_sessions", type=int, default=200000)
    ap.add_argument("--max_train", type=int, default=50000, help="cap training sessions for speed")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.demo:
        seqs, y, templates = make_demo()
    elif args.traces:
        seqs, y, templates = load_hdfs(args.traces, args.templates, args.max_sessions)
    else:
        ap.error("use --demo or --traces PATH")
    print(f"sessions={len(seqs):,}  anomalies={y.sum():,} ({y.mean():.2%})")
    rng = np.random.default_rng(args.seed)

    def cap(tr):  # random subsample of the training set
        return rng.permutation(tr)[: args.max_train] if len(tr) > args.max_train else tr

    rows = []
    tr, te = split_p1(len(seqs), rng)
    run_protocol("P1", cap(tr), te, seqs, y, templates, [0.0], args, rows)
    tr, te = split_p3(seqs)
    run_protocol("P3", cap(tr), te, seqs, y, templates, RATES, args, rows)

    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/all_runs.csv", index=False)
    summary = summarise(res)
    summary.to_csv("results/summary.csv")
    print("\n=== SUMMARY ===")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    try:  # drift-curve figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for m, g in res[res.protocol == "P3"].groupby("model"):
            plt.plot(g.drift, g.F1, marker="o", label=m)
        plt.xlabel("drift rate"); plt.ylabel("F1 (P3)"); plt.legend(); plt.title("Robustness to log drift")
        plt.savefig("results/drift_curve.png", dpi=150, bbox_inches="tight")
    except ImportError:
        pass


if __name__ == "__main__":
    main()