# LogScope-Lite

Leakage-aware, drift-tested log anomaly detection for distributed systems (NLP + deep learning).

Most papers report F1 around 0.97 on HDFS logs. This project asks: **is that number honest?**
It audits the benchmark, re-evaluates models under a stricter split, stress-tests them with simulated log changes, and compares a small deep model (HALO) against simpler baselines.

## What's inside

| File | Purpose |
|---|---|
| `data.py` | Load HDFS sessions (Loghub) or generate fake demo data |
| `audit.py` | **Auditor**: duplication, train/test overlap, unseen events, Trivial Signal Baseline |
| `drift.py` | **DriftForge-lite**: reword log statements + add/drop events in test data |
| `models.py` | TrivialRule, Count+LR, DeepLog (LSTM), **HALO** (BiLSTM + attention + text embeddings + open-set gate) |
| `run.py` | Runs protocols, prints the summary table, saves `results/` |

## Protocols

- **P1**: random 80/20 split (what most papers do)
- **P3**: chronological 80/20, test sequences already seen in training removed, plus drift at 10/20/30%

## Metrics

Precision, Recall, F1, PR-AUC, MCC, plus three summary numbers:

- **LIF** (Leakage Inflation Factor) = F1 on P1 / F1 on P3. Above 1 means the random split flatters the model.
- **LiftOverTSB** = F1 minus the F1 of a single-event rule. Shows what the model adds beyond a trivial check.
- **DRS** (Drift Resilience Score) = average of F1 under drift / F1 without drift.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py --demo               # fake data, runs in a few minutes
```

Real data (Loghub HDFS_v1, preprocessed folder):

```bash
python run.py --traces data/Event_traces.csv \
              --templates data/HDFS.log_templates.csv \
              --max_sessions 200000
```

## Notes and limits

- `Event_traces.csv` rows are assumed to be in time order (needed for the chronological split). Verify this on your copy.
- After removing already-seen test sequences, the test set has a much higher anomaly share. Compare models against each other, not against P1 numbers.
- Text embeddings use TF-IDF. Replacing `TfidfEmbedder` with sentence-transformers MiniLM is the obvious upgrade.
- Single seed. Run with 3 seeds and report mean ± std before claiming results.

## Results

_Fill in after running:_ paste `results/summary.csv` and `results/drift_curve.png` here.
