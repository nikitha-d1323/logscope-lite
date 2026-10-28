"""Load HDFS sessions (or generate fake demo data).

A "session" = the ordered list of event IDs of one HDFS block, e.g. ["E5", "E22", "E5"].
Every dataset is returned as: seqs (list of lists), y (0 normal / 1 anomaly), templates (event id -> text).
"""
import random

import numpy as np
import pandas as pd

DEMO_TEMPLATES = {
    "E1": "Receiving block <*> src <*> dest <*>",
    "E2": "Allocating block <*> to datanode <*>",
    "E3": "Writing block <*> to disk",
    "E4": "PacketResponder <*> for block <*> terminating",
    "E5": "Received block <*> of size <*> from <*>",
    "E6": "Deleting block <*> file <*>",
    "E7": "Exception in receiveBlock for block <*> error",
    "E8": "Verification succeeded for block <*>",
}


def parse_features(text):
    """'[E5,E22,E5]' -> ['E5', 'E22', 'E5']"""
    items = str(text).strip().strip("[]").split(",")
    return [t.strip().strip("'\"") for t in items if t.strip()]


def load_hdfs(traces_csv, templates_csv=None, max_sessions=None):
    """Loghub HDFS_v1 files: preprocessed/Event_traces.csv and preprocessed/HDFS.log_templates.csv.

    Row order of Event_traces.csv is treated as time order (needed for the chronological split).
    """
    df = pd.read_csv(traces_csv, nrows=max_sessions)
    seqs = [parse_features(f) for f in df["Features"]]
    y = (df["Label"] == "Fail").astype(int).to_numpy()
    templates = {}
    if templates_csv:
        t = pd.read_csv(templates_csv)
        templates = dict(zip(t["EventId"], t["EventTemplate"]))
    return seqs, y, templates


def make_demo(n=20000, seed=0):
    """Fake logs: a fixed workflow for normal sessions; 4% anomalies that break it."""
    rng = random.Random(seed)
    seqs, y = [], []
    for _ in range(n):
        s = ["E1", "E2", "E3"] + ["E4"] * (2 + int(rng.expovariate(0.2))) + ["E5"] + ["E8"] * rng.choice([0, 0, 1, 2]) + ["E6"]
        label = 0
        if rng.random() < 0.04:
            label = 1
            kind = rng.choice(["error", "swap", "drop"])
            if kind == "error":  # easy case: a tell-tale error event
                s.insert(rng.randrange(2, len(s)), "E7")
            elif kind == "swap":  # subtle: two different neighbouring steps in the wrong order
                j = rng.choice([j for j in range(len(s) - 1) if s[j] != s[j + 1]])
                s[j], s[j + 1] = s[j + 1], s[j]
            else:  # subtle: a mandatory step is missing
                s.remove(rng.choice(["E2", "E3", "E5"]))
        seqs.append(s)
        y.append(label)
    return seqs, np.array(y), dict(DEMO_TEMPLATES)