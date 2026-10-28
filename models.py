"""Four detectors. Every model has: fit(seqs, y, templates) and scores(seqs, templates) -> anomaly score.
A session is predicted anomalous when score > 0.5.

  TrivialRule  one event decides everything (the sanity-check baseline)
  CountLR      event counts + logistic regression (classical)
  DeepLog      LSTM predicts next event, unsupervised (Du et al. 2017)
  HALO         BiLSTM + attention over [event-ID embedding + text embedding] with an open-set gate
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from audit import best_single_event

PAD, UNK, MAXLEN = 0, 1, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Vocab:
    """Event string -> integer id (0 = PAD, 1 = UNK for events never seen in training)."""

    def __init__(self, seqs):
        self.idx = {e: i + 2 for i, e in enumerate(sorted({e for s in seqs for e in s}))}

    def enc(self, seq):
        return [self.idx.get(e, UNK) for e in seq]


class TfidfEmbedder:
    """Turns template TEXT into a vector. Swap for sentence-transformers MiniLM for a stronger version."""

    def __init__(self, texts):
        self.v = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[A-Za-z]+").fit(texts)

    def __call__(self, texts):
        return self.v.transform(texts).toarray().astype(np.float32)


class Base:
    def predict(self, seqs, templates):
        return (self.scores(seqs, templates) > 0.5).astype(int)


class TrivialRule(Base):
    name = "TrivialRule"

    def fit(self, seqs, y, templates):
        self.event, _ = best_single_event(seqs, y)

    def scores(self, seqs, templates):
        return np.array([float(self.event in s) for s in seqs])


class CountLR(Base):
    name = "Count+LR"

    def _x(self, seqs):
        X = np.zeros((len(seqs), len(self.idx)), dtype=np.float32)
        for i, s in enumerate(seqs):
            for e in s:
                if e in self.idx:
                    X[i, self.idx[e]] += 1
        return np.log1p(X)

    def fit(self, seqs, y, templates):
        self.idx = {e: i for i, e in enumerate(sorted({e for s in seqs for e in s}))}
        self.clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(self._x(seqs), y)

    def scores(self, seqs, templates):
        return self.clf.predict_proba(self._x(seqs))[:, 1]


class _NextEvent(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 32, padding_idx=PAD)
        self.lstm = nn.LSTM(32, 64, 2, batch_first=True)
        self.out = nn.Linear(64, V)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.out(h[:, -1])


class DeepLog(Base):
    name = "DeepLog"

    def __init__(self, h=10, epochs=3, seed=0):
        self.h, self.epochs, self.seed = h, epochs, seed

    def _windows(self, seqs):
        X, Y, S = [], [], []
        for i, q in enumerate(seqs):
            p = [PAD] * self.h + self.vocab.enc(q)
            for t in range(len(q)):
                X.append(p[t:t + self.h])
                Y.append(p[t + self.h])
                S.append(i)
        return np.array(X), np.array(Y), np.array(S)

    def fit(self, seqs, y, templates):
        torch.manual_seed(self.seed)
        normal = [q for q, l in zip(seqs, y) if l == 0]  # unsupervised: normal sessions only
        self.vocab = Vocab(normal)
        self.g = min(9, max(1, len(self.vocab.idx) // 3))  # top-g candidates
        X, Y, _ = self._windows(normal)
        X, Y = torch.tensor(X).to(DEV), torch.tensor(Y).to(DEV)
        self.net = _NextEvent(len(self.vocab.idx) + 2).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        for _ in range(self.epochs):
            for b in torch.randperm(len(X), device=DEV).split(512):
                opt.zero_grad()
                nn.functional.cross_entropy(self.net(X[b]), Y[b]).backward()
                opt.step()

    @torch.no_grad()
    def scores(self, seqs, templates):
        X, Y, S = self._windows(seqs)
        self.net.eval()
        miss = []
        for i in range(0, len(X), 8192):
            top = self.net(torch.tensor(X[i:i + 8192]).to(DEV)).topk(self.g, 1).indices.cpu()
            miss.append(~(top == torch.tensor(Y[i:i + 8192])[:, None]).any(1))
        miss = torch.cat(miss).numpy()
        return np.bincount(S[miss], minlength=len(seqs)).astype(float)  # misses per session


class _HaloNet(nn.Module):
    def __init__(self, V, D, d=32, hid=64):
        super().__init__()
        self.idemb = nn.Embedding(V, d, padding_idx=PAD)
        self.proj = nn.Linear(D, d)
        self.lstm = nn.LSTM(2 * d, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hid, 1)

    def forward(self, x, sem, nn_id, alpha, drop=None):
        mask = x > 0
        a = alpha[x]
        if drop is not None:  # training: randomly hide some IDs so the UNK embedding learns something
            a = a.masked_fill(drop, 0.0)
        a = a.unsqueeze(-1)
        # open-set gate: blend nearest known event's embedding with UNK, weighted by text similarity
        ident = a * self.idemb(nn_id[x]) + (1 - a) * self.idemb.weight[UNK]
        inp = torch.cat([ident, self.proj(sem[x])], -1)
        packed = pack_padded_sequence(inp, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        w = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(1)  # attention over events
        return self.out(self.drop((w.unsqueeze(-1) * h).sum(1))).squeeze(-1), w


class Halo(Base):
    def __init__(self, open_set=True, epochs=5, seed=0):
        self.open_set, self.epochs, self.seed = open_set, epochs, seed
        self.name = "HALO" if open_set else "HALO-noGate"

    def _prepare(self, seqs, templates):
        """Build per-event lookup tables (text vector, nearest known id, gate weight) + padded id matrix."""
        events = sorted({e for q in seqs for e in q})
        row = {e: i + 1 for i, e in enumerate(events)}  # row 0 = PAD
        vecs = self.embed([templates.get(e, e) for e in events])
        nn_id = np.zeros(len(events) + 1, dtype=np.int64)
        alpha = np.ones(len(events) + 1, dtype=np.float32)
        for e, r in row.items():
            if e in self.vocab.idx:
                nn_id[r] = self.vocab.idx[e]
            else:  # unseen event: find the most similar known event by text
                sims = self.known_vecs @ vecs[r - 1]
                j = int(sims.argmax())
                nn_id[r] = self.known_ids[j] if self.open_set else UNK
                alpha[r] = max(float(sims[j]), 0.0) if self.open_set else 0.0
        X = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
        for i, q in enumerate(seqs):
            ids = [row[e] for e in q[:MAXLEN]]
            X[i, :len(ids)] = ids
        sem = np.vstack([np.zeros((1, vecs.shape[1]), np.float32), vecs])
        return X, torch.tensor(sem).to(DEV), torch.tensor(nn_id).to(DEV), torch.tensor(alpha).to(DEV)

    def fit(self, seqs, y, templates):
        torch.manual_seed(self.seed)
        self.vocab = Vocab(seqs)
        known = list(self.vocab.idx)
        self.embed = TfidfEmbedder([templates.get(e, e) for e in known])
        self.known_vecs = self.embed([templates.get(e, e) for e in known])
        self.known_ids = np.array([self.vocab.idx[e] for e in known])
        X, sem, nn_id, alpha = self._prepare(seqs, templates)
        X, y = torch.tensor(X).to(DEV), torch.tensor(y, dtype=torch.float32).to(DEV)
        self.net = _HaloNet(len(self.vocab.idx) + 2, self.known_vecs.shape[1]).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        pos_w = torch.sqrt((y == 0).sum() / y.sum().clamp(min=1))  # softened class-imbalance weight
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        self.net.train()
        for _ in range(self.epochs):
            for b in torch.randperm(len(X), device=DEV).split(256):
                xb = X[b][:, :max(int((X[b] > 0).sum(1).max()), 1)]
                logits, _ = self.net(xb, sem, nn_id, alpha, drop=torch.rand(xb.shape, device=DEV) < 0.05)
                opt.zero_grad()
                loss_fn(logits, y[b]).backward()
                opt.step()

    @torch.no_grad()
    def scores(self, seqs, templates):
        X, sem, nn_id, alpha = self._prepare(seqs, templates)
        X = torch.tensor(X).to(DEV)
        self.net.eval()
        out = []
        for b in torch.arange(len(X), device=DEV).split(2048):
            xb = X[b][:, :max(int((X[b] > 0).sum(1).max()), 1)]
            out.append(torch.sigmoid(self.net(xb, sem, nn_id, alpha)[0]).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def explain(self, seq, templates, k=3):
        """Top-k most attended log lines of one session: [(position, event, template text), ...]."""
        X, sem, nn_id, alpha = self._prepare([seq], templates)
        self.net.eval()
        w = self.net(torch.tensor(X).to(DEV)[:, :len(seq)], sem, nn_id, alpha)[1][0]
        return [(i, seq[i], templates.get(seq[i], "")) for i in w.topk(min(k, len(seq))).indices.tolist()]
