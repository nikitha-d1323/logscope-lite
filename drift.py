"""DriftForge-lite: age the TEST logs the way a software update would.

Two kinds of change, both keep the true label:
  - rewrite log statements: a fraction `rate` of templates get reworded text AND a new event ID
    (an ID-only model has never seen that ID; a semantic model can still read the text)
  - event noise: with probability rate/2 a session loses an event or repeats one (retry logic)
"""
import random

SYNONYMS = {
    "Receiving": "Getting", "Received": "Obtained", "block": "chunk", "Deleting": "Removing",
    "Allocating": "Assigning", "terminating": "closing", "Writing": "Saving", "Verification": "Check",
    "succeeded": "passed", "Exception": "Failure", "error": "fault", "size": "length", "file": "path",
}


def reword(text):
    new = " ".join(SYNONYMS.get(w, w) for w in text.split())
    return new if new != text else text + " now"  # fallback: insert an extra token


def drift(seqs, templates, rate, seed=0):
    """Return (aged sequences, extended template dict). rate=0 returns the input unchanged."""
    if rate == 0:
        return seqs, templates
    rng = random.Random(seed)
    events = sorted(templates)
    changed = set(rng.sample(events, round(rate * len(events))))
    new_templates = dict(templates)
    for e in changed:
        new_templates[e + "'"] = reword(templates[e])
    aged = []
    for s in seqs:
        s = [e + "'" if e in changed else e for e in s]
        if rng.random() < rate / 2:
            i = rng.randrange(len(s))
            s = s[:i] + s[i + 1:] if rng.random() < 0.5 and len(s) > 1 else s[:i + 1] + s[i:]
        aged.append(s)
    return aged, new_templates
