import json
import random
from pathlib import Path

from prompting.models import problem_from_record


class ProblemSet:
    def __init__(self, name, records):
        self.name = name
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def head(self, n=5):
        return ProblemSet(f"{self.name}_head{n}", self.records[:n])

    def subset(self, indices, name=None):
        rows = [self.records[i] for i in indices]
        return ProblemSet(name or self.name, rows)

    def problems(self):
        return [problem_from_record(record) for record in self.records]

    def summary(self):
        n_mcq = sum(bool(r.get("options")) for r in self.records)
        n_free = len(self.records) - n_mcq
        n_answered = sum("answer" in r and r.get("answer") is not None for r in self.records)

        return {
            "name": self.name,
            "n": len(self.records),
            "n_mcq": n_mcq,
            "n_free_form": n_free,
            "n_answered": n_answered,
        }


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_problem_set(path, name=None):
    path = Path(path)
    return ProblemSet(name or path.stem, load_jsonl(path))


def basic_problem_key(record):
    if record.get("options"):
        return "mcq"
    return "free_form"


def stratified_split(problem_set, val_frac=0.2, seed=414, key_fn=basic_problem_key):
    rng = random.Random(seed)
    groups = {}

    for record in problem_set.records:
        key = key_fn(record)
        groups.setdefault(key, []).append(record)

    train = []
    val = []

    for key, records in groups.items():
        records = list(records)
        rng.shuffle(records)

        if len(records) <= 1:
            train.extend(records)
            continue

        n_val = max(1, round(len(records) * val_frac))
        val.extend(records[:n_val])
        train.extend(records[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    return {
        "train": ProblemSet(f"{problem_set.name}_train", train),
        "val": ProblemSet(f"{problem_set.name}_val", val),
    }


def load_public_splits(path="data/public.jsonl", val_frac=0.2, seed=414):
    public_set = load_problem_set(path, name="public")
    splits = stratified_split(public_set, val_frac=val_frac, seed=seed)
    splits["public"] = public_set
    return splits


def load_private_set(path="data/private.jsonl"):
    return load_problem_set(path, name="private")