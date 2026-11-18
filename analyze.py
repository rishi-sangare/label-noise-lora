"""Per-class recall (mean over seeds) and the label mix of each training set.

Explains *why* a condition scored what it did, not just how much.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import LABELS

CONDITIONS = ["clean", "random20", "random40", "systematic20", "random40_filtered", "clean_sizematched"]


def recall_by_class(pred_file):
    hit, tot = Counter(), Counter()
    for line in open(pred_file):
        d = json.loads(line)
        tot[d["gold"]] += 1
        hit[d["gold"]] += d["gold"] == d["pred"]
    return {lab: hit[lab] / tot[lab] for lab in LABELS}


def main():
    lines = ["| Condition | " + " | ".join(LABELS) + " | Train label mix (W/Sp/B/ST) |",
             "|---" * (len(LABELS) + 2) + "|"]
    for name in ["zero-shot", "judge-3b-zero-shot"] + CONDITIONS:
        files = sorted(Path("results/preds").glob(f"{name}-s*.jsonl")) or [Path(f"results/preds/{name}.jsonl")]
        per = defaultdict(list)
        for f in files:
            for lab, r in recall_by_class(f).items():
                per[lab].append(r)
        mix = ""
        train = Path(f"data/{name}/train.jsonl")
        if train.exists():
            c = Counter(json.loads(l)["messages"][1]["content"] for l in open(train))
            mix = "/".join(str(c[l]) for l in LABELS)
        cells = " | ".join(f"{100 * sum(v) / len(v):.0f}%" for v in (per[l] for l in LABELS))
        lines.append(f"| {name} | {cells} | {mix} |")
    table = "\n".join(lines)
    Path("results/per_class.md").write_text(table + "\n")
    print(table)
    print()
    print(Path("data/random40_filtered/filter_stats.json").read_text())


if __name__ == "__main__":
    main()
