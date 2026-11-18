"""Aggregate results/results.jsonl into a table (mean ± std over seeds) and a chart."""
import json
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["zero-shot", "clean", "random20", "random40", "systematic20",
         "random40_filtered", "clean_sizematched", "judge-3b-zero-shot"]
NICE = {
    "zero-shot": "No fine-tune (0.5B)",
    "clean": "Clean labels",
    "random20": "20% random noise",
    "random40": "40% random noise",
    "systematic20": "20% systematic noise",
    "random40_filtered": "40% noise, judge-filtered",
    "clean_sizematched": "Clean, same size as filtered",
    "judge-3b-zero-shot": "3B judge, no fine-tune",
}


def main():
    groups = defaultdict(list)
    for line in open("results/results.jsonl"):
        r = json.loads(line)
        key = r["name"].rsplit("-s", 1)[0] if r["name"][-3:-1] == "-s" else r["name"]
        groups[key].append(r)

    rows = []
    for key in ORDER:
        if key not in groups:
            continue
        g = groups[key]
        acc = [r["accuracy"] for r in g]
        f1 = [r["macro_f1"] for r in g]
        fmt = [r["valid_format"] for r in g]
        rows.append({
            "key": key, "label": NICE[key], "runs": len(g),
            "acc": st.mean(acc), "acc_sd": st.stdev(acc) if len(acc) > 1 else 0.0,
            "f1": st.mean(f1), "fmt": st.mean(fmt),
        })

    lines = ["| Training data | Runs | Accuracy (mean ± sd) | Macro-F1 | Valid format |",
             "|---|---|---|---|---|"]
    for r in rows:
        sd = f" ± {100 * r['acc_sd']:.1f}" if r["runs"] > 1 else ""
        lines.append(f"| {r['label']} | {r['runs']} | {100 * r['acc']:.1f}%{sd} | "
                     f"{100 * r['f1']:.1f} | {100 * r['fmt']:.1f}% |")
    table = "\n".join(lines)
    open("results/summary.md", "w").write(table + "\n")
    print(table)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([r["label"] for r in rows][::-1], [100 * r["acc"] for r in rows][::-1],
            xerr=[100 * r["acc_sd"] for r in rows][::-1], color="#2f5d50", capsize=3)
    for i, r in enumerate(rows[::-1]):
        ax.text(100 * (r["acc"] + r["acc_sd"]) + 1.5, i, f"{100 * r['acc']:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Test accuracy on 500 held-out AG News articles (%)")
    ax.set_title("Qwen2.5-0.5B + LoRA: what label noise does to a fine-tune")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/accuracy.png", dpi=160)


if __name__ == "__main__":
    main()
