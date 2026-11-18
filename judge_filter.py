"""Model-assisted QA: a bigger local model re-labels the noisy training set,
and we keep only the examples where it agrees with the given label.

This mimics a common data-vendor workflow: humans label, a model flags
disagreements, and flagged items get dropped or sent for review.

Usage: python judge_filter.py --src random40 --dst random40_filtered
"""
import argparse
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from evaluate import parse
from prepare_data import PROMPT

JUDGE = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def main(src, dst, judge=JUDGE):
    model, tok = load(judge)
    sampler = make_sampler(temp=0.0)
    train = [json.loads(l) for l in open(f"data/{src}/train.jsonl")]
    truth = [json.loads(l)["label"] for l in open("data/train_clean_labels.jsonl")]
    kept, stats = [], {"kept": 0, "kept_wrong": 0, "dropped": 0, "dropped_correct": 0}
    judge_correct = 0
    for ex, true_label in zip(train, truth):
        given = ex["messages"][1]["content"]
        msgs = [ex["messages"][0]]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        guess, _ = parse(generate(model, tok, prompt=prompt, max_tokens=8, sampler=sampler))
        judge_correct += guess == true_label
        if guess == given:
            kept.append(ex)
            stats["kept"] += 1
            stats["kept_wrong"] += given != true_label
        else:
            stats["dropped"] += 1
            stats["dropped_correct"] += given == true_label
    stats["judge_accuracy_vs_true_labels"] = round(judge_correct / len(train), 4)
    stats["noise_rate_after"] = round(stats["kept_wrong"] / max(stats["kept"], 1), 4)
    out = Path(f"data/{dst}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (out / "valid.jsonl").write_text(Path(f"data/{src}/valid.jsonl").read_text())
    (out / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    main(a.src, a.dst)
