"""Eval harness: run a model (optionally with a LoRA adapter) over the fixed test set.

Reports accuracy, macro-F1, valid-format rate and speed, and writes per-example
predictions so any number in the README can be traced back to outputs.

Usage:
  python evaluate.py --name zero-shot
  python evaluate.py --name clean-s0 --adapter adapters/clean-s0
"""
import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from prepare_data import LABELS, PROMPT

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def parse(output):
    """Return (label or None, strictly_valid)."""
    out = output.strip()
    if out in LABELS:
        return out, True
    low = out.lower()
    for lab in LABELS:  # lenient: label appears at the start of the answer
        if low.startswith(lab.lower()) or low.startswith(lab.split("/")[0].lower()):
            return lab, False
    return None, False


def macro_f1(golds, preds):
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def run(name, adapter=None, model_id=MODEL, test_path="data/test.jsonl", limit=None):
    model, tok = load(model_id, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)  # greedy: deterministic
    rows = [json.loads(l) for l in open(test_path)]
    if limit:
        rows = rows[:limit]
    preds, golds, records = [], [], []
    strict = 0
    t0 = time.time()
    for r in rows:
        msgs = [{"role": "user", "content": PROMPT.format(text=r["text"])}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=prompt, max_tokens=8, sampler=sampler)
        label, ok = parse(out)
        strict += ok
        preds.append(label)
        golds.append(r["label"])
        records.append({"gold": r["label"], "pred": label, "raw": out})
    secs = time.time() - t0
    n = len(rows)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    result = {
        "name": name,
        "adapter": adapter,
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1(golds, preds), 4),
        "valid_format": round(strict / n, 4),
        "unparseable": sum(p is None for p in preds),
        "ms_per_example": round(1000 * secs / n, 1),
    }
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    with open(f"results/preds/{name}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open("results/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.name, a.adapter, a.model, limit=a.limit)
