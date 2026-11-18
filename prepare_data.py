"""Build the AG News subsets and the label-noise variants.

Every variant uses the same 2,000 training articles. Only the labels change.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 0
N_TRAIN, N_VALID, N_TEST = 2000, 200, 500
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
# For "systematic" noise: the class a tired annotator most plausibly confuses it with.
CONFUSABLE = {"World": "Business", "Sports": "World", "Business": "Sci/Tech", "Sci/Tech": "Business"}

PROMPT = (
    "Classify the news article into exactly one category: World, Sports, Business, Sci/Tech.\n"
    "Answer with the category name only.\n\nArticle: {text}"
)

OUT = Path("data")


def to_chat(text, label):
    return {"messages": [
        {"role": "user", "content": PROMPT.format(text=text)},
        {"role": "assistant", "content": label},
    ]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def flip(rows, rate, mode, rng):
    """Return (noisy_rows, n_flipped). rows are (text, label) pairs."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    to_flip = set(idx[: int(rate * len(rows))])
    out = []
    for i, (text, label) in enumerate(rows):
        if i in to_flip:
            if mode == "random":
                label = rng.choice([l for l in LABELS if l != label])
            else:
                label = CONFUSABLE[label]
        out.append((text, label))
    return out, len(to_flip)


def main():
    rng = random.Random(SEED)
    ds = load_dataset("fancyzhx/ag_news")
    train_pool = [(r["text"], LABELS[r["label"]]) for r in ds["train"]]
    test_pool = [(r["text"], LABELS[r["label"]]) for r in ds["test"]]
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)

    train = train_pool[:N_TRAIN]
    valid = train_pool[N_TRAIN : N_TRAIN + N_VALID]
    test = test_pool[:N_TEST]

    write_jsonl(OUT / "test.jsonl", [{"text": t, "label": l} for t, l in test])
    write_jsonl(OUT / "train_clean_labels.jsonl", [{"text": t, "label": l} for t, l in train])

    variants = {
        "clean": (0.0, "random"),
        "random20": (0.2, "random"),
        "random40": (0.4, "random"),
        "systematic20": (0.2, "systematic"),
    }
    summary = {}
    for name, (rate, mode) in variants.items():
        noisy, n = flip(train, rate, mode, random.Random(SEED + 1))
        d = OUT / name
        write_jsonl(d / "train.jsonl", [to_chat(t, l) for t, l in noisy])
        # Validation stays clean: it only tracks loss during training.
        write_jsonl(d / "valid.jsonl", [to_chat(t, l) for t, l in valid])
        write_jsonl(d / "labels.jsonl", [{"label": l} for _, l in noisy])
        wrong = sum(a[1] != b[1] for a, b in zip(noisy, train))
        summary[name] = {"rate": rate, "mode": mode, "flipped": n, "actually_wrong": wrong}
    (OUT / "variants.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
