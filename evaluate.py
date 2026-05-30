"""
evaluate.py
===========
Third-Party Breach Haber Siniflandirma -- Test Seti Degerlendirme

Kullanim ornekleri:
    # Fine-tuned modeli degerlendir
    python evaluate.py --model_dir outputs/smollm2

    # Tum egitilmis modelleri karsılastir
    python evaluate.py --compare outputs/smollm2 outputs/tinyllama outputs/qwen outputs/gemma

    # Zero-shot degerlendirme
    python evaluate.py --model_name "HuggingFaceTB/SmolLM2-360M" --zero_shot
"""

import argparse
import os
import json
import csv

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GPU sunucu ortami icin headless
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ── Yardimci: CSV yukle ────────────────────────────────────────────────────────

def load_csv(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


# ── Fine-tuned model degerlendirmesi ─────────────────────────────────────────

def evaluate_finetuned(model_dir: str, test_csv: str, batch_size: int = 16):
    print(f"\n[Fine-tuned] {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True
    )
    model.eval()

    texts, labels = load_csv(test_csv)
    all_preds, all_probs = [], []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model(**enc)

        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        preds = np.argmax(out.logits.cpu().numpy(), axis=-1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    return compute_and_print(labels, all_preds, all_probs, model_dir)


# ── Zero-shot degerlendirme ───────────────────────────────────────────────────

ZERO_SHOT_PROMPT = (
    "Is the following news article about a third-party data breach "
    "(i.e., a breach caused by a supplier, vendor, or partner)? "
    "Answer with yes or no only.\n\nArticle:\n{text}\n\nAnswer:"
)

def evaluate_zero_shot(model_name: str, test_csv: str):
    print(f"\n[Zero-shot] {model_name}")
    texts, labels = load_csv(test_csv)

    gen = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        trust_remote_code=True,
        max_new_tokens=5,
        pad_token_id=2,
    )

    preds = []
    for text in texts:
        prompt  = ZERO_SHOT_PROMPT.format(text=text[:800])
        output  = gen(prompt)[0]["generated_text"]
        answer  = output.split("Answer:")[-1].strip().lower()
        pred    = 1 if answer.startswith("yes") else 0
        preds.append(pred)

    # Zero-shot'ta olasilik yok, ROC_AUC hesaplanamaz
    return compute_and_print(labels, preds, preds, f"{model_name} (zero-shot)")


# ── Metrik hesaplama & yazici ─────────────────────────────────────────────────

def compute_and_print(labels, preds, probs, name: str) -> dict:
    labels = np.array(labels)
    preds  = np.array(preds)
    probs  = np.array(probs)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["Not Breach", "Breach"])

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{report}")

    return {
        "name"     : name,
        "accuracy" : round(acc,  4),
        "f1"       : round(f1,   4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "roc_auc"  : round(auc,  4) if not np.isnan(auc) else None,
        "confusion_matrix": cm.tolist(),
    }


# ── Karsilastirma tablosu ve grafik ──────────────────────────────────────────

def plot_comparison(results: list, out_dir: str = "outputs"):
    models = [r["name"].split("/")[-1] for r in results]
    metrics = ["accuracy", "f1", "precision", "recall"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Karsilastirmasi -- Third-Party Breach Siniflandirma")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"\nGrafik kaydedildi: {path}")


# ── Arguman ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--zero_shot",  action="store_true")
    parser.add_argument("--compare",    nargs="+", default=None,
                        help="Karsilastirilacak model dizinleri")
    parser.add_argument("--test_csv",   type=str, default="data/test.csv")
    parser.add_argument("--out_dir",    type=str, default="outputs")
    return parser.parse_args()


# ── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.compare:
        all_results = []
        for model_dir in args.compare:
            res = evaluate_finetuned(model_dir, args.test_csv)
            all_results.append(res)

        print("\n" + "=" * 60)
        print("  OZET KARSILASTIRMA TABLOSU")
        print("=" * 60)
        header = f"{'Model':<25} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>7}"
        print(header)
        print("-" * 60)
        for r in all_results:
            name = r["name"].split("/")[-1][:24]
            auc  = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "  N/A "
            print(f"{name:<25} {r['accuracy']:>6.4f} {r['f1']:>6.4f} "
                  f"{r['precision']:>6.4f} {r['recall']:>6.4f} {auc:>7}")

        plot_comparison(all_results, args.out_dir)

        summary_path = os.path.join(args.out_dir, "comparison_results.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Sonuclar kaydedildi: {summary_path}")

    elif args.zero_shot and args.model_name:
        evaluate_zero_shot(args.model_name, args.test_csv)

    elif args.model_dir:
        evaluate_finetuned(args.model_dir, args.test_csv)

    else:
        print("--model_dir, --compare veya (--model_name + --zero_shot) belirtin.")


if __name__ == "__main__":
    main()
