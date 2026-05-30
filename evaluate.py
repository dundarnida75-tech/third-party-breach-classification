"""
evaluate.py  (v2 -- 3-class + verimlilik metrikleri)
=====================================================
Third-Party Breach Haber Siniflandirma -- Test Seti Degerlendirme

Metrikler:
  Performans : Accuracy, Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall
  Verimlilik : Inference suresi (ms/ornek), Model boyutu (MB), GPU bellek (GB)
  Analiz     : Confusion matrix, per-class rapor, hata ornekleri

Kullanim:
    # Tek model
    python evaluate.py --model_dir outputs/smollm2

    # Tum modelleri karsilastir
    python evaluate.py --compare outputs/smollm2 outputs/tinyllama outputs/qwen outputs/gemma

    # Zero-shot
    python evaluate.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --zero_shot
"""

import argparse, os, json, csv, time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

NUM_LABELS  = 3
LABEL_NAMES = ["Ihlal Yok", "Dogrudan Ihlal", "3. Taraf Ihlali"]

# ── Yardimci ─────────────────────────────────────────────────────────────

def load_csv(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def model_disk_size_mb(model_dir: str) -> float:
    total = 0
    for root, _, files in os.walk(model_dir):
        for fname in files:
            if fname.endswith((".bin", ".safetensors", ".pt")):
                total += os.path.getsize(os.path.join(root, fname))
    return round(total / (1024 ** 2), 1)


def gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    return 0.0


# ── Fine-tuned degerlendirme ───────────────────────────────────────────────

def evaluate_finetuned(model_dir: str, test_csv: str, batch_size: int = 16) -> dict:
    print(f"\n{'='*65}")
    print(f"[Fine-tuned] {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True
    )
    model.eval()

    texts, labels = load_csv(test_csv)
    all_preds, inference_times = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=512,
            padding=True, return_tensors="pt"
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**enc)
        t1 = time.perf_counter()

        preds = np.argmax(out.logits.cpu().numpy(), axis=-1)
        all_preds.extend(preds.tolist())
        batch_ms = (t1 - t0) * 1000 / len(batch)
        inference_times.append(batch_ms)

    # Verimlilik metrikleri
    avg_inf_ms  = round(np.mean(inference_times), 2)
    gpu_gb      = gpu_memory_gb()
    disk_mb     = model_disk_size_mb(model_dir)

    # Model meta varsa oku
    meta_path = os.path.join(model_dir, "model_meta.json")
    model_name = model_dir
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("model_name", model_dir)

    return compute_and_print(
        labels, all_preds, model_name,
        extra={
            "inference_ms_per_sample": avg_inf_ms,
            "gpu_memory_gb": gpu_gb,
            "disk_size_mb": disk_mb,
        }
    )


# ── Zero-shot degerlendirme ───────────────────────────────────────────────

ZERO_SHOT_PROMPT = (
    "Classify the following cybersecurity news article into one of three categories:\n"
    "0 = No breach (general security news)\n"
    "1 = Direct breach (the organization was directly attacked)\n"
    "2 = Third-party breach (caused by a vendor, supplier, or partner)\n\n"
    "Article:\n{text}\n\n"
    "Answer with only the number (0, 1, or 2):"
)

def evaluate_zero_shot(model_name: str, test_csv: str) -> dict:
    print(f"\n[Zero-shot] {model_name}")
    texts, labels = load_csv(test_csv)

    gen = pipeline(
        "text-generation", model=model_name,
        device_map="auto", trust_remote_code=True,
        max_new_tokens=5, torch_dtype=torch.float16,
    )

    preds, inf_times = [], []
    for text in texts:
        prompt = ZERO_SHOT_PROMPT.format(text=text[:800])
        t0 = time.perf_counter()
        out = gen(prompt)[0]["generated_text"]
        t1 = time.perf_counter()
        inf_times.append((t1 - t0) * 1000)

        answer = out.split("Answer with only the number")[-1].strip()
        digits = [c for c in answer if c in "012"]
        pred = int(digits[0]) if digits else 0
        preds.append(pred)

    return compute_and_print(
        labels, preds, f"{model_name} (zero-shot)",
        extra={"inference_ms_per_sample": round(np.mean(inf_times), 2)}
    )


# ── Metrik hesaplama ──────────────────────────────────────────────────────

def compute_and_print(labels, preds, name: str, extra: dict = None) -> dict:
    labels = np.array(labels)
    preds  = np.array(preds)

    acc  = accuracy_score(labels, preds)
    mf1  = f1_score(labels, preds, average="macro",    zero_division=0)
    wf1  = f1_score(labels, preds, average="weighted", zero_division=0)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels,    preds, average="macro", zero_division=0)
    cm   = confusion_matrix(labels, preds, labels=[0, 1, 2])
    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, zero_division=0
    )

    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro-F1        : {mf1:.4f}")
    print(f"  Weighted-F1     : {wf1:.4f}")
    print(f"  Macro-Precision : {prec:.4f}")
    print(f"  Macro-Recall    : {rec:.4f}")
    if extra:
        for k, v in extra.items():
            unit = " ms/sample" if "ms" in k else (" GB" if "gb" in k else (" MB" if "mb" in k else ""))
            print(f"  {k:25s}: {v}{unit}")
    print(f"\n{report}")
    print("Confusion Matrix:")
    print(f"  (rows=true, cols=pred) | {' | '.join(LABEL_NAMES)}")
    print(cm)

    result = {
        "name"             : name,
        "accuracy"         : round(acc,  4),
        "macro_f1"         : round(mf1,  4),
        "weighted_f1"      : round(wf1,  4),
        "macro_precision"  : round(prec, 4),
        "macro_recall"     : round(rec,  4),
        "confusion_matrix" : cm.tolist(),
    }
    if extra:
        result.update(extra)
    return result


# ── Grafikler ─────────────────────────────────────────────────────────────

def plot_comparison(results: list, out_dir: str):
    names   = [r["name"].split("/")[-1][:20] for r in results]
    metrics = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    labels  = ["Accuracy", "Macro-F1", "Weighted-F1", "Macro-Prec", "Macro-Rec"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9B59B6"]

    x, width = np.arange(len(names)), 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (m, label, c) in enumerate(zip(metrics, labels, colors)):
        vals = [r.get(m, 0) for r in results]
        bars = ax.bar(x + i * width, vals, width, label=label, color=c)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Karsilastirmasi -- 3-Class Third-Party Breach Siniflandirma")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Grafik kaydedildi: {path}")


def plot_efficiency(results: list, out_dir: str):
    """Model boyutu vs Macro-F1 scatter grafigi."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        if "disk_size_mb" in r:
            ax.scatter(r["disk_size_mb"], r["macro_f1"], s=120, zorder=3)
            ax.annotate(
                r["name"].split("/")[-1][:15],
                (r["disk_size_mb"], r["macro_f1"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
    ax.set_xlabel("Model Disk Boyutu (MB)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Performans - Verimlilik Dengesi")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "efficiency_plot.png")
    plt.savefig(path, dpi=150)
    print(f"Verimlilik grafigi kaydedildi: {path}")


def plot_confusion(result: dict, out_dir: str):
    cm = np.array(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    short = result["name"].split("/")[-1][:20]
    ax.set_title(f"Confusion Matrix — {short}")
    plt.tight_layout()
    safe_name = short.replace(" ", "_").replace("/", "_")
    path = os.path.join(out_dir, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=130)
    print(f"Confusion matrix kaydedildi: {path}")


# ── Hata analizi ──────────────────────────────────────────────────────────

def error_analysis(model_dir: str, test_csv: str, out_dir: str, batch_size: int = 16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True
    )
    model.eval()

    texts, labels = load_csv(test_csv)
    errors = []

    for i, (text, label) in enumerate(zip(texts, labels)):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        pred = int(torch.argmax(logits))
        conf = float(torch.softmax(logits, dim=-1).max())
        if pred != label:
            errors.append({
                "idx"        : i,
                "true_label" : label,
                "true_name"  : LABEL_NAMES[label],
                "pred_label" : pred,
                "pred_name"  : LABEL_NAMES[pred],
                "confidence" : round(conf, 4),
                "text_snippet": text[:300],
            })

    print(f"\nHata Analizi: {len(errors)}/{len(texts)} yanlis siniflandirma "
          f"({len(errors)/len(texts)*100:.1f}%)")

    # Hangi siniflar karisiyor?
    from collections import Counter
    pair_counts = Counter((e["true_name"][:10], e["pred_name"][:10]) for e in errors)
    print("  En sik karisan sinif ciftleri:")
    for (t, p), n in pair_counts.most_common(5):
        print(f"    True={t} -> Pred={p}: {n} kez")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "error_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"  Hata analizi kaydedildi: {path}")
    return errors


# ── Arguman ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  type=str, default=None)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--zero_shot",  action="store_true")
    p.add_argument("--compare",    nargs="+", default=None)
    p.add_argument("--error_analysis", action="store_true")
    p.add_argument("--test_csv",   type=str, default="data/test.csv")
    p.add_argument("--out_dir",    type=str, default="outputs")
    return p.parse_args()


# ── Ana akis ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.compare:
        all_results = []
        for model_dir in args.compare:
            res = evaluate_finetuned(model_dir, args.test_csv)
            all_results.append(res)
            plot_confusion(res, args.out_dir)

        print("\n" + "=" * 75)
        print("  OZET KARSILASTIRMA TABLOSU")
        print("=" * 75)
        hdr = f"{'Model':<22} {'Acc':>6} {'MacF1':>7} {'WgtF1':>7} {'Prec':>6} {'Rec':>6} {'ms/s':>6} {'MB':>6}"
        print(hdr)
        print("-" * 75)
        for r in all_results:
            name = r["name"].split("/")[-1][:21]
            ms   = f"{r.get('inference_ms_per_sample', '-'):>6.1f}" if isinstance(r.get('inference_ms_per_sample'), float) else "    -"
            mb   = f"{r.get('disk_size_mb', '-'):>6.0f}" if isinstance(r.get('disk_size_mb'), float) else "    -"
            print(f"{name:<22} {r['accuracy']:>6.4f} {r['macro_f1']:>7.4f} "
                  f"{r['weighted_f1']:>7.4f} {r['macro_precision']:>6.4f} "
                  f"{r['macro_recall']:>6.4f} {ms} {mb}")

        plot_comparison(all_results, args.out_dir)
        plot_efficiency(all_results, args.out_dir)

        summary = os.path.join(args.out_dir, "comparison_results.json")
        with open(summary, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSonuclar: {summary}")

    elif args.zero_shot and args.model_name:
        evaluate_zero_shot(args.model_name, args.test_csv)

    elif args.model_dir:
        res = evaluate_finetuned(args.model_dir, args.test_csv)
        plot_confusion(res, args.out_dir)
        if args.error_analysis:
            error_analysis(args.model_dir, args.test_csv, args.out_dir)

    else:
        print("--model_dir, --compare veya (--model_name + --zero_shot) belirtin.")


if __name__ == "__main__":
    main()
