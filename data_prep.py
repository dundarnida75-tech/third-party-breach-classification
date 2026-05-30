"""
data_prep.py
============
Third-Party Breach Haber Sınıflandırma Projesi — Veri Hazırlama

Kullanım:
    python data_prep.py

Çıktılar:
    data/train.csv
    data/val.csv
    data/test.csv
    data/dataset_stats.json
"""

import json
import re
import os
import random
import csv
import math

# ── Ayarlar ──────────────────────────────────────────────────────────────────
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# title + content birleştirme için max karakter (~512 token × 4 char)
MAX_CHARS = 2048

TP_FILE  = "third_party_news.json"
NTP_FILE = "not_third_party_news.json"
OUT_DIR  = "data"


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Fazladan boşluk ve satır sonlarını temizle."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_input(title: str, content: str, max_chars: int = MAX_CHARS) -> str:
    """
    title ve content'i birleştir, modele verilecek giriş metnini oluştur.
    Format: 'Title: <title> [SEP] Content: <content>'
    """
    title   = clean_text(title)
    content = clean_text(content)
    combined = f"Title: {title} [SEP] Content: {content}"
    return combined[:max_chars]


def load_json(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_split(data: list, train_r: float, val_r: float, seed: int):
    """Sınıf etiketine göre stratified train/val/test split."""
    random.seed(seed)

    class0 = [d for d in data if d["label"] == 0]
    class1 = [d for d in data if d["label"] == 1]

    def split_class(items):
        random.shuffle(items)
        n = len(items)
        n_train = math.floor(n * train_r)
        n_val   = math.floor(n * val_r)
        return (
            items[:n_train],
            items[n_train:n_train + n_val],
            items[n_train + n_val:]
        )

    tr0, v0, te0 = split_class(class0)
    tr1, v1, te1 = split_class(class1)

    train = tr0 + tr1
    val   = v0  + v1
    test  = te0 + te1

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_csv(data: list, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "url"])
        writer.writeheader()
        writer.writerows(data)
    print(f"  Kaydedildi: {filepath}  ({len(data)} ornek)")


def save_json(obj, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Third-Party Breach -- Veri Hazirlama")
    print("=" * 60)

    # 1. Yukle
    tp_raw  = load_json(TP_FILE)
    ntp_raw = load_json(NTP_FILE)
    print(f"\nYuklendi:")
    print(f"  Pozitif (third-party breach) : {len(tp_raw):>4} haber")
    print(f"  Negatif (diger)              : {len(ntp_raw):>4} haber")

    # 2. Etiketle ve birlestir
    dataset = []

    for item in tp_raw:
        dataset.append({
            "text" : build_input(item.get("title", ""), item.get("content", "")),
            "label": 1,
            "url"  : item.get("url", ""),
        })

    for item in ntp_raw:
        dataset.append({
            "text" : build_input(item.get("title", ""), item.get("content", "")),
            "label": 0,
            "url"  : item.get("url", ""),
        })

    print(f"\nToplam birlestirilen veri: {len(dataset)} ornek")

    # 3. Bos/cok kisa metin kontrolu
    before = len(dataset)
    dataset = [d for d in dataset if len(d["text"]) > 50]
    dropped = before - len(dataset)
    if dropped:
        print(f"  Cok kisa metin nedeniyle {dropped} ornek kaldirildi.")

    # 4. Split
    train, val, test = stratified_split(dataset, TRAIN_RATIO, VAL_RATIO, SEED)

    print(f"\nSplit (stratified, seed={SEED}):")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_pos = sum(d["label"] == 1 for d in split)
        n_neg = sum(d["label"] == 0 for d in split)
        print(f"  {name:5s}: {len(split):>4} ornek  (pos={n_pos}, neg={n_neg})")

    # 5. Kaydet
    print(f"\nCSV dosyalari kaydediliyor --> {OUT_DIR}/")
    save_csv(train, os.path.join(OUT_DIR, "train.csv"))
    save_csv(val,   os.path.join(OUT_DIR, "val.csv"))
    save_csv(test,  os.path.join(OUT_DIR, "test.csv"))

    # 6. Istatistik
    avg_len = sum(len(d["text"]) for d in dataset) / len(dataset)
    stats = {
        "total"          : len(dataset),
        "positive"       : sum(d["label"] == 1 for d in dataset),
        "negative"       : sum(d["label"] == 0 for d in dataset),
        "train"          : len(train),
        "val"            : len(val),
        "test"           : len(test),
        "avg_text_chars" : round(avg_len, 1),
        "seed"           : SEED,
    }
    save_json(stats, os.path.join(OUT_DIR, "dataset_stats.json"))
    print(f"\nOrtalama metin uzunlugu: {stats['avg_text_chars']} karakter")
    print("\nVeri hazirlama tamamlandi!")


if __name__ == "__main__":
    main()
