"""
data_prep.py  (v2 — 3-class)
=============================
Third-Party Breach Haber Siniflandirma -- Veri Hazirlama

Sinif yapisi (hoca onayladi):
  0  -> Ihlal Yok          : Genel siber guvenlik haberleri
  1  -> Dogrudan Ihlal     : Kurum dogrudan hedef alindi
  2  -> 3. Taraf Ihlali    : Tedarik zinciri / vendor kaynakli

Mantik:
  third_party_news.json    -> tamamen Class 2
  not_third_party_news.json -> heuristik ile Class 0 / Class 1'e ayrilir
    - Breach anahtar kelimeleri iceriyorsa -> Class 1 (Dogrudan Ihlal)
    - Icermiyorsa                          -> Class 0 (Ihlal Yok)

Split: %80 train / %10 val / %10 test (stratified)

Kullanim:
    python data_prep.py
    python data_prep.py --show_samples   # ornek satirlari yazdirir
"""

import json, re, os, random, csv, math, argparse

# ── Ayarlar ──────────────────────────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
MAX_CHARS   = 2048          # ~512 token

TP_FILE  = "third_party_news.json"   # -> Class 2
NTP_FILE = "not_third_party_news.json"  # -> Class 0 veya Class 1
OUT_DIR  = "data"

# Class 1 (Dogrudan Ihlal) icin anahtar kelimeler
# NOT: "third-party", "vendor breach" gibi ifadeler Class 2'ye isaret eder,
# bu nedenle bu liste sadece dogrudan ihlal sinyali tasiyanlar.
DIRECT_BREACH_KEYWORDS = [
    r"\bdata breach\b", r"\bdata leak\b", r"\bdata exposed\b",
    r"\bhacked\b", r"\bransomware\b", r"\bencrypted.*files\b",
    r"\bcyberattack\b", r"\bsecurity incident\b", r"\bcompromised\b",
    r"\bunauthorized access\b", r"\bcredentials.*stolen\b",
    r"\bpersonal.*information.*exposed\b", r"\brecords.*leaked\b",
    r"\battacked by\b", r"\bsuffered.*breach\b", r"\bvictim.*attack\b",
]

# Class 2 kacinilmaz gostergeler (bu varsa mutlaka Class 2 kalsin)
THIRD_PARTY_SIGNALS = [
    r"\bthird.party\b", r"\bsupply chain\b", r"\bvendor\b",
    r"\bservice provider\b", r"\bsoftware provider\b", r"\bsaas\b",
    r"\bpartner.*breach\b", r"\bbreach.*partner\b",
]


# ── Yardimci fonksiyonlar ──────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)        # HTML etiketleri
    text = re.sub(r'http\S+', '', text)         # URL
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_input(title: str, content: str) -> str:
    t = clean_text(title)
    c = clean_text(content)
    combined = f"Title: {t} [SEP] Content: {c}"
    return combined[:MAX_CHARS]


def has_pattern(text: str, patterns: list) -> bool:
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


def label_ntp(item: dict) -> int:
    """
    not_third_party haberini Class 0 veya Class 1 olarak etiketle.
    Oncelikle: eger third-party sinyali varsa Class 2 (veri kirliligi onlemi).
    Sonra: direct breach sinyali varsa Class 1, yoksa Class 0.
    """
    combined = (item.get("title", "") + " " + item.get("content", ""))
    # Guvenlik: eger 3. taraf sinyali varsa (hata olarak bu dosyaya girmis)
    if has_pattern(combined, THIRD_PARTY_SIGNALS):
        return 2
    if has_pattern(combined, DIRECT_BREACH_KEYWORDS):
        return 1
    return 0


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_split(data: list, train_r: float, val_r: float, seed: int):
    random.seed(seed)
    classes = sorted(set(d["label"] for d in data))
    splits = {c: [] for c in classes}
    for d in data:
        splits[d["label"]].append(d)

    train, val, test = [], [], []
    for c in classes:
        items = splits[c][:]
        random.shuffle(items)
        n = len(items)
        n_train = math.floor(n * train_r)
        n_val   = math.floor(n * val_r)
        train += items[:n_train]
        val   += items[n_train:n_train + n_val]
        test  += items[n_train + n_val:]

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


def save_json(obj, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def label_name(l: int) -> str:
    return {0: "Ihlal Yok", 1: "Dogrudan Ihlal", 2: "3. Taraf Ihlali"}.get(l, str(l))


# ── Ana akis ──────────────────────────────────────────────────────────────

def main(show_samples=False):
    print("=" * 65)
    print("  Third-Party Breach -- Veri Hazirlama (3-Class, v2)")
    print("=" * 65)

    tp_raw  = load_json(TP_FILE)
    ntp_raw = load_json(NTP_FILE)
    print(f"\nHam veri:")
    print(f"  third_party_news.json    : {len(tp_raw):>4} haber  -> Class 2")
    print(f"  not_third_party_news.json: {len(ntp_raw):>4} haber  -> Class 0 / 1")

    # -- Etiketle --
    dataset = []

    # Class 2: 3. taraf ihlali
    for item in tp_raw:
        text = build_input(item.get("title", ""), item.get("content", ""))
        if len(text) > 50:
            dataset.append({"text": text, "label": 2, "url": item.get("url", "")})

    # Class 0 / 1: heuristik
    ntp_labels = {0: 0, 1: 0, 2: 0}
    for item in ntp_raw:
        text  = build_input(item.get("title", ""), item.get("content", ""))
        lbl   = label_ntp(item)
        ntp_labels[lbl] += 1
        if len(text) > 50:
            dataset.append({"text": text, "label": lbl, "url": item.get("url", "")})

    print(f"\nnot_third_party heuristik dagilimi:")
    for l, n in sorted(ntp_labels.items()):
        print(f"  Class {l} ({label_name(l):20s}): {n:>4}")

    print(f"\nToplam veri seti: {len(dataset)} ornek")
    print(f"\nSinif dagilimi:")
    from collections import Counter
    cnt = Counter(d["label"] for d in dataset)
    for l in sorted(cnt):
        print(f"  Class {l} ({label_name(l):20s}): {cnt[l]:>4}  ({cnt[l]/len(dataset)*100:.1f}%)")

    # -- Split 80/10/10 --
    train, val, test = stratified_split(dataset, TRAIN_RATIO, VAL_RATIO, SEED)

    print(f"\nSplit (stratified, seed={SEED}, 80/10/10):")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        c = Counter(d["label"] for d in split)
        print(f"  {name:5s}: {len(split):>4} ornek  "
              f"[C0={c[0]} C1={c[1]} C2={c[2]}]")

    # -- Kaydet --
    print(f"\nKaydediliyor --> {OUT_DIR}/")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        save_csv(split, path)
        print(f"  {path}  ({len(split)} ornek)")

    # -- Istatistik --
    avg_len = sum(len(d["text"]) for d in dataset) / len(dataset)
    stats = {
        "total": len(dataset),
        "class_counts": {str(l): cnt[l] for l in sorted(cnt)},
        "train": len(train), "val": len(val), "test": len(test),
        "avg_text_chars": round(avg_len, 1),
        "split_ratio": "80/10/10",
        "seed": SEED,
        "note": (
            "Class 0: no breach, Class 1: direct breach (heuristic), "
            "Class 2: third-party breach"
        ),
    }
    save_json(stats, os.path.join(OUT_DIR, "dataset_stats.json"))
    print(f"\nOrtalama metin uzunlugu: {stats['avg_text_chars']} karakter")

    if show_samples:
        print("\n--- Ornek satirlar ---")
        for lbl in [0, 1, 2]:
            sample = next((d for d in dataset if d["label"] == lbl), None)
            if sample:
                print(f"\n[Class {lbl}] {label_name(lbl)}")
                print(sample["text"][:300])

    print("\nVeri hazirlama tamamlandi!")
    print("\n*** ONEMLI: not_third_party haberleri icin heuristik etiketleme yapildi.")
    print("    CTI analisti olarak Class 1 orneklerini gozden gecirmeniz onerilir.")
    print(f"    Gozden gecirmek icin: data/train.csv dosyasinda label=1 olan satirlari filtreleyin.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_samples", action="store_true")
    args = parser.parse_args()
    main(show_samples=args.show_samples)
