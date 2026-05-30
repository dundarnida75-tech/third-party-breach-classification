"""
train.py  (v2 -- 3-class)
==========================
Third-Party Breach Haber Siniflandirma -- LoRA / QLoRA Fine-Tuning

Siniflar:
  0 -> Ihlal Yok        (genel siber guvenlik haberleri)
  1 -> Dogrudan Ihlal   (kurum dogrudan hedef alindi)
  2 -> 3. Taraf Ihlali  (vendor / supply-chain kaynakli)

Kullanim ornekleri:
    python train.py --model smollm2
    python train.py --model tinyllama
    python train.py --model qwen --quantize
    python train.py --model gemma --quantize
"""

import argparse, os, json, csv, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ── Model katalogu ─────────────────────────────────────────────────────────

MODEL_CATALOG = {
    "smollm2"  : "HuggingFaceTB/SmolLM2-360M",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen"     : "Qwen/Qwen2.5-1.5B",
    "gemma"    : "google/gemma-4-e2b-it",
}

NUM_LABELS = 3
LABEL_NAMES = ["Ihlal Yok", "Dogrudan Ihlal", "3. Taraf Ihlali"]

# ── Dataset ───────────────────────────────────────────────────────────────

class BreachDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "text" : row["text"],
                    "label": int(row["label"]),
                })
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels"        : torch.tensor(s["label"], dtype=torch.long),
        }

    def get_labels(self):
        return [s["label"] for s in self.samples]


# ── Class-weighted Trainer ─────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Sinif dengesizligini gidermek icin agirlikli cross-entropy."""
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrikler ──────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy"  : accuracy_score(labels, preds),
        "macro_f1"  : f1_score(labels, preds, average="macro",    zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro",    zero_division=0),
        "macro_recall"   : recall_score(labels,    preds, average="macro",    zero_division=0),
    }


# ── Arguman ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      choices=list(MODEL_CATALOG.keys()))
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--run_name",   type=str, default=None)
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--quantize",   action="store_true", help="4-bit QLoRA")
    p.add_argument("--max_length", type=int,   default=512)
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--lora_r",     type=int,   default=16)
    p.add_argument("--lora_alpha", type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ── Ana akis ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.model_name:
        model_name = args.model_name
        short_name = args.model_name.split("/")[-1]
    elif args.model:
        model_name = MODEL_CATALOG[args.model]
        short_name = args.model
    else:
        raise ValueError("--model veya --model_name belirtilmeli.")

    run_name   = args.run_name or short_name
    output_dir = os.path.join(args.output_dir, run_name)

    print(f"\n{'='*65}")
    print(f"  Model     : {model_name}")
    print(f"  Sinif     : {NUM_LABELS} ({', '.join(LABEL_NAMES)})")
    print(f"  Quantize  : {args.quantize}")
    print(f"  Cikti     : {output_dir}")
    print(f"{'='*65}\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    train_ds = BreachDataset(
        os.path.join(args.data_dir, "train.csv"), tokenizer, args.max_length
    )
    val_ds = BreachDataset(
        os.path.join(args.data_dir, "val.csv"), tokenizer, args.max_length
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Sinif agirliklari (dengesizlik icin)
    all_labels = train_ds.get_labels()
    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=all_labels)
    class_weights = torch.tensor(cw, dtype=torch.float)
    print(f"Class weights: " + ", ".join(
        f"C{i}={w:.3f}" for i, w in enumerate(cw)
    ))

    # Model
    bnb_config = None
    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Sinif isimlerini kaydet
    model.config.id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    model.config.label2id = {n: i for i, n in enumerate(LABEL_NAMES)}
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA
    if args.quantize:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Egitim
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=20,
        seed=args.seed,
        fp16=True,
        report_to="none",
        run_name=run_name,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    val_results = trainer.evaluate()
    val_results["train_time_sec"] = round(train_time, 1)
    print("\nValidation sonuclari:")
    for k, v in val_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "val_results.json"), "w") as f:
        json.dump(val_results, f, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Model boyutunu kaydet
    size_mb = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)
    meta = {
        "model_name": model_name,
        "run_name"  : run_name,
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "model_size_mb": round(size_mb, 1),
        "quantized": args.quantize,
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel kaydedildi: {output_dir}")
    print(f"Model boyutu (RAM): {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
