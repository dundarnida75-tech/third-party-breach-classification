"""
train.py
========
Third-Party Breach Haber Siniflandirma -- LoRA / QLoRA Fine-Tuning

Kullanim ornekleri:
    # SmolLM2-360M -- LoRA
    python train.py --model smollm2

    # TinyLlama-1.1B -- LoRA
    python train.py --model tinyllama

    # Qwen2.5-1.5B -- QLoRA (4-bit)
    python train.py --model qwen --quantize

    # Gemma-4 E2B -- QLoRA (4-bit)
    python train.py --model gemma --quantize

    # Ozel model
    python train.py --model_name "HuggingFaceTB/SmolLM2-360M" --run_name my_run
"""

import argparse
import os
import json
import csv
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Model katalogu ────────────────────────────────────────────────────────────

MODEL_CATALOG = {
    "smollm2"  : "HuggingFaceTB/SmolLM2-360M",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen"     : "Qwen/Qwen2.5-1.5B",
    "gemma"    : "google/gemma-4-e2b-it",
}

# ── Dataset sinifi ────────────────────────────────────────────────────────────

class BreachDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "text" : row["text"],
                    "label": int(row["label"]),
                })
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels"        : torch.tensor(sample["label"], dtype=torch.long),
        }


# ── Metrik hesaplama ──────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float), dim=-1)[:, 1].numpy()
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall"   : recall_score(labels, preds, average="macro", zero_division=0),
        "roc_auc"  : roc_auc_score(labels, probs),
    }


# ── Arguman ayristirici ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA fine-tuning for third-party breach classification"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CATALOG.keys()),
        help="Katalogdaki model kisaltmasi",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model adi (katalog disinda model icin)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Deney adi (cikti klasoru icin)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="CSV split dosyalarinin dizini",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Model ve log cikti dizini",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="4-bit QLoRA kullan (buyuk modeller icin)",
    )
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


# ── Ana akis ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Model adini belirle
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

    print(f"\n{'='*60}")
    print(f"  Model     : {model_name}")
    print(f"  Quantize  : {args.quantize}")
    print(f"  Cikti     : {output_dir}")
    print(f"{'='*60}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = BreachDataset(
        os.path.join(args.data_dir, "train.csv"), tokenizer, args.max_length
    )
    val_ds = BreachDataset(
        os.path.join(args.data_dir, "val.csv"), tokenizer, args.max_length
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model yukle ───────────────────────────────────────────────────────────
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
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if args.quantize:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Hedef moduller -- model mimarisine gore otomatik secilir
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Egitim ayarlari ───────────────────────────────────────────────────────
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
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=20,
        seed=args.seed,
        fp16=True,
        report_to="none",   # W&B icin "wandb" yapabilirsiniz
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Egit ──────────────────────────────────────────────────────────────────
    trainer.train()

    # ── Val metriklerini kaydet ───────────────────────────────────────────────
    val_results = trainer.evaluate()
    print("\nValidation sonuclari:")
    for k, v in val_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_path = os.path.join(output_dir, "val_results.json")
    with open(results_path, "w") as f:
        json.dump(val_results, f, indent=2)
    print(f"\nSonuclar kaydedildi: {results_path}")

    # ── Modeli kaydet ─────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model kaydedildi: {output_dir}")


if __name__ == "__main__":
    main()
