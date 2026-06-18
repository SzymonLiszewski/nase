"""Standalone text-classification fine-tuning for the English subset of custom_local_news.

This script does not use the project's Hydra/NASE training stack.
It fine-tunes a Hugging Face sequence classification model on the TSV files in
`data/custom_local_news/eng/`.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone classification fine-tuning for aiana94/NaSE")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/custom_local_news/eng",
        help="Folder containing train.tsv/dev.tsv/test.tsv and labels.txt.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="aiana94/NaSE",
        help="Base Hugging Face encoder to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/simple_classification_finetune_nase",
        help="Where to save the final model.",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional hard limit on training steps.")
    parser.add_argument(
        "--eval_every_steps_initial",
        type=int,
        default=1,
        help="Eval cadence during the first part of training.",
    )
    parser.add_argument(
        "--eval_initial_steps",
        type=int,
        default=20,
        help="Use the dense eval cadence for the first N steps.",
    )
    parser.add_argument(
        "--eval_every_steps_later",
        type=int,
        default=25,
        help="Eval cadence after the initial warmup window.",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=0,
        help="Optional cap on the number of train rows to load. Use 0 for all rows.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(data_dir: str) -> list[str]:
    labels_path = Path(data_dir) / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.txt: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("labels.txt is empty.")
    return labels


def load_split(data_dir: str, split_name: str, label_to_id: dict[str, int], limit_rows: int = 0) -> pd.DataFrame:
    split_path = Path(data_dir) / f"{split_name}.tsv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_name}.tsv: {split_path}")

    df = pd.read_table(split_path)
    expected_columns = {"category", "headline", "text", "url"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{split_name}.tsv is missing columns: {sorted(missing_columns)}")

    if limit_rows and limit_rows > 0:
        df = df.iloc[:limit_rows].copy()

    df = df.dropna(subset=["category", "headline", "text"]).copy()
    df["label"] = df["category"].map(label_to_id)
    if df["label"].isna().any():
        unknown = sorted(set(df.loc[df["label"].isna(), "category"].astype(str).tolist()))
        raise ValueError(f"Unknown labels in {split_name}.tsv: {unknown}")

    df["label"] = df["label"].astype(int)
    df["input_text"] = (df["headline"].astype(str) + " [SEP] " + df["text"].astype(str)).str.strip()
    df = df[df["input_text"].str.len() > 0].reset_index(drop=True)
    return df


class ClassificationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        return {"text": row["input_text"], "label": int(row["label"])}


class ClassificationCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        texts = [str(item["text"]) for item in batch]
        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        encoded["labels"] = labels
        return encoded


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_examples = 0
    correct = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)

        total_loss += float(loss.item())
        total_batches += 1
        total_examples += int(batch["labels"].shape[0])
        correct += int((predictions == batch["labels"]).sum().item())
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    model.train()

    accuracy = correct / max(1, total_examples)
    macro_f1 = macro_f1_score(all_predictions, all_labels, num_classes=model.config.num_labels)
    average_loss = total_loss / max(1, total_batches)

    return {
        "eval_loss": average_loss,
        "eval_accuracy": accuracy,
        "eval_macro_f1": macro_f1,
    }


def macro_f1_score(predictions: list[int], labels: list[int], num_classes: int) -> float:
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for prediction, label in zip(predictions, labels):
        if prediction == label:
            tp[label] += 1
        else:
            fp[prediction] += 1
            fn[label] += 1

    f1_values = []
    for class_index in range(num_classes):
        precision_denominator = tp[class_index] + fp[class_index]
        recall_denominator = tp[class_index] + fn[class_index]
        precision = tp[class_index] / precision_denominator if precision_denominator else 0.0
        recall = tp[class_index] / recall_denominator if recall_denominator else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return sum(f1_values) / max(1, num_classes)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels = load_labels(args.data_dir)
    label_to_id = {label: index for index, label in enumerate(labels)}

    train_df = load_split(args.data_dir, "train", label_to_id, limit_rows=args.max_train_rows)
    valid_df = load_split(args.data_dir, "dev", label_to_id)
    test_df = load_split(args.data_dir, "test", label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    ).to(device)

    train_dataset = ClassificationDataset(train_df)
    valid_dataset = ClassificationDataset(valid_df)
    test_dataset = ClassificationDataset(test_df)

    collator = ClassificationCollator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    metrics = evaluate(model, valid_loader, device)
    print("initial_eval " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items()))

    model.train()
    stop_training = False

    for epoch in range(args.num_train_epochs):
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            print(f"step={global_step} loss={loss.item():.4f}")

            should_eval = False
            if global_step <= args.eval_initial_steps:
                should_eval = global_step % max(1, args.eval_every_steps_initial) == 0
            else:
                should_eval = global_step % max(1, args.eval_every_steps_later) == 0

            if should_eval:
                metrics = evaluate(model, valid_loader, device)
                print(
                    f"eval@step={global_step} "
                    + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
                )

            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break

    final_valid = evaluate(model, valid_loader, device)
    final_test = evaluate(model, test_loader, device)
    print("final_valid " + " ".join(f"{name}={value:.4f}" for name, value in final_valid.items()))
    print("final_test " + " ".join(f"{name}={value:.4f}" for name, value in final_test.items()))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()