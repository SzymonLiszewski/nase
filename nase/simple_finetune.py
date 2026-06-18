"""Standalone fine-tuning script for aiana94/NaSE on a local TSV dataset.

This script intentionally does not use the project's Hydra/NASE training stack.
It trains a masked language model directly with Hugging Face Transformers on a
monolingual TSV file that contains a `text` column.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone fine-tuning for aiana94/NaSE")
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/custom_local_news/train.tsv",
        help="Path to a TSV file with a `text` column.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="aiana94/NaSE",
        help="Base Hugging Face model to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/simple_finetune_nase",
        help="Where to save checkpoints and the final model.",
    )
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max sequence length.")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length used for MLM training.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Masking probability.")
    parser.add_argument(
        "--eval_every_steps_initial",
        type=int,
        default=1,
        help="How often to run eval during the first part of training.",
    )
    parser.add_argument(
        "--eval_initial_steps",
        type=int,
        default=20,
        help="For how many initial steps to use the denser eval cadence.",
    )
    parser.add_argument(
        "--eval_every_steps_later",
        type=int,
        default=25,
        help="Eval cadence after the initial warmup window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional hard limit on training steps. Use 0 to rely on epochs.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Run a tiny eval split if the dataset is large enough.",
    )
    return parser.parse_args()


def load_text_dataset(data_file: str) -> Dataset:
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset = load_dataset(
        "csv",
        data_files=str(path),
        delimiter="\t",
        split="train",
    )

    if "text" not in dataset.column_names:
        raise ValueError("Expected a TSV file with a `text` column.")

    dataset = dataset.filter(lambda example: bool(str(example["text"]).strip()))
    return dataset


def tokenize_and_group_texts(dataset: Dataset, tokenizer, max_length: int, block_size: int) -> Dataset:
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    if len(tokenized) == 0:
        raise ValueError("No training examples left after tokenization.")
    return tokenized


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_masked = 0
    correct_top1 = 0
    correct_top5 = 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]

        total_loss += float(loss.item())
        total_batches += 1

        mask = labels != -100
        if mask.any():
            masked_logits = logits[mask]
            masked_labels = labels[mask]
            top1_predictions = masked_logits.argmax(dim=-1)
            top5_predictions = masked_logits.topk(k=min(5, masked_logits.shape[-1]), dim=-1).indices

            correct_top1 += int((top1_predictions == masked_labels).sum().item())
            correct_top5 += int(
                top5_predictions.eq(masked_labels.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            total_masked += int(masked_labels.numel())

    model.train()

    average_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(min(average_loss, 20.0))
    top1_accuracy = correct_top1 / max(1, total_masked)
    top5_accuracy = correct_top5 / max(1, total_masked)

    return {
        "eval_loss": average_loss,
        "eval_perplexity": perplexity,
        "eval_top1_accuracy": top1_accuracy,
        "eval_top5_accuracy": top5_accuracy,
    }


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    dataset = load_text_dataset(args.data_file)
    if len(dataset) < 2:
        train_dataset = dataset
        eval_dataset = None
    else:
        split = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"] if args.do_eval else None

    train_dataset = tokenize_and_group_texts(train_dataset, tokenizer, args.max_length, args.block_size)
    if eval_dataset is not None:
        eval_dataset = tokenize_and_group_texts(eval_dataset, tokenizer, args.max_length, args.block_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    stop_training = False

    if eval_loader is not None:
        metrics = evaluate(model, eval_loader, device)
        print(
            "initial_eval "
            + " ".join(
                f"{name}={value:.4f}" for name, value in metrics.items()
            )
        )

    for epoch in range(int(args.num_train_epochs) or 1):
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            print(f"step={global_step} loss={loss.item():.4f}")

            should_eval = False
            if eval_loader is not None:
                if global_step <= args.eval_initial_steps:
                    should_eval = global_step % max(1, args.eval_every_steps_initial) == 0
                else:
                    should_eval = global_step % max(1, args.eval_every_steps_later) == 0

            if should_eval and eval_loader is not None:
                metrics = evaluate(model, eval_loader, device)
                print(
                    f"eval@step={global_step} "
                    + " ".join(
                        f"{name}={value:.4f}" for name, value in metrics.items()
                    )
                )

            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()