"""Standalone recommendation fine-tuning for aiana94/NaSE.

This script trains a simple click-ranking model on MIND-style data:
- news.tsv with news text
- behaviors.tsv with user histories and impression labels

It is intentionally independent of the project's Hydra/NASE training stack.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import to_dense_batch
from torchmetrics.retrieval import RetrievalAUROC, RetrievalMRR, RetrievalNormalizedDCG
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone recommendation fine-tuning for aiana94/NaSE")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/custom_recommendation",
        help="Folder containing news.tsv and behaviors.tsv in MIND format.",
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
        default="outputs/simple_recommendation_finetune_nase",
        help="Where to save the final model.",
    )
    parser.add_argument("--max_length", type=int, default=96, help="Tokenizer max sequence length.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional hard limit on training steps.")
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Fraction of behaviors used for validation if no time split is given.",
    )
    parser.add_argument(
        "--valid_time_split",
        type=str,
        default="",
        help="Optional time cutoff (e.g. '11/14/2019 12:00:00 AM'). If set, train uses rows before cutoff.",
    )
    parser.add_argument(
        "--max_history_len",
        type=int,
        default=50,
        help="Maximum number of clicked items from the user history to use.",
    )
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
        "--freeze_encoder",
        action="store_true",
        help="Freeze the encoder and train only the ranking head.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision on CUDA to reduce memory usage.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_news_texts(data_dir: str) -> dict[str, str]:
    news_path = Path(data_dir) / "news.tsv"
    if not news_path.exists():
        raise FileNotFoundError(f"Missing news.tsv: {news_path}")

    columns = [
        "nid",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    news = pd.read_table(news_path, header=None, names=columns, usecols=range(len(columns)))
    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")
    news["text"] = (news["title"].astype(str) + " " + news["abstract"].astype(str)).str.strip()
    news.loc[news["text"] == "", "text"] = news["title"]
    return dict(zip(news["nid"].astype(str), news["text"].astype(str)))


def load_behaviors(data_dir: str, valid_ratio: float, valid_time_split: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    behaviors_path = Path(data_dir) / "behaviors.tsv"
    if not behaviors_path.exists():
        raise FileNotFoundError(f"Missing behaviors.tsv: {behaviors_path}")

    columns = ["impid", "uid", "time", "history", "impressions"]
    behaviors = pd.read_table(behaviors_path, header=None, names=columns, usecols=range(len(columns)))
    behaviors["time"] = pd.to_datetime(behaviors["time"], errors="coerce")
    behaviors["history"] = behaviors["history"].fillna("").astype(str).str.split()
    behaviors["impressions"] = behaviors["impressions"].fillna("").astype(str).str.split()

    behaviors = behaviors[behaviors["history"].map(len) > 0].reset_index(drop=True)
    behaviors = behaviors[behaviors["impressions"].map(len) > 0].reset_index(drop=True)

    def split_impressions(values: Iterable[str]) -> tuple[list[str], list[int]]:
        candidates: list[str] = []
        labels: list[int] = []
        for impression in values:
            candidate, label = impression.rsplit("-", 1)
            candidates.append(candidate)
            labels.append(int(label))
        return candidates, labels

    split_results = behaviors["impressions"].apply(split_impressions)
    behaviors["candidates"] = split_results.apply(lambda item: item[0])
    behaviors["labels"] = split_results.apply(lambda item: item[1])

    behaviors = behaviors.drop(columns=["impressions"])

    if valid_time_split:
        cutoff = pd.to_datetime(valid_time_split, errors="raise")
        train_df = behaviors.loc[behaviors["time"] < cutoff].reset_index(drop=True)
        valid_df = behaviors.loc[behaviors["time"] >= cutoff].reset_index(drop=True)
    else:
        shuffled = behaviors.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        valid_size = max(1, int(len(shuffled) * valid_ratio)) if len(shuffled) > 1 else 0
        valid_df = shuffled.iloc[:valid_size].reset_index(drop=True) if valid_size else shuffled.iloc[0:0].copy()
        train_df = shuffled.iloc[valid_size:].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("No training behaviors left after the split.")

    return train_df, valid_df


@dataclass
class RecommendationExample:
    history: list[str]
    candidates: list[str]
    labels: list[int]


class RecommendationDataset(Dataset):
    def __init__(self, behaviors: pd.DataFrame, max_history_len: int) -> None:
        self.behaviors = behaviors
        self.max_history_len = max_history_len

    def __len__(self) -> int:
        return len(self.behaviors)

    def __getitem__(self, index: int) -> RecommendationExample:
        row = self.behaviors.iloc[index]
        return RecommendationExample(
            history=list(row["history"][: self.max_history_len]),
            candidates=list(row["candidates"]),
            labels=list(row["labels"]),
        )


class RecommendationCollator:
    def __init__(self, tokenizer, news_texts: dict[str, str], max_length: int) -> None:
        self.tokenizer = tokenizer
        self.news_texts = news_texts
        self.max_length = max_length

    def _lookup(self, news_id: str) -> str:
        return self.news_texts.get(str(news_id), "")

    def __call__(self, batch: list[RecommendationExample]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        history_texts: list[str] = []
        candidate_texts: list[str] = []
        batch_hist: list[int] = []
        batch_cand: list[int] = []
        labels: list[int] = []

        for item_index, item in enumerate(batch):
            history_texts.extend(self._lookup(news_id) for news_id in item.history)
            batch_hist.extend([item_index] * len(item.history))

            candidate_texts.extend(self._lookup(news_id) for news_id in item.candidates)
            batch_cand.extend([item_index] * len(item.candidates))
            labels.extend(item.labels)

        hist_enc = self.tokenizer(
            history_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        cand_enc = self.tokenizer(
            candidate_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )

        return {
            "batch_hist": torch.tensor(batch_hist, dtype=torch.long),
            "batch_cand": torch.tensor(batch_cand, dtype=torch.long),
            "x_hist": {"text": hist_enc},
            "x_cand": {"text": cand_enc},
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class SimpleRecommendationModel(nn.Module):
    def __init__(self, model_name: str, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.scorer = nn.Identity()
        self.loss_fn = nn.BCEWithLogitsLoss()

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def encode(self, batch_encoding: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**batch_encoding)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_vec = self.encode(batch["x_hist"]["text"])
        hist_vec_agg, mask_hist = to_dense_batch(hist_vec, batch["batch_hist"])

        cand_vec = self.encode(batch["x_cand"]["text"])
        cand_vec_agg, mask_cand = to_dense_batch(cand_vec, batch["batch_cand"])

        hist_size = mask_hist.sum(dim=1).clamp_min(1).unsqueeze(-1)
        user_vec = hist_vec_agg.sum(dim=1) / hist_size

        scores = torch.bmm(user_vec.unsqueeze(1), cand_vec_agg.permute(0, 2, 1)).squeeze(1)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        return scores, y_true, mask_cand

    def loss(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor:
        scores, y_true, mask_cand = self.forward(batch)
        return self.loss_fn(scores[mask_cand], y_true[mask_cand])


@torch.no_grad()
def evaluate(model: SimpleRecommendationModel, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    mrr = RetrievalMRR().to(device)
    auroc = RetrievalAUROC().to(device)
    ndcg10 = RetrievalNormalizedDCG(top_k=10).to(device)

    for batch in dataloader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else {subkey: subvalue.to(device) for subkey, subvalue in value.items()}
            for key, value in batch.items()
        }
        scores, y_true, mask_cand = model.forward(batch)
        loss = model.loss_fn(scores[mask_cand], y_true[mask_cand])

        batch_preds = torch.cat([scores[index][mask_cand[index]] for index in range(mask_cand.shape[0])], dim=0)
        batch_targets = torch.cat([y_true[index][mask_cand[index]] for index in range(mask_cand.shape[0])], dim=0)
        candidate_counts = torch.stack([mask_cand[index].sum() for index in range(mask_cand.shape[0])]).long()
        batch_indexes = torch.arange(candidate_counts.shape[0], device=device).repeat_interleave(candidate_counts)

        total_loss += float(loss.item())
        total_batches += 1

        mrr.update(batch_preds, batch_targets, batch_indexes)
        auroc.update(batch_preds, batch_targets, batch_indexes)
        ndcg10.update(batch_preds, batch_targets, batch_indexes)

    average_loss = total_loss / max(1, total_batches)
    return {
        "eval_loss": average_loss,
        "eval_perplexity": math.exp(min(average_loss, 20.0)),
        "eval_mrr": float(mrr.compute().item()),
        "eval_auroc": float(auroc.compute().item()),
        "eval_ndcg10": float(ndcg10.compute().item()),
    }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else {subkey: subvalue.to(device) for subkey, subvalue in value.items()}
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    print(f"Using device: {device}")

    news_texts = load_news_texts(args.data_dir)
    train_behaviors, valid_behaviors = load_behaviors(
        args.data_dir,
        valid_ratio=args.valid_ratio,
        valid_time_split=args.valid_time_split,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = RecommendationDataset(train_behaviors, max_history_len=args.max_history_len)
    valid_dataset = RecommendationDataset(valid_behaviors, max_history_len=args.max_history_len) if len(valid_behaviors) else None

    collator = RecommendationCollator(tokenizer=tokenizer, news_texts=news_texts, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        if valid_dataset is not None
        else None
    )

    model = SimpleRecommendationModel(args.model_name, freeze_encoder=args.freeze_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    if valid_loader is not None:
        metrics = evaluate(model, valid_loader, device)
        print(
            "initial_eval "
            + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )

    model.train()
    stop_training = False

    for epoch in range(args.num_train_epochs):
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model.loss(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            print(f"step={global_step} loss={loss.item():.4f}")

            should_eval = False
            if valid_loader is not None:
                if global_step <= args.eval_initial_steps:
                    should_eval = global_step % max(1, args.eval_every_steps_initial) == 0
                else:
                    should_eval = global_step % max(1, args.eval_every_steps_later) == 0

            if should_eval and valid_loader is not None:
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

    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()