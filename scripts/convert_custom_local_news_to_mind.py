"""Convert a simple local news dump to MIND-style news.tsv + behaviors.tsv.

Expects:
  data/custom_local_news/shared_articles.csv/shared_articles.csv
  data/custom_local_news/users_interactions.csv/users_interactions.csv

Produces:
  <output_dir>/news.tsv
  <output_dir>/behaviors.tsv

The converter groups interactions by sessionId (falls back to personId when sessionId is missing),
orders them by timestamp and emits one behavior row per session that has at least one candidate.

Labels: 1 for actions in ACTION_POSITIVE (LIKE, BOOKMARK, FOLLOW, COMMENT CREATED),
0 for other viewed items. If a session has only positives (no negatives) we sample negative items
from popular articles to make training feasible.

This is a pragmatic, small utility for quick experiments. It is NOT a drop-in perfect MIND generator
— adjust sampling/label rules for your needs.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ACTION_POSITIVE = {"LIKE", "BOOKMARK", "FOLLOW", "COMMENT CREATED"}


def load_articles(path: Path) -> Dict[str, Dict[str, str]]:
    # read articles and return mapping contentId -> metadata dict
    df = pd.read_csv(path, dtype=str, header=0, keep_default_na=False)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    meta: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        cid = str(r.get("contentId", "")).strip()
        if not cid:
            continue
        title = str(r.get("title", "")).strip()
        text = str(r.get("text", "")).strip()
        url = str(r.get("url", "")).strip()
        lang = str(r.get("lang", "")).strip()
        meta[cid] = {"title": title, "abstract": text, "url": url, "lang": lang}
    return meta


def load_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, header=0, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    # ensure timestamp numeric
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
    return df


def build_behaviors(
    interactions: pd.DataFrame,
    articles_meta: Dict[str, Dict[str, str]],
    min_candidates: int = 5,
    sample_negative_pool: int = 100,
) -> List[Tuple[str, str, str, List[str]]]:
    # group by sessionId (fallback to personId)
    sessions: Dict[str, List[dict]] = defaultdict(list)
    for _, r in interactions.iterrows():
        session = r.get("sessionId") or r.get("personId") or "no_session"
        sessions[session].append(r.to_dict())

    # popularity for negatives
    pop = Counter(interactions["contentId"].dropna().astype(str).tolist())
    popular = [c for c, _ in pop.most_common(sample_negative_pool) if c in articles_meta]

    behaviors: List[Tuple[str, str, str, List[str]]] = []
    imp_id = 0
    for session_id, rows in sessions.items():
        # sort by time
        rows = sorted(rows, key=lambda x: int(x.get("timestamp") or 0))
        history: List[str] = []
        candidates: List[str] = []
        labels: List[int] = []

        # collect candidate items and label them
        for row in rows:
            cid = str(row.get("contentId", "")).strip()
            if not cid or cid not in articles_meta:
                continue
            etype = str(row.get("eventType", "VIEW")).strip().upper()
            # treat VIEW as candidate; positive if in ACTION_POSITIVE
            if etype in ("VIEW", "CLICK", "IMPRESSION"):
                candidates.append(cid)
                labels.append(0)
            elif etype in ACTION_POSITIVE:
                # positive action: add as positive candidate
                candidates.append(cid)
                labels.append(1)
            # update history with clicked/liked items
            if etype in ("VIEW", "CLICK") or etype in ACTION_POSITIVE:
                history.append(cid)

        if not candidates:
            continue

        # Ensure we have some negatives; if not, sample from popular pool
        if sum(labels) == len(labels):
            # all positives - need negatives
            sampled = [c for c in popular if c not in set(candidates)]
            while len(sampled) and len(candidates) < min_candidates:
                candidates.append(sampled.pop(0))
                labels.append(0)

        # If still too few candidates, skip
        if len(candidates) < 2:
            continue

        # Build impressions string format: nid-label pairs
        impressions = []
        for c, l in zip(candidates, labels):
            impressions.append(f"{c}-{int(bool(l))}")

        imp_id += 1
        impid = f"imp{imp_id}"
        uid = rows[0].get("personId") or session_id
        time = str(rows[-1].get("timestamp") or 0)
        behaviors.append((impid, str(uid), time, history, impressions))

    return behaviors


def write_news(news_path: Path, articles_meta: Dict[str, Dict[str, str]]) -> None:
    cols = ["nid", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    with news_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for nid, meta in articles_meta.items():
            row = [nid, "", "", meta.get("title", ""), meta.get("abstract", ""), meta.get("url", ""), "", ""]
            writer.writerow(row)


def write_behaviors(beh_path: Path, behaviors: List[Tuple[str, str, str, List[str]]]) -> None:
    with beh_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for impid, uid, time, history, impressions in behaviors:
            history_field = " ".join(history)
            impressions_field = " ".join(impressions)
            writer.writerow([impid, uid, time, history_field, impressions_field])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/custom_local_news")
    parser.add_argument("--output_dir", type=str, default="data/custom_local_news_mind")
    parser.add_argument("--min_candidates", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    articles_path = input_dir / "shared_articles.csv" / "shared_articles.csv"
    interactions_path = input_dir / "users_interactions.csv" / "users_interactions.csv"

    if not articles_path.exists() or not interactions_path.exists():
        print("Expected files not found under", input_dir)
        print("Make sure you placed: shared_articles.csv/shared_articles.csv and users_interactions.csv/users_interactions.csv")
        return

    print("Loading articles...")
    articles = load_articles(articles_path)
    print(f"Found {len(articles)} articles")

    print("Loading interactions...")
    interactions = load_interactions(interactions_path)
    print(f"Found {len(interactions)} interaction rows")

    print("Building behaviors ...")
    behaviors = build_behaviors(interactions, articles, min_candidates=args.min_candidates)
    print(f"Built {len(behaviors)} behavior rows")

    news_out = out_dir / "news.tsv"
    beh_out = out_dir / "behaviors.tsv"

    print("Writing news.tsv ...")
    write_news(news_out, articles)
    print("Writing behaviors.tsv ...")
    write_behaviors(beh_out, behaviors)

    print("Done. Output:")
    print(" ", news_out)
    print(" ", beh_out)


if __name__ == "__main__":
    main()
