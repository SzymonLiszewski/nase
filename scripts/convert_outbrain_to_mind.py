"""Convert Kaggle Outbrain click-prediction files into MIND-style news.tsv & behaviors.tsv.

Usage:
  1) Download Outbrain files into a folder, e.g. `data/outbrain` (see README below).
  2) Run:
       python scripts/convert_outbrain_to_mind.py --input_dir data/outbrain --output_dir data/outbrain_mind

Notes:
  - The script expects at least `clicks_train.csv` and `events.csv` and `promoted_content.csv`.
  - To enrich news titles, also provide `documents.csv` (optional).
  - For memory-heavy datasets, consider using `--sample_fraction` to create a small subset first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_csv_maybe_gz(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def build_mappings(input_dir: Path) -> tuple[Dict[int, int], Dict[int, Dict[str, str]]]:
    promoted_path = input_dir / "promoted_content.csv"
    doc_path = input_dir / "documents.csv"

    ad2doc: Dict[int, int] = {}
    doc_meta: Dict[int, Dict[str, str]] = {}

    if promoted_path.exists():
        promoted = read_csv_maybe_gz(promoted_path)
        if "ad_id" in promoted.columns and "document_id" in promoted.columns:
            ad2doc = dict(zip(promoted["ad_id"].astype(int), promoted["document_id"].astype(int)))

    if doc_path.exists():
        docs = read_csv_maybe_gz(doc_path, dtype={"document_id": object})
        for _, row in docs.iterrows():
            try:
                doc_id = int(row.get("document_id"))
            except Exception:
                continue
            title = str(row.get("doc_title", ""))
            if not title:
                title = str(row.get("title", ""))
            url = str(row.get("doc_url", ""))
            doc_meta[doc_id] = {"title": title or "", "url": url or ""}

    return ad2doc, doc_meta


def convert(
    input_dir: Path,
    output_dir: Path,
    sample_fraction: float = 1.0,
    min_history: int = 1,
    max_history_len: int = 50,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    clicks_path = input_dir / "clicks_train.csv"
    events_path = input_dir / "events.csv"

    if not clicks_path.exists() or not events_path.exists():
        raise FileNotFoundError("Require clicks_train.csv and events.csv in input_dir")

    print("Reading promoted_content/documents (optional)...")
    ad2doc, doc_meta = build_mappings(input_dir)

    print("Reading events...")
    events = read_csv_maybe_gz(events_path)
    # events.csv has columns: display_id, uuid, document_id, timestamp, platform, geo_location, ..
    if "display_id" not in events.columns or "uuid" not in events.columns:
        raise ValueError("events.csv missing required columns: display_id, uuid")
    events = events[["display_id", "uuid", "timestamp"]].drop_duplicates()
    events["display_id"] = events["display_id"].astype(int)
    # timestamps are in unix seconds in Outbrain events
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="s", errors="coerce")

    print("Reading clicks (this can be large)...")
    clicks = read_csv_maybe_gz(clicks_path)
    clicks = clicks[clicks["display_id"].notna() & clicks["ad_id"].notna()]
    clicks["display_id"] = clicks["display_id"].astype(int)
    clicks["ad_id"] = clicks["ad_id"].astype(int)
    clicks["clicked"] = clicks["clicked"].astype(int)

    if sample_fraction < 1.0:
        clicks = clicks.sample(frac=sample_fraction, random_state=42)

    # Build impressions per display
    print("Grouping impressions per display...")
    impressions = defaultdict(list)  # display_id -> list of (ad_id, clicked)
    for _, r in clicks.iterrows():
        impressions[int(r["display_id"])].append((int(r["ad_id"]), int(r["clicked"])))

    # Build clicked map per display for history building
    clicked_per_display = {did: [ad for ad, c in pairs if c == 1] for did, pairs in impressions.items()}

    # Merge events with impressions to associate displays with users
    print("Merging displays with impressions to build behaviors...")
    displays = pd.DataFrame({"display_id": list(impressions.keys())})
    displays = displays.merge(events, on="display_id", how="left")
    displays = displays.sort_values(["uuid", "timestamp"]).reset_index(drop=True)

    # For each user, iterate displays chronologically and accumulate history
    behaviors_rows: List[List[str]] = []
    all_ad_ids = set()

    grouped = displays.groupby("uuid")
    for uid, group in grouped:
        # sort
        group = group.sort_values("timestamp")
        history: List[int] = []
        for _, row in group.iterrows():
            did = int(row["display_id"])
            ts = row["timestamp"]
            if did not in impressions:
                continue
            # build impressions string
            imp_pairs = impressions[did]
            imp_strs = [f"{ad}-{clicked}" for ad, clicked in imp_pairs]
            candidate_ads = [ad for ad, _ in imp_pairs]
            all_ad_ids.update(candidate_ads)

            # history uses previously clicked ad_ids (fallback to previously displayed ads if no clicks)
            if len(history) < min_history:
                # try to use previous clicked ads
                pass

            if len(history) >= min_history:
                hist_trunc = history[-max_history_len:]
                histories = " ".join(str(x) for x in hist_trunc)
                impressions_field = " ".join(imp_strs)
                behaviors_rows.append([str(did), str(uid), str(ts), histories, impressions_field])

            # update history: append clicked ads from this display; if none, append displayed ads as weak signal
            clicked_ads = clicked_per_display.get(did, [])
            if clicked_ads:
                history.extend(clicked_ads)
            else:
                # include candidate ads as weak history signal to avoid empty histories later
                history.extend([ad for ad, _ in imp_pairs])

    if not behaviors_rows:
        raise ValueError("No behavior rows with non-empty history were produced. Try lowering --min_history or provide more data.")

    behaviors_df = pd.DataFrame(behaviors_rows, columns=["impid", "uid", "time", "history", "impressions"])
    # write behaviors.tsv
    out_beh = output_dir / "behaviors.tsv"
    print(f"Writing behaviors to {out_beh} ({len(behaviors_df)} rows)")
    behaviors_df.to_csv(out_beh, sep="\t", index=False, header=False)

    # Build news.tsv from all_ad_ids (use promoted_content/documents when available)
    print("Building news.tsv...")
    rows = []
    for ad in sorted(all_ad_ids):
        nid = str(ad)
        category = ""
        subcategory = ""
        title = ""
        abstract = ""
        url = ""
        if ad in ad2doc:
            docid = ad2doc[ad]
            meta = doc_meta.get(docid, {})
            title = meta.get("title", "")
            url = meta.get("url", "")
        rows.append([nid, category, subcategory, title, abstract, url, "", ""])

    news_df = pd.DataFrame(rows)
    out_news = output_dir / "news.tsv"
    print(f"Writing news to {out_news} ({len(news_df)} rows)")
    news_df.to_csv(out_news, sep="\t", index=False, header=False)

    print("Conversion completed.")


def main():
    parser = argparse.ArgumentParser(description="Convert Outbrain click files to MIND-style tsvs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Subsample fraction for clicks (0-1)")
    parser.add_argument("--min_history", type=int, default=1, help="Minimum history length required for a behavior row")
    parser.add_argument("--max_history_len", type=int, default=50)

    args = parser.parse_args()
    convert(Path(args.input_dir), Path(args.output_dir), sample_fraction=args.sample_fraction, min_history=args.min_history, max_history_len=args.max_history_len)


if __name__ == "__main__":
    main()
