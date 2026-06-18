import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def get_sentence_embeddings(model, tokenizer, texts, device, max_length=128):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        emb = output.pooler_output
    else:
        # Fallback for models without pooler_output: mean-pooling over tokens.
        last_hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        emb = summed / counts

    return F.normalize(emb, p=2, dim=1)


def main():
    parser = argparse.ArgumentParser(
        description="Demo: load multilingual model and show sentence similarity ranking."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-multilingual-cased",
        help="HF model id or local path (e.g. aiana94/NaSE, bert-base-multilingual-cased).",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--query", type=str, default="Suomen talous kasvoi odotettua nopeammin.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "Finland's economy grew faster than expected.",
            "The football team won the cup final yesterday.",
            "Global markets reacted to inflation data.",
        ],
        help="Candidate sentences separated by spaces. Use quotes for each sentence.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer/model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    texts = [args.query] + args.candidates
    embeddings = get_sentence_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=args.max_length,
    )

    query_emb = embeddings[0:1]
    cand_embs = embeddings[1:]
    sims = torch.mm(query_emb, cand_embs.T).squeeze(0).cpu()

    ranked = sorted(
        [(float(score), text) for score, text in zip(sims, args.candidates)],
        key=lambda x: x[0],
        reverse=True,
    )

    print("\nQuery:")
    print(f"  {args.query}")
    print("\nRanking candidates by cosine similarity:")
    for idx, (score, text) in enumerate(ranked, start=1):
        print(f"  {idx}. score={score:.4f} | {text}")


if __name__ == "__main__":
    main()