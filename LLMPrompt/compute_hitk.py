import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedupe_by_candidate_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        idx = to_int(row.get("candidate_index"))
        if idx is None:
            continue
        latest[idx] = row
    return list(latest.values())


def group_by_query(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue
        grouped.setdefault(query_id, []).append(row)
    return grouped


def rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -(to_int(r.get("score")) or 0),
            to_int(r.get("candidate_index")) if to_int(r.get("candidate_index")) is not None else 10**9,
        ),
    )


def collect_positive_indices(rows: List[Dict[str, Any]]) -> Set[int]:
    positives: Set[int] = set()
    for row in rows:
        label = to_int(row.get("ground_truth_label"))
        idx = to_int(row.get("candidate_index"))
        if label == 1 and idx is not None:
            positives.add(idx)
    return positives


def compute_hitk(grouped: Dict[str, List[Dict[str, Any]]], ks: List[int]) -> Dict[str, Any]:
    ks = sorted(set(k for k in ks if k > 0))
    hit_counts = {k: 0 for k in ks}

    query_count = len(grouped)
    evaluable_query_count = 0
    query_without_positive_count = 0

    per_query: List[Dict[str, Any]] = []

    for query_id, rows in grouped.items():
        deduped = dedupe_by_candidate_index(rows)
        ranked = rank_rows(deduped)
        positive_indices = collect_positive_indices(deduped)

        if not positive_indices:
            query_without_positive_count += 1
            continue

        evaluable_query_count += 1

        ranked_indices = [to_int(r.get("candidate_index")) for r in ranked]
        ranked_indices = [idx for idx in ranked_indices if idx is not None]

        query_result: Dict[str, Any] = {
            "query_id": query_id,
            "positive_candidate_indices": sorted(positive_indices),
            "ranked_candidate_indices_top10": ranked_indices[:10],
        }

        for k in ks:
            topk = set(ranked_indices[:k])
            hit = len(topk.intersection(positive_indices)) > 0
            if hit:
                hit_counts[k] += 1
            query_result[f"hit@{k}"] = bool(hit)

        per_query.append(query_result)

    metrics: Dict[str, Any] = {
        "query_count": query_count,
        "evaluable_query_count": evaluable_query_count,
        "query_without_positive_count": query_without_positive_count,
    }

    for k in ks:
        key = f"hit@{k}"
        metrics[key] = (hit_counts[k] / evaluable_query_count) if evaluable_query_count > 0 else 0.0
        metrics[f"hit@{k}_count"] = hit_counts[k]

    return {
        "metrics": metrics,
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute hit@k from valid_pair_scores.jsonl")
    parser.add_argument(
        "--pair-scores",
        type=Path,
        default="out/llama/valid_pair_scores.jsonl",
        # required=True,
        help="Path to valid_pair_scores.jsonl",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[3, 5],
        help="k values for hit@k, e.g. --ks 3 5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output json path (default: <pair_scores_dir>/hitk_metrics.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pair_scores_path = args.pair_scores.resolve()
    if not pair_scores_path.exists():
        raise FileNotFoundError(f"pair scores file not found: {pair_scores_path}")

    rows = load_jsonl(pair_scores_path)
    grouped = group_by_query(rows)
    result = compute_hitk(grouped=grouped, ks=args.ks)

    output_path = args.output
    if output_path is None:
        output_path = pair_scores_path.parent / "hitk_metrics.json"
    output_path = output_path.resolve()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    metrics = result["metrics"]
    print(f"pair_scores: {pair_scores_path}")
    print(f"queries_total: {metrics['query_count']}")
    print(f"queries_evaluable: {metrics['evaluable_query_count']}")
    print(f"queries_without_positive: {metrics['query_without_positive_count']}")

    for k in sorted(set(k for k in args.ks if k > 0)):
        print(f"hit@{k}: {metrics[f'hit@{k}']:.6f} ({metrics[f'hit@{k}_count']})")

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()