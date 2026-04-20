import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PromptConfig:
    max_title_chars: int = 300
    max_query_abstract_chars: int = 1600
    max_candidate_abstract_chars: int = 1200
    max_context_chars: int = 1200
    max_keywords: int = 12


DEFAULT_PROMPT_CONFIG = PromptConfig()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def clean_text(value: Any, max_chars: int) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v is not None)
    text = str(value)
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return "N/A"
    return _truncate(text, max_chars=max_chars)


def normalize_keywords(value: Any, max_keywords: int) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (list, tuple)):
        keywords = [str(v).strip() for v in value if str(v).strip()]
        if not keywords:
            return "N/A"
        return ", ".join(keywords[:max_keywords])
    text = str(value).strip()
    if not text:
        return "N/A"
    if "[SEP]" in text:
        parts = [p.strip() for p in text.split("[SEP]") if p.strip()]
        if not parts:
            return "N/A"
        return ", ".join(parts[:max_keywords])
    return text


def resolve_candidate_info(
    query_id: str,
    candidate_id: Optional[str],
    candidate_bid: str,
    paper_info: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    if candidate_id and candidate_id in paper_info:
        return candidate_id, paper_info[candidate_id]
    query_bid_id = f"{query_id}{candidate_bid}"
    if query_bid_id in paper_info:
        return query_bid_id, paper_info[query_bid_id]
    if candidate_id:
        return candidate_id, {}
    return query_bid_id, {}


def get_citation_context(
    bib_to_contexts: Dict[str, Any],
    query_id: str,
    candidate_bid: str,
    max_chars: int,
) -> str:
    query_ctx = bib_to_contexts.get(query_id, {})
    context_value: Any = ""
    if isinstance(query_ctx, dict):
        context_value = query_ctx.get(candidate_bid, "")
    elif isinstance(query_ctx, (list, tuple)):
        if candidate_bid.startswith("b") and candidate_bid[1:].isdigit():
            idx = int(candidate_bid[1:])
            if 0 <= idx < len(query_ctx):
                context_value = query_ctx[idx]
    if isinstance(context_value, (list, tuple)):
        context_value = " ".join(str(v) for v in context_value if v is not None)
    return clean_text(context_value, max_chars=max_chars)


def infer_labels_for_query(
    query_id: str,
    refs: Sequence[Sequence[Any]],
    truth_label: Dict[str, Any],
) -> List[Optional[int]]:
    labels = truth_label.get(query_id)
    if isinstance(labels, list) and len(labels) == len(refs):
        output: List[Optional[int]] = []
        for v in labels:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                iv = None
            output.append(iv)
        return output

    output = []
    for ref in refs:
        label = None
        if isinstance(ref, (list, tuple)) and len(ref) >= 3:
            try:
                label = int(ref[2])
            except (TypeError, ValueError):
                label = None
        output.append(label)
    return output


def build_pair_payload(
    query_id: str,
    ref: Sequence[Any],
    label: Optional[int],
    paper_info: Dict[str, Dict[str, Any]],
    bib_to_contexts: Dict[str, Any],
    cfg: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> Dict[str, Any]:
    candidate_input_id = None
    candidate_bid = ""
    if len(ref) > 0:
        candidate_input_id = ref[0]
    if len(ref) > 1:
        candidate_bid = str(ref[1])

    query_info = paper_info.get(query_id, {})
    resolved_candidate_id, candidate_info = resolve_candidate_info(
        query_id=query_id,
        candidate_id=candidate_input_id,
        candidate_bid=candidate_bid,
        paper_info=paper_info,
    )

    payload = {
        "query_id": query_id,
        "candidate_input_id": candidate_input_id,
        "candidate_resolved_id": resolved_candidate_id,
        "candidate_bid": candidate_bid,
        "label": label,
        "query_title": clean_text(query_info.get("title", ""), cfg.max_title_chars),
        "query_abstract": clean_text(
            query_info.get("abstract", ""),
            cfg.max_query_abstract_chars,
        ),
        "query_keywords": normalize_keywords(
            query_info.get("keywords", []),
            cfg.max_keywords,
        ),
        "candidate_title": clean_text(
            candidate_info.get("title", ""),
            cfg.max_title_chars,
        ),
        "candidate_abstract": clean_text(
            candidate_info.get("abstract", ""),
            cfg.max_candidate_abstract_chars,
        ),
        "citation_context": get_citation_context(
            bib_to_contexts=bib_to_contexts,
            query_id=query_id,
            candidate_bid=candidate_bid,
            max_chars=cfg.max_context_chars,
        ),
    }
    return payload


def format_pair_block(pair_payload: Dict[str, Any]) -> str:
    return (
        "Query Paper:\n"
        f"Title: {pair_payload['query_title']}\n"
        f"Abstract: {pair_payload['query_abstract']}\n"
        f"Keywords: {pair_payload['query_keywords']}\n"
        "Candidate Citation:\n"
        f"Title: {pair_payload['candidate_title']}\n"
        f"Abstract: {pair_payload['candidate_abstract']}\n"
        f"Citation Context: {pair_payload['citation_context']}\n"
    )


def format_example_block(
    index: int,
    pair_payload: Dict[str, Any],
    example_score: int,
) -> str:
    return (
        f"Example {index}\n"
        f"{format_pair_block(pair_payload)}"
        f"Score: {int(example_score)}\n"
    )

def build_scoring_prompt(
    positive_example: Dict[str, Any],
    negative_example: Dict[str, Any],
    target_pair: Dict[str, Any],
) -> str:
    instructions = (
        "Task: estimate how likely the Candidate Citation is a core source citation "
        "for the Query Paper.\n"
        "Score range: 0 to 100.\n"
        "Higher score means higher probability that it is a source citation.\n"
        "Important: the score is a probability estimate, not a binary class label.\n"
        "Do NOT default to only 0 or 100. Use intermediate values when uncertainty exists.\n"
        "Scoring guidance:\n"
        "- 80-100: directly foundational or method-critical citation.\n"
        "- 50-79: clearly relevant but not obviously core.\n"
        "- 20-49: weak relation.\n"
        "- 0-19: mostly unrelated.\n"
        "Output format (strict):\n"
        "Score: <one integer between 0 and 100>\n"
        "Do not repeat or paraphrase the prompt.\n"
    )
    examples = (
        f"{format_example_block(1, positive_example, 88)}\n"
        f"{format_example_block(2, negative_example, 12)}\n"
    )
    target = (
        "Now evaluate the following target pair.\n"
        f"{format_pair_block(target_pair)}"
        "Return exactly one line in this format:\n"
        "Score: <0-100 integer>\n"
    )
    return f"{instructions}\n{examples}\n{target}"
