import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from prompt_builder import (
    DEFAULT_PROMPT_CONFIG,
    PromptConfig,
    build_pair_payload,
    build_scoring_prompt,
    infer_labels_for_query,
)


SYSTEM_PROMPT = (
    "You are an expert citation-analysis assistant. "
    "Given one query paper and one candidate citation, "
    "estimate how likely the candidate is a core source citation. "
    "The output score is a probability estimate and is not limited to binary 0/100. "
    "Return exactly one line: Score: <0-100 integer>."
)

SCORE_LINE_RE = re.compile(r"^\s*(?:score|final\s*score|answer)\s*:\s*(100|[1-9]?\d)\s*$", re.IGNORECASE)
PURE_SCORE_RE = re.compile(r"^\s*(100|[1-9]?\d)\s*$")

MODEL_CANDIDATE_REL_PATHS = {
    "llama2-7b-chat": [
        Path("pretrain_models") / "llama-chat",
        Path("pretrain_models") / "llama-chat" / "llama2-7b-chat-hf",
        Path("pretrain_models") / "llama-chat" / "llama-2-7b-chat-hf",
    ],
    "gemma2-2b-it": [
        Path("pretrain_models") / "gemma-it",
        Path("pretrain_models") / "gemma-it" / "gemma-2-2b-it",
        Path("pretrain_models") / "gemma-it" / "gemma2-2b-it",
    ],
}

MODEL_NAME_ALIASES = {
    "llama2-7b-chat": "llama2-7b-chat",
    "llama2": "llama2-7b-chat",
    "gemma2-2b-it": "gemma2-2b-it",
    "gemma2": "gemma2-2b-it",
}

MODEL_PROMPT_FORMAT = {
    "llama2-7b-chat": "llama2_inst",
    "gemma2-2b-it": "gemma_turn_tokens",
}


def canonicalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in MODEL_NAME_ALIASES:
        supported = ", ".join(sorted(MODEL_NAME_ALIASES.keys()))
        raise ValueError(f"Unsupported model '{model_name}'. Supported values: {supported}")
    return MODEL_NAME_ALIASES[key]


def resolve_default_model_path(project_root: Path, model_name: str) -> Path:
    candidates = MODEL_CANDIDATE_REL_PATHS[model_name]
    for rel_path in candidates:
        candidate = (project_root / rel_path).resolve()
        if (candidate / "config.json").exists():
            return candidate
    return (project_root / candidates[0]).resolve()

def format_prompt_for_model(model_name: str, user_prompt: str) -> str:
    if model_name == "gemma2-2b-it":
        merged_user_content = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        return (
            "<bos><start_of_turn>user\n"
            f"{merged_user_content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    if model_name == "llama2-7b-chat":
        return (
            "<s>[INST] <<SYS>>\n"
            f"{SYSTEM_PROMPT}\n"
            "<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )

    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"User: {user_prompt}\n"
        "Assistant:"
    )

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_score(text: str) -> Optional[int]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None

    pure_match = PURE_SCORE_RE.match(cleaned)
    if pure_match is not None:
        return int(pure_match.group(1))

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    last_line_score = SCORE_LINE_RE.match(last_line)
    if last_line_score is not None:
        return int(last_line_score.group(1))

    # If model output is long/echoed, do not mine numbers from prompt text.
    if len(lines) > 3 or len(cleaned) > 120:
        return None

    for line in reversed(lines):
        score_match = SCORE_LINE_RE.match(line)
        if score_match is not None:
            return int(score_match.group(1))
        one_line_match = PURE_SCORE_RE.match(line)
        if one_line_match is not None:
            return int(one_line_match.group(1))

    return None

def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _bytes_to_gib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 3)


def reset_peak_cuda_memory_stats() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(device_index)
        except Exception:
            # Keep running even if one device does not expose stats correctly.
            pass


def collect_peak_cuda_memory_stats() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_count": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "peak_allocated_gib": 0.0,
            "peak_reserved_gib": 0.0,
            "per_device": [],
        }

    per_device: List[Dict[str, Any]] = []
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    device_count = torch.cuda.device_count()

    for device_index in range(device_count):
        try:
            allocated_bytes = int(torch.cuda.max_memory_allocated(device_index))
            reserved_bytes = int(torch.cuda.max_memory_reserved(device_index))
            device_name = torch.cuda.get_device_name(device_index)
        except Exception:
            allocated_bytes = 0
            reserved_bytes = 0
            device_name = f"cuda:{device_index}"

        peak_allocated_bytes = max(peak_allocated_bytes, allocated_bytes)
        peak_reserved_bytes = max(peak_reserved_bytes, reserved_bytes)

        per_device.append(
            {
                "device_index": device_index,
                "device_name": device_name,
                "peak_allocated_bytes": allocated_bytes,
                "peak_reserved_bytes": reserved_bytes,
                "peak_allocated_gib": _bytes_to_gib(allocated_bytes),
                "peak_reserved_gib": _bytes_to_gib(reserved_bytes),
            }
        )

    return {
        "cuda_available": True,
        "device_count": device_count,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_allocated_gib": _bytes_to_gib(peak_allocated_bytes),
        "peak_reserved_gib": _bytes_to_gib(peak_reserved_bytes),
        "per_device": per_device,
    }


class LocalModelScorer:
    def __init__(
        self,
        model_name: str,
        model_path: Path,
        max_input_tokens: int,
        max_new_tokens: int,
        device: str,
        dtype_name: str,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.torch_dtype = resolve_torch_dtype(dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.model, self.input_device = self._load_model()

    def _load_model(self) -> Tuple[AutoModelForCausalLM, torch.device]:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")

        prefer_cuda = (self.device in {"auto", "cuda"}) and torch.cuda.is_available()

        if prefer_cuda:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                )
                first_device = next(model.parameters()).device
                return model.eval(), first_device
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.torch_dtype,
                )
                model = model.to("cuda")
                return model.eval(), torch.device("cuda")

        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self.torch_dtype,
        )
        model = model.to("cpu")
        return model.eval(), torch.device("cpu")

    def _format_prompt(self, user_prompt: str) -> str:
        return format_prompt_for_model(
            model_name=self.model_name,
            user_prompt=user_prompt,
        )

    def generate_text(self, user_prompt: str) -> str:
        prompt_text = self._format_prompt(user_prompt)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text


def call_llm_once(scorer: LocalModelScorer, prompt: str) -> str:
    return scorer.generate_text(prompt)


def score_with_retries(
    scorer: LocalModelScorer,
    prompt: str,
    max_retries: int,
) -> Tuple[int, str, Optional[str]]:
    followup_prompt = prompt
    last_text = ""
    last_error: Optional[str] = None

    for _ in range(max_retries + 1):
        try:
            text = call_llm_once(scorer=scorer, prompt=followup_prompt)
            last_text = text.strip()
            score = extract_score(last_text)
            if score is not None:
                return score, last_text, None

            followup_prompt = (
                f"{prompt}\n\n"
                "Your previous output did not follow the required output format.\n"
                f"Previous output:\n{last_text}\n\n"
                "Now output exactly one line and nothing else:\n"
                "Score: <0-100 integer>\n"
                "Reminder: this is a probability estimate, so intermediate values are encouraged when uncertain.\n"
            )
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0)

    if last_error is not None:
        return 0, last_text, last_error
    return 0, last_text, "score_parse_failed"

def find_few_shot_examples(
    train_pairs: Dict[str, List[List[Any]]],
    truth_label: Dict[str, Any],
    paper_info: Dict[str, Dict[str, Any]],
    bib_to_contexts: Dict[str, Any],
    cfg: PromptConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    positive_example: Optional[Dict[str, Any]] = None
    negative_example: Optional[Dict[str, Any]] = None

    for query_id, refs in train_pairs.items():
        labels = infer_labels_for_query(
            query_id=query_id,
            refs=refs,
            truth_label=truth_label,
        )

        for idx, ref in enumerate(refs):
            label = labels[idx] if idx < len(labels) else None
            if label not in (0, 1):
                continue

            pair_payload = build_pair_payload(
                query_id=query_id,
                ref=ref,
                label=label,
                paper_info=paper_info,
                bib_to_contexts=bib_to_contexts,
                cfg=cfg,
            )

            if label == 1 and positive_example is None:
                positive_example = pair_payload
            if label == 0 and negative_example is None:
                negative_example = pair_payload

            if positive_example is not None and negative_example is not None:
                return positive_example, negative_example

    raise RuntimeError(
        "Could not find both a positive and a negative few-shot example from train data."
    )


def load_existing_records(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    records: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (str(rec["query_id"]), int(rec["candidate_index"]))
            records[key] = rec
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pseudo_score(label: Optional[int], bid: str) -> int:
    base = 85 if label == 1 else 15
    noise = (sum(ord(ch) for ch in bid) % 11) - 5
    return max(0, min(100, base + noise))


def build_prompt_template_text(
    positive_example: Dict[str, Any],
    negative_example: Dict[str, Any],
    model_name: str,
) -> str:
    placeholder_target = {
        "query_title": "<QUERY_TITLE>",
        "query_abstract": "<QUERY_ABSTRACT>",
        "query_keywords": "<QUERY_KEYWORDS>",
        "candidate_title": "<CANDIDATE_TITLE>",
        "candidate_abstract": "<CANDIDATE_ABSTRACT>",
        "citation_context": "<CITATION_CONTEXT>",
    }
    user_prompt = build_scoring_prompt(
        positive_example=positive_example,
        negative_example=negative_example,
        target_pair=placeholder_target,
    )
    return format_prompt_for_model(model_name=model_name, user_prompt=user_prompt)

def emit_prompt_template(run_dir: Path, template_text: str) -> Path:
    template_path = run_dir / "prompt_template.txt"
    with template_path.open("w", encoding="utf-8") as f:
        f.write(template_text)

    print("\n===== FEW-SHOT PROMPT TEMPLATE (for paper) =====")
    print(template_text)
    print("===== END TEMPLATE =====")
    print(f"Prompt template saved to: {template_path}\n")
    return template_path


def compute_metrics(
    selected_query_ids: List[str],
    valid_pairs: Dict[str, List[List[Any]]],
    record_map: Dict[Tuple[str, int], Dict[str, Any]],
    truth_label: Dict[str, Any],
    truth_index: Dict[str, List[str]],
    truth_bid: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    query_summaries: List[Dict[str, Any]] = []

    success_count = 0
    complete_query_count = 0
    incomplete_query_count = 0

    for query_id in selected_query_ids:
        refs = valid_pairs[query_id]
        labels = infer_labels_for_query(query_id, refs, truth_label)

        records_for_query: List[Dict[str, Any]] = []
        missing = False
        for idx in range(len(refs)):
            rec = record_map.get((query_id, idx))
            if rec is None:
                missing = True
                break
            records_for_query.append(rec)

        if missing:
            incomplete_query_count += 1
            continue

        complete_query_count += 1
        top_rec = max(records_for_query, key=lambda r: (int(r["score"]), -int(r["candidate_index"])))
        top_idx = int(top_rec["candidate_index"])

        predicted_bid = str(top_rec.get("candidate_bid", ""))
        predicted_resolved_id = top_rec.get("candidate_resolved_id")
        predicted_resolved_id_str = (
            str(predicted_resolved_id) if predicted_resolved_id is not None else None
        )
        predicted_input_id = top_rec.get("candidate_input_id")
        predicted_input_id_str = (
            str(predicted_input_id) if predicted_input_id is not None else None
        )

        gt_bids = {
            str(v).strip()
            for v in (truth_bid.get(query_id, []) or [])
            if v is not None and str(v).strip()
        }
        gt_indices = {
            str(v).strip()
            for v in (truth_index.get(query_id, []) or [])
            if v is not None and str(v).strip()
        }

        if gt_bids:
            success = predicted_bid in gt_bids
        else:
            fallback_id = f"{query_id}{predicted_bid}"
            success = (
                (predicted_resolved_id_str in gt_indices if predicted_resolved_id_str else False)
                or (predicted_input_id_str in gt_indices if predicted_input_id_str else False)
                or (fallback_id in gt_indices)
            )

        if success:
            success_count += 1

        predicted_label = labels[top_idx] if top_idx < len(labels) else None
        query_summaries.append(
            {
                "query_id": query_id,
                "predicted_candidate_index": top_idx,
                "predicted_candidate_bid": predicted_bid,
                "predicted_candidate_input_id": top_rec.get("candidate_input_id"),
                "predicted_candidate_resolved_id": predicted_resolved_id,
                "predicted_score": int(top_rec["score"]),
                "predicted_label": predicted_label,
                "ground_truth_bids": sorted(gt_bids),
                "ground_truth_indices": sorted(gt_indices),
                "success": bool(success),
            }
        )

    success_rate = (success_count / complete_query_count) if complete_query_count else 0.0
    metrics = {
        "complete_query_count": complete_query_count,
        "incomplete_query_count": incomplete_query_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "succress_rate": success_rate,
    }
    return metrics, query_summaries


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = project_root / "data_processed"

    parser = argparse.ArgumentParser(
        description="Few-shot local LLM scoring for query-citation pairs on validation set."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=project_root / "LLMPrompt" / "output")

    parser.add_argument("--train-file", type=str, default="Train_papers_refs_pair.json")
    parser.add_argument("--valid-file", type=str, default="Valid_papers_refs_pair.json")
    parser.add_argument("--paper-info-file", type=str, default="paper_info.json")
    parser.add_argument("--bib-context-file", type=str, default="bib_to_contexts.json")
    parser.add_argument("--truth-label-file", type=str, default="ground_truth_label.json")
    parser.add_argument("--truth-index-file", type=str, default="ground_truth_index.json")
    parser.add_argument("--truth-bid-file", type=str, default="ground_truth_bid.json")

    parser.add_argument("--model", type=str, choices=["llama2-7b-chat", "gemma2-2b-it", "llama2", "gemma2"], default="gemma2-2b-it")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)

    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--max-queries", type=int, default=10)       # 0
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = canonicalize_model_name(args.model)

    data_dir = args.data_dir
    train_pairs = load_json(data_dir / args.train_file)
    valid_pairs = load_json(data_dir / args.valid_file)
    paper_info = load_json(data_dir / args.paper_info_file)
    bib_to_contexts = load_json(data_dir / args.bib_context_file)
    truth_label = load_json(data_dir / args.truth_label_file)
    truth_index = load_json(data_dir / args.truth_index_file)
    truth_bid = load_json(data_dir / args.truth_bid_file)

    valid_query_ids = list(valid_pairs.keys())
    if args.max_queries > 0:
        valid_query_ids = valid_query_ids[: args.max_queries]

    cfg = DEFAULT_PROMPT_CONFIG
    positive_example, negative_example = find_few_shot_examples(
        train_pairs=train_pairs,
        truth_label=truth_label,
        paper_info=paper_info,
        bib_to_contexts=bib_to_contexts,
        cfg=cfg,
    )

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pair_scores_path = run_dir / "valid_pair_scores.jsonl"
    query_summary_path = run_dir / "query_top_predictions.json"
    query_latency_path = run_dir / "query_inference_latency.json"
    metrics_path = run_dir / "metrics.json"
    few_shot_path = run_dir / "few_shot_examples.json"
    config_path = run_dir / "run_config.json"

    resolved_model_path = args.model_path
    if resolved_model_path is None:
        resolved_model_path = resolve_default_model_path(
            project_root=args.project_root,
            model_name=args.model,
        )
    else:
        resolved_model_path = resolved_model_path.resolve()


    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "prompt_format": MODEL_PROMPT_FORMAT[args.model],
                "model_path": str(resolved_model_path),
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "dry_run": args.dry_run,
                "max_queries": args.max_queries,
                "max_retries": args.max_retries,
                "sleep_seconds": args.sleep_seconds,
                "resume": args.resume,
                "data_dir": str(data_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with few_shot_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "positive_example": positive_example,
                "negative_example": negative_example,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    prompt_template_text = build_prompt_template_text(
        positive_example=positive_example,
        negative_example=negative_example,
        model_name=args.model,
    )
    prompt_template_path = emit_prompt_template(
        run_dir=run_dir,
        template_text=prompt_template_text,
    )

    if args.resume:
        record_map = load_existing_records(pair_scores_path)
    else:
        if pair_scores_path.exists():
            pair_scores_path.unlink()
        record_map = {}

    scorer: Optional[LocalModelScorer] = None
    if not args.dry_run:
        if not resolved_model_path.exists():
            raise RuntimeError(f"Local model path does not exist: {resolved_model_path}")

        scorer = LocalModelScorer(
            model_name=args.model,
            model_path=resolved_model_path,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype_name=args.torch_dtype,
        )

    total_pairs = sum(len(valid_pairs[qid]) for qid in valid_query_ids)
    existing_pair_count = sum(
        1
        for qid in valid_query_ids
        for cidx in range(len(valid_pairs[qid]))
        if (qid, cidx) in record_map
    )
    done_pairs = existing_pair_count
    new_pairs = 0

    print(f"Selected valid queries: {len(valid_query_ids)}")
    print(f"Total candidate pairs to score: {total_pairs}")
    print(f"Model: {args.model} ({resolved_model_path})")
    if existing_pair_count > 0:
        print(f"Already scored pairs (resume): {existing_pair_count}")

    reset_peak_cuda_memory_stats()
    inference_start_time = time.perf_counter()
    model_inference_time_sum_sec = 0.0
    query_latency_rows: List[Dict[str, Any]] = []

    pbar = tqdm(
        total=total_pairs,
        initial=existing_pair_count,
        desc="Scoring pairs",
        unit="pair",
    )

    try:
        for q_idx, query_id in enumerate(valid_query_ids, start=1):
            query_start_time = time.perf_counter()
            refs = valid_pairs[query_id]
            labels = infer_labels_for_query(query_id, refs, truth_label)
            query_new_pairs = 0

            for c_idx, ref in enumerate(refs):
                key = (query_id, c_idx)
                if key in record_map:
                    continue

                label = labels[c_idx] if c_idx < len(labels) else None
                pair_payload = build_pair_payload(
                    query_id=query_id,
                    ref=ref,
                    label=label,
                    paper_info=paper_info,
                    bib_to_contexts=bib_to_contexts,
                    cfg=cfg,
                )

                prompt = build_scoring_prompt(
                    positive_example=positive_example,
                    negative_example=negative_example,
                    target_pair=pair_payload,
                )

                if args.dry_run:
                    score = pseudo_score(label=label, bid=pair_payload["candidate_bid"])
                    raw_response = str(score)
                    error = None
                else:
                    assert scorer is not None
                    t0 = time.perf_counter()
                    score, raw_response, error = score_with_retries(
                        scorer=scorer,
                        prompt=prompt,
                        max_retries=args.max_retries,
                    )
                    model_inference_time_sum_sec += time.perf_counter() - t0

                record = {
                    "query_id": query_id,
                    "candidate_index": c_idx,
                    "candidate_input_id": pair_payload["candidate_input_id"],
                    "candidate_resolved_id": pair_payload["candidate_resolved_id"],
                    "candidate_bid": pair_payload["candidate_bid"],
                    "ground_truth_label": label,
                    "score": int(score),
                    "model": args.model,
                    "raw_response": raw_response,
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                }
                append_jsonl(pair_scores_path, record)
                record_map[key] = record

                done_pairs += 1
                new_pairs += 1
                query_new_pairs += 1
                pbar.update(1)

                if done_pairs % max(1, args.log_every) == 0 or done_pairs == total_pairs:
                    print(
                        f"Progress: {done_pairs}/{total_pairs} pairs "
                        f"(query {q_idx}/{len(valid_query_ids)}), new={new_pairs}"
                    )

                if args.sleep_seconds > 0 and not args.dry_run:
                    time.sleep(args.sleep_seconds)

            query_elapsed_sec = time.perf_counter() - query_start_time
            query_latency_rows.append(
                {
                    "query_id": query_id,
                    "citation_count": len(refs),
                    "new_pairs_in_this_run": query_new_pairs,
                    "query_elapsed_sec": query_elapsed_sec,
                    "avg_sec_per_citation": (query_elapsed_sec / len(refs)) if refs else 0.0,
                }
            )
    finally:
        pbar.close()

    inference_elapsed_sec = time.perf_counter() - inference_start_time
    peak_gpu_memory_stats = collect_peak_cuda_memory_stats()

    with query_latency_path.open("w", encoding="utf-8") as f:
        json.dump(query_latency_rows, f, indent=2, ensure_ascii=False)

    avg_query_latency_sec = (
        inference_elapsed_sec / len(valid_query_ids)
        if len(valid_query_ids) > 0
        else 0.0
    )
    avg_pair_latency_sec = (inference_elapsed_sec / total_pairs) if total_pairs > 0 else 0.0
    avg_new_pair_latency_sec = (
        model_inference_time_sum_sec / new_pairs
        if new_pairs > 0
        else 0.0
    )

    metrics, query_summaries = compute_metrics(
        selected_query_ids=valid_query_ids,
        valid_pairs=valid_pairs,
        record_map=record_map,
        truth_label=truth_label,
        truth_index=truth_index,
        truth_bid=truth_bid,
    )

    metrics.update(
        {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "prompt_format": MODEL_PROMPT_FORMAT[args.model],
            "model_path": str(resolved_model_path),
            "run_name": args.run_name,
            "prompt_template_path": str(prompt_template_path),
            "query_latency_path": str(query_latency_path),
            "selected_query_count": len(valid_query_ids),
            "total_pair_count": total_pairs,
            "new_pair_count": new_pairs,
            "inference_total_elapsed_sec": inference_elapsed_sec,
            "inference_total_elapsed_min": (inference_elapsed_sec / 60.0),
            "inference_model_time_sum_sec": model_inference_time_sum_sec,
            "avg_latency_sec_per_query_all_citations": avg_query_latency_sec,
            "avg_latency_sec_per_query_citation_pair": avg_pair_latency_sec,
            "avg_model_latency_sec_per_new_pair": avg_new_pair_latency_sec,
            "peak_gpu_allocated_gib": peak_gpu_memory_stats["peak_allocated_gib"],
            "peak_gpu_reserved_gib": peak_gpu_memory_stats["peak_reserved_gib"],
            "peak_gpu_memory": peak_gpu_memory_stats,
            "output_dir": str(run_dir),
        }
    )

    with query_summary_path.open("w", encoding="utf-8") as f:
        json.dump(query_summaries, f, indent=2, ensure_ascii=False)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Pair scores: {pair_scores_path}")
    print(f"Top predictions: {query_summary_path}")
    print(f"Metrics: {metrics_path}")
    if peak_gpu_memory_stats["cuda_available"] and peak_gpu_memory_stats["device_count"] > 0:
        print(f"Peak GPU allocated: {peak_gpu_memory_stats['peak_allocated_gib']:.4f} GiB")
        print(f"Peak GPU reserved: {peak_gpu_memory_stats['peak_reserved_gib']:.4f} GiB")
    print(f"Success rate: {metrics['success_rate']:.6f}")


if __name__ == "__main__":
    main()

