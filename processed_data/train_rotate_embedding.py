import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def find_existing_file(base_dir: Path, candidates: Sequence[str], required: bool = True) -> Optional[Path]:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(
            "None of these files exist under {}: {}".format(base_dir, ", ".join(candidates))
        )
    return None


def load_id_map(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = "\t".join(parts[:-1]).strip()
            value = int(parts[-1].strip())
            if key in mapping and mapping[key] != value:
                raise ValueError("Conflicting IDs for key '{}': {} vs {}".format(key, mapping[key], value))
            mapping[key] = value
    return mapping


def merge_entity_maps(primary: Dict[str, int], secondary: Dict[str, int]) -> Dict[str, int]:
    merged = dict(primary)
    for key, value in secondary.items():
        if key in merged and merged[key] != value:
            raise ValueError("Entity ID mismatch for '{}': {} vs {}".format(key, merged[key], value))
        merged[key] = value
    return merged


def load_title_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def parse_id_triples(path: Path) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            triples.append((h, r, t))
    return triples


def parse_text_triples(path: Path, entity2id: Dict[str, int], relation2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h not in entity2id or t not in entity2id or r not in relation2id:
                continue
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def parse_json_triples(path: Path, entity2id: Dict[str, int], relation2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return triples
    for item in data:
        if not isinstance(item, dict):
            continue
        h = item.get("head")
        r = item.get("relation")
        t = item.get("tail")
        if h not in entity2id or t not in entity2id or r not in relation2id:
            continue
        triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def load_triples(data_dir: Path, entity2id: Dict[str, int], relation2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    id_path = find_existing_file(
        data_dir,
        ["knowledge_graph_triples_ids.txt", "knowledge_graph_triples_id.txt"],
        required=False,
    )
    if id_path is not None:
        triples = parse_id_triples(id_path)
        if triples:
            print("Loaded triples from {}: {}".format(id_path, len(triples)))
            return triples

    txt_path = find_existing_file(data_dir, ["knowledge_graph_triples.txt"], required=False)
    if txt_path is not None:
        triples = parse_text_triples(txt_path, entity2id, relation2id)
        if triples:
            print("Loaded triples from {}: {}".format(txt_path, len(triples)))
            return triples

    json_path = find_existing_file(data_dir, ["knowledge_graph_triples.json"], required=False)
    if json_path is not None:
        triples = parse_json_triples(json_path, entity2id, relation2id)
        if triples:
            print("Loaded triples from {}: {}".format(json_path, len(triples)))
            return triples

    raise RuntimeError("No usable knowledge graph triples were found in {}".format(data_dir))


class RotatE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, complex_dim: int):
        super().__init__()
        self.entity_re = nn.Embedding(num_entities, complex_dim)
        self.entity_im = nn.Embedding(num_entities, complex_dim)
        self.relation_phase = nn.Embedding(num_relations, complex_dim)
        self._reset_parameters(complex_dim)

    def _reset_parameters(self, complex_dim: int) -> None:
        bound = 6.0 / math.sqrt(complex_dim)
        nn.init.uniform_(self.entity_re.weight.data, -bound, bound)
        nn.init.uniform_(self.entity_im.weight.data, -bound, bound)
        nn.init.uniform_(self.relation_phase.weight.data, -math.pi, math.pi)

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        head = triples[:, 0]
        rel = triples[:, 1]
        tail = triples[:, 2]

        h_re = self.entity_re(head)
        h_im = self.entity_im(head)
        t_re = self.entity_re(tail)
        t_im = self.entity_im(tail)

        phase = self.relation_phase(rel)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        distance = torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12).sum(dim=1)
        return -distance


def sample_negative_triples(batch: torch.Tensor, num_entities: int) -> torch.Tensor:
    negative = batch.clone()
    bsz = batch.size(0)
    random_entities = torch.randint(0, num_entities, (bsz,), device=batch.device)
    corrupt_head = torch.rand(bsz, device=batch.device) < 0.5
    negative[corrupt_head, 0] = random_entities[corrupt_head]
    negative[~corrupt_head, 2] = random_entities[~corrupt_head]
    return negative


def iter_minibatches(triples: torch.Tensor, batch_size: int, device: torch.device) -> Iterable[torch.Tensor]:
    index = torch.randperm(triples.size(0))
    for start in range(0, triples.size(0), batch_size):
        idx = index[start:start + batch_size]
        yield triples[idx].to(device)


def train_rotate(
    triples: List[Tuple[int, int, int]],
    num_entities: int,
    num_relations: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    log_interval: int,
    device: torch.device,
) -> RotatE:
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be even, because RotatE uses complex embeddings.")

    complex_dim = embedding_dim // 2
    model = RotatE(num_entities, num_relations, complex_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    triples_tensor = torch.tensor(triples, dtype=torch.long)

    best_state = None
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in iter_minibatches(triples_tensor, batch_size, device):
            negative = sample_negative_triples(batch, num_entities)
            pos_score = model.score(batch)
            neg_score = model.score(negative)

            loss = (-F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()) * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = batch.size(0)
            total_loss += loss.item() * count
            seen += count

        avg_loss = total_loss / max(seen, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            print("Epoch {:4d}/{:4d} | loss {:.6f}".format(epoch, epochs, avg_loss))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_entity_embedding_dict(model: RotatE, entity2id: Dict[str, int], output_dim: int) -> Dict[str, List[float]]:
    entity_re = model.entity_re.weight.detach().cpu().float()
    entity_im = model.entity_im.weight.detach().cpu().float()
    concat_entity = torch.cat([entity_re, entity_im], dim=1)

    if concat_entity.size(1) != output_dim:
        raise ValueError("Export dim mismatch: got {}, expected {}".format(concat_entity.size(1), output_dim))

    result: Dict[str, List[float]] = {}
    max_idx = concat_entity.size(0) - 1
    for entity, idx in entity2id.items():
        if 0 <= idx <= max_idx:
            result[entity] = concat_entity[idx].tolist()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RotatE and export PSTEnv-compatible entity embeddings.")
    parser.add_argument("--data-dir", default="data_processed", help="Graph data directory.")
    parser.add_argument(
        "--output",
        default="data_processed/RotatE_ett_emb.pth",
        help="Output .pth path in TransE_ett_emb-compatible dict format.",
    )
    parser.add_argument("--embedding-dim", type=int, default=200, help="Final exported embedding dimension.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-7, help="Weight decay.")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    project_root = Path(settings.PROJ_DIR)
    data_dir = resolve_path(project_root, args.data_dir)
    output_path = resolve_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ett_map_path = find_existing_file(data_dir, ["ett2ID.txt", "ett2intID.txt"])
    str_map_path = find_existing_file(data_dir, ["strID2ID.txt", "strID2intID.txt"], required=False)
    rel_map_path = find_existing_file(data_dir, ["relation2ID.txt", "relation2intID.txt", "relatrion2intID.txt"])
    title_map_path = find_existing_file(data_dir, ["strID_to_title.json"], required=False)

    ett2id = load_id_map(ett_map_path)
    str2id = load_id_map(str_map_path) if str_map_path is not None else {}
    relation2id = load_id_map(rel_map_path)
    title_map = load_title_map(title_map_path)

    entity2id = merge_entity_maps(str2id, ett2id)
    triples = load_triples(data_dir, entity2id, relation2id)

    num_entities = max(entity2id.values()) + 1
    num_relations = max(relation2id.values()) + 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Data dir: {}".format(data_dir))
    print("Entity map: {} entries".format(len(entity2id)))
    print("Relation map: {} entries".format(len(relation2id)))
    print("Triples: {}".format(len(triples)))
    print("Device: {}".format(device))

    if title_map:
        covered = sum(1 for sid in title_map if sid in entity2id)
        print("strID_to_title coverage in entity map: {}/{}".format(covered, len(title_map)))

    model = train_rotate(
        triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        device=device,
    )

    entity_emb = build_entity_embedding_dict(model, entity2id, args.embedding_dim)
    torch.save(entity_emb, output_path)

    print("Saved RotatE entity embedding to: {}".format(output_path))
    print("Format is compatible with pstenv.py TransE_ett_emb.pth loader.")


if __name__ == "__main__":
    main()
