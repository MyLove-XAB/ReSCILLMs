import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def build_edges(triples: List[Tuple[int, int, int]], undirected: bool = True) -> Tuple[List[int], List[int], List[int]]:
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_rel: List[int] = []
    for h, r, t in triples:
        edge_src.append(h)
        edge_dst.append(t)
        edge_rel.append(r)
        if undirected:
            edge_src.append(t)
            edge_dst.append(h)
            edge_rel.append(r)
    return edge_src, edge_dst, edge_rel


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim, bias=True)
        self.linear_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.zeros_(self.linear_self.bias)
        nn.init.xavier_uniform_(self.linear_neigh.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_message: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_feat = x[edge_src]
        if edge_message is not None:
            src_feat = src_feat + edge_message

        agg = torch.zeros((x.size(0), src_feat.size(1)), dtype=x.dtype, device=x.device)
        agg.index_add_(0, edge_dst, src_feat)

        degree = torch.bincount(edge_dst, minlength=x.size(0)).to(x.device)
        degree = degree.clamp(min=1).to(x.dtype).unsqueeze(1)
        neigh_mean = agg / degree
        return self.linear_self(x) + self.linear_neigh(neigh_mean)


class GraphSAGEEmbedding(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.node_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.layer1 = GraphSAGELayer(embedding_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, embedding_dim)
        self.relation_to_hidden = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.relation_to_hidden.weight)
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, edge_src: torch.Tensor, edge_dst: torch.Tensor, edge_rel: torch.Tensor) -> torch.Tensor:
        x = self.node_embedding.weight
        rel_msg = self.relation_embedding(edge_rel)

        x = self.layer1(x, edge_src, edge_dst, rel_msg)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        rel_msg_hidden = self.relation_to_hidden(rel_msg)
        x = self.layer2(x, edge_src, edge_dst, rel_msg_hidden)
        x = F.normalize(x, p=2, dim=1)
        return x


def train_graphsage(
    model: GraphSAGEEmbedding,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_rel: torch.Tensor,
    num_entities: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    log_interval: int,
) -> GraphSAGEEmbedding:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = None

    num_edges = edge_src.size(0)
    if num_edges == 0:
        raise RuntimeError("No edges available for GraphSAGE training.")

    device = edge_src.device

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            z = model(edge_src, edge_dst, edge_rel)

            pos_idx = torch.randint(0, num_edges, (batch_size,), device=device)
            pos_u = edge_src[pos_idx]
            pos_v = edge_dst[pos_idx]
            neg_v = torch.randint(0, num_entities, (batch_size,), device=device)

            pos_logits = (z[pos_u] * z[pos_v]).sum(dim=1)
            neg_logits = (z[pos_u] * z[neg_v]).sum(dim=1)

            loss = F.softplus(-pos_logits).mean() + F.softplus(neg_logits).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch == epochs or epoch % log_interval == 0:
            print("Epoch {:4d}/{:4d} | loss {:.6f}".format(epoch, epochs, avg_loss))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_entity_embedding_dict(
    model: GraphSAGEEmbedding,
    entity2id: Dict[str, int],
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_rel: torch.Tensor,
) -> Dict[str, List[float]]:
    model.eval()
    with torch.no_grad():
        embeddings = model(edge_src, edge_dst, edge_rel).detach().cpu().float()

    result: Dict[str, List[float]] = {}
    max_idx = embeddings.size(0) - 1
    for entity, idx in entity2id.items():
        if 0 <= idx <= max_idx:
            result[entity] = embeddings[idx].tolist()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE and export PSTEnv-compatible entity embeddings.")
    parser.add_argument("--data-dir", default="data_processed", help="Graph data directory.")
    parser.add_argument(
        "--output",
        default="data_processed/GraphSAGE_ett_emb.pth",
        help="Output .pth path in TransE_ett_emb-compatible dict format.",
    )
    parser.add_argument("--embedding-dim", type=int, default=200, help="Exported embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=200, help="Hidden dimension of GraphSAGE.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=5, help="Optimization steps per epoch.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Positive/negative edge sample size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-7, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used between GraphSAGE layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda.")
    parser.add_argument("--directed", action="store_true", help="Keep graph directed (default: add reverse edges).")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.embedding_dim != 200:
        raise ValueError("For compatibility with pstenv.py, --embedding-dim must be 200.")

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
    edge_src_list, edge_dst_list, edge_rel_list = build_edges(triples, undirected=not args.directed)

    num_entities = max(entity2id.values()) + 1
    num_relations = max(relation2id.values()) + 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    edge_src = torch.tensor(edge_src_list, dtype=torch.long, device=device)
    edge_dst = torch.tensor(edge_dst_list, dtype=torch.long, device=device)
    edge_rel = torch.tensor(edge_rel_list, dtype=torch.long, device=device)

    print("Data dir: {}".format(data_dir))
    print("Entity map: {} entries".format(len(entity2id)))
    print("Relation map: {} entries".format(len(relation2id)))
    print("Triples: {}".format(len(triples)))
    print("Training edges: {} (directed={})".format(edge_src.size(0), args.directed))
    print("Device: {}".format(device))

    if title_map:
        covered = sum(1 for sid in title_map if sid in entity2id)
        print("strID_to_title coverage in entity map: {}/{}".format(covered, len(title_map)))

    model = GraphSAGEEmbedding(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    model = train_graphsage(
        model=model,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_rel=edge_rel,
        num_entities=num_entities,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
    )

    entity_emb = build_entity_embedding_dict(
        model=model,
        entity2id=entity2id,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_rel=edge_rel,
    )
    torch.save(entity_emb, output_path)

    print("Saved GraphSAGE entity embedding to: {}".format(output_path))
    print("Format is compatible with pstenv.py TransE_ett_emb.pth loader.")


if __name__ == "__main__":
    main()

