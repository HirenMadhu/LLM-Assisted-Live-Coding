#!/usr/bin/env python
import os
import json
import math
import argparse
from typing import Tuple, List, Dict

import numpy as np
import torch
from datasets import load_from_disk

# ----- You have this in your repo -----
from model import EmbeddingAligner   # must match your training definition


# ============ Dissimilarity & utils ============

def cosine_distance_matrix(X: torch.Tensor) -> torch.Tensor:
    X = torch.nn.functional.normalize(X, p=2, dim=1, eps=1e-12)  # add eps
    sim = X @ X.t()
    sim = sim.clamp(-1.0, 1.0)
    D = 1.0 - sim
    import pdb; pdb.set_trace()
    return D

def euclidean_distance_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    X: [N, D], returns pairwise Euclidean distance matrix [N, N]
    """
    # (x - y)^2 = x^2 + y^2 - 2xy
    sq = (X * X).sum(dim=1, keepdim=True)      # [N,1]
    d2 = sq + sq.t() - 2.0 * (X @ X.t())
    d2 = torch.clamp(d2, min=0.0)
    return torch.sqrt(d2 + 1e-12)

def pairwise_stats(D: torch.Tensor) -> Dict[str, float]:
    """
    For a distance matrix D, compute summary over off-diagonal entries.
    """
    N = D.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=D.device)
    vals = D[mask]
    return {
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "median": float(vals.median().item()),
        "std": float(vals.std(unbiased=False).item()),
    }

def farthest_point_subset(D: torch.Tensor, k: int) -> List[int]:
    N = D.shape[0]
    assert N >= 3, "Need at least 3 points"

    # find farthest *off-diagonal* pair
    iu = torch.triu_indices(N, N, offset=1, device=D.device)
    off_vals = D[iu[0], iu[1]]
    best = torch.argmax(off_vals)
    i0 = int(iu[0, best].item())
    j0 = int(iu[1, best].item())

    # average distance to the two anchors
    avg = 0.5 * (D[:, i0] + D[:, j0])
    avg[i0] = -1e9
    avg[j0] = -1e9
    k0 = int(torch.argmax(avg).item())

    return [i0, j0, k0]


def jaccard(a: List[int], b: List[int]) -> float:
    A, B = set(a), set(b)
    return len(A & B) / float(len(A | B)) if (A or B) else 1.0

def overlap_at_k(a: List[int], b: List[int], k: int) -> float:
    A, B = set(a[:k]), set(b[:k])
    return len(A & B) / float(k)

def rank_corr(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Spearman correlation between two vectors (1D).
    """
    from scipy.stats import spearmanr  # light dep; if unavailable, comment and use numpy rank
    corr, _ = spearmanr(vec_a, vec_b)
    return float(np.nan_to_num(corr))


# ============ Loading embeddings ============

def load_arrow_embeddings(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (code_embeddings [N,Dc], wav_embeddings [N,Dw]) as float32 torch tensors.
    """
    ds = load_from_disk(path)
    code, wav = [], []
    for ex in ds:
        code.append(ex["code_embedding"])
        wav.append(ex["wav_embedding"])
    code = torch.tensor(np.asarray(code), dtype=torch.float32)
    wav = torch.tensor(np.asarray(wav), dtype=torch.float32)
    return code, wav

def discover_arrow_files(root: str) -> Dict[str, List[str]]:
    """
    Expects structure like:
      root/
        3-cand/
          label_embs_*.arrow
        10-cand/
          label_embs_*.arrow

    Returns {"3-cand": [file1, ...], "10-cand": [file1, ...]} for existing subdirs.
    """
    out = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root '{root}' is not a directory")

    for sub in os.listdir(root):
        cand_dir = os.path.join(root, sub)
        if os.path.isdir(cand_dir):
            arrow_files = []
            for name in os.listdir(cand_dir):
                if name.endswith(".arrow"):
                    arrow_files.append(os.path.join(cand_dir, name))
            if arrow_files:
                out[sub] = sorted(arrow_files)
    return out


# ============ Evaluation core ============

def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device) if t.device != device else t

def build_code_model(checkpoint_path: str, input_dim: int, hidden: int, output: int, layers: int, device: torch.device) -> EmbeddingAligner:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = EmbeddingAligner(input_dim, hidden, output, layers)
    model.load_state_dict(ckpt["code_model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model

def flatten_upper_triangle(D: torch.Tensor) -> np.ndarray:
    """
    Return vectorized upper-triangular (i<j) distances as numpy array.
    """
    N = D.shape[0]
    iu = torch.triu_indices(N, N, offset=1, device=D.device)
    return D[iu[0], iu[1]].detach().cpu().numpy()

def evaluate_set_agreement(
    wav_D: torch.Tensor,
    code_D: torch.Tensor,
    raw_code_D: torch.Tensor,
    k: int = 3
) -> Dict[str, float]:
    # Gold from wav embeddings
    gold = farthest_point_subset(wav_D, k)

    # Predicted from aligned code model outputs
    pred_aligned = farthest_point_subset(code_D, k)

    # Predicted from raw code embeddings
    pred_raw = farthest_point_subset(raw_code_D, k)

    # Metrics vs gold
    metrics = {
        "gold_indices": gold,
        "pred_aligned_indices": pred_aligned,
        "pred_raw_indices": pred_raw,
        "exact_match_aligned": float(set(gold) == set(pred_aligned)),
        "exact_match_raw": float(set(gold) == set(pred_raw)),
        "overlap3_aligned": float(len(set(gold) & set(pred_aligned)) / 3.0),
        "overlap3_raw": float(len(set(gold) & set(pred_raw)) / 3.0),
        "jaccard_aligned": jaccard(gold, pred_aligned),
        "jaccard_raw": jaccard(gold, pred_raw),
    }
    return metrics


# ============ Main CLI ============

def main():
    parser = argparse.ArgumentParser(description="Evaluate dissimilarity & selection alignment")
    parser.add_argument("--scenario_root", type=str, required=True,
                        help="Path to scenario root (e.g., 's1-mel')")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a saved checkpoint .pt (from your training) containing code_model_state_dict")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--k", type=int, default=3, help="Subset size to select for diversity")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", type=str, default=None,
                        help="If set, write JSON summary to this path")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Find folders like "3-cand" and "10-cand"
    found = discover_arrow_files(args.scenario_root)
    if not found:
        raise RuntimeError(f"No .arrow files discovered under {args.scenario_root}")

    # Load all groups
    loaded = {}
    for group, files in found.items():
        code_all, wav_all = [], []
        for f in files:
            c, w = load_arrow_embeddings(f)
            code_all.append(c)
            wav_all.append(w)
        code = torch.cat(code_all, dim=0)
        wav = torch.cat(wav_all, dim=0)
        loaded[group] = (code, wav)

    # Build code model (input_dim inferred from code embeddings in the biggest pool we have)
    # Prefer the largest candidate pool (e.g., "10-cand") for robust input_dim
    key_for_dim = max(loaded.keys(), key=lambda k: loaded[k][0].shape[0])
    input_dim_code = loaded[key_for_dim][0].shape[1]
    code_model = build_code_model(
        checkpoint_path=args.checkpoint,
        input_dim=input_dim_code,
        hidden=args.hidden_dim,
        output=args.output_dim,
        layers=args.num_layers,
        device=device
    )

    # Choose distance function
    dist_fn = cosine_distance_matrix if args.metric == "cosine" else euclidean_distance_matrix

    results = {
        "scenario_root": args.scenario_root,
        "metric": args.metric,
        "k": args.k,
        "groups": {}
    }

    # --- 1) Show 3-cand wav are "not dissimilar enough" ---
    if "3-cand" in loaded:
        code3, wav3 = loaded["3-cand"]
        wav3 = to_device(wav3, device)
        D_wav3 = dist_fn(wav3)
        stats_3cand = pairwise_stats(D_wav3)
        results["groups"]["3-cand"] = {
            "N": int(wav3.shape[0]),
            "wav_pairwise_stats": stats_3cand
        }

    # --- 2) On 10-cand: select gold via wav; compare predictions from aligned code vs raw code ---
    if "10-cand" in loaded:
        code10, wav10 = loaded["10-cand"]
        N10 = int(code10.shape[0])

        # Raw code embeddings
        code10_raw = to_device(code10, device)
        D_code_raw = dist_fn(code10_raw)

        # Aligned code embeddings (through code model)
        with torch.no_grad():
            z_code10 = code_model(to_device(wav10, device))
        D_code_aligned = dist_fn(z_code10)

        # Wav embeddings (gold)
        wav10 = to_device(wav10, device)
        D_wav10 = dist_fn(wav10)

        # Pairwise summaries
        stats_10cand = {
            "wav_pairwise_stats": pairwise_stats(D_wav10),
            "raw_code_pairwise_stats": pairwise_stats(D_code_raw),
            "aligned_code_pairwise_stats": pairwise_stats(D_code_aligned),
        }

        # Agreement on diverse-3 selection
        agree = evaluate_set_agreement(D_wav10, D_code_aligned, D_code_raw, k=args.k)

        # Optional: correlation between pairwise distance profiles (upper triangles)
        wav_vec = flatten_upper_triangle(D_wav10)
        raw_vec = flatten_upper_triangle(D_code_raw)
        ali_vec = flatten_upper_triangle(D_code_aligned)

        # If scipy isn't available, you can swap rank_corr with np.corrcoef on ranks
        try:
            corr_aligned = rank_corr(wav_vec, ali_vec)
            corr_raw = rank_corr(wav_vec, raw_vec)
        except Exception:
            # fallback: Pearson on ranks
            wav_r = np.argsort(np.argsort(wav_vec))
            ali_r = np.argsort(np.argsort(ali_vec))
            raw_r = np.argsort(np.argsort(raw_vec))
            corr_aligned = float(np.corrcoef(wav_r, ali_r)[0,1])
            corr_raw = float(np.corrcoef(wav_r, raw_r)[0,1])

        results["groups"]["10-cand"] = {
            "N": N10,
            "pairwise_stats": stats_10cand,
            "diverse_subset_agreement": agree,
            "pairwise_distance_profile_spearman": {
                "aligned_vs_wav": corr_aligned,
                "raw_vs_wav": corr_raw,
            }
        }

        # Also record which 3 got picked in each case for transparency
        results["groups"]["10-cand"]["selected_indices_detail"] = {
            "gold_wav": agree["gold_indices"],
            "pred_aligned_code": agree["pred_aligned_indices"],
            "pred_raw_code": agree["pred_raw_indices"]
        }

    # --- Pretty print / save ---
    print(json.dumps(results, indent=2))
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
