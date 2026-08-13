from datasets import load_from_disk
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
import os, json, re

BASE_SAVE = "../saved_models"

def _sanitize(s: str) -> str:
    # keep it filesystem-safe; also replace '.' with 'p' for floats
    return re.sub(r'[^A-Za-z0-9_=.-]+', '_', s).replace('.', 'p')

def _args_tag(a) -> str:
    parts = [
        f"hd_{a.hidden_dim}",
        f"od_{a.output_dim}",
        f"nl_{a.num_layers}",
        f"bs_{a.batch_size}",
        f"lr_{a.lr}",
        f"wd_{a.wd}",
        f"ne_{a.num_epochs}",
        f"nr_{a.num_runs}",
    ]
    # include scheduler knobs if present
    if hasattr(a, "scheduler"): parts.append(f"sch_{a.scheduler}")
    if hasattr(a, "warmup_ratio"): parts.append(f"wr_{a.warmup_ratio}")
    if hasattr(a, "min_lr"): parts.append(f"minlr_{a.min_lr}")
    return _sanitize("_".join(parts))


import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from utils import EmbeddingDataset, cka, cca
from model import EmbeddingAligner, info_nce_loss

parser = ArgumentParser(description="LLM-Assisted-Live-Coding")
parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default=128, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default=5, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine','none'], help="LR schedule type")
parser.add_argument('--warmup_ratio', type=float, default=0.05, help="Fraction of total steps used for linear warmup")
parser.add_argument('--min_lr', type=float, default=1e-5, help="Floor LR at the end of decay")

parser.add_argument('--lr', type=float, default=3e-3, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--num_runs', type=int, default=5, help="Number of epochs")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

def train(args, train_loader, val_loader, input_dim_code, input_dim_wav):
    code_model = EmbeddingAligner(input_dim_code, args.hidden_dim, args.output_dim, args.num_layers).to(args.device)
    wav_model  = EmbeddingAligner(input_dim_wav,  args.hidden_dim, args.output_dim, args.num_layers).to(args.device)

    optimizer = torch.optim.AdamW(
        list(code_model.parameters()) + list(wav_model.parameters()),
        lr=args.lr, weight_decay=args.wd
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    min_lr_scale = args.min_lr / max(args.lr, 1e-12)  # avoid div-by-zero

    if args.scheduler == 'cosine':
        import math
        def lr_lambda(current_step: int):
            if current_step < warmup_steps and warmup_steps > 0:
                # linear warmup from 0 -> 1
                return float(current_step) / float(max(1, warmup_steps))
            # cosine decay from 1 -> min_lr_scale
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    best_val_loss = float('inf')
    best_models = None
    global_step = 0

    with tqdm(range(args.num_epochs)) as tq:
        for e in tq:
            code_model.train(); wav_model.train()
            total_loss = 0.0

            for code_emb, wav_emb in train_loader:
                code_emb = code_emb.to(args.device); wav_emb = wav_emb.to(args.device)

                z1 = code_model(code_emb)
                z2 = wav_model(wav_emb)

                loss = info_nce_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                global_step += 1

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_code_all, val_wav_all = test(args, val_loader, code_model, wav_model)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_models = (deepcopy(code_model), deepcopy(wav_model))
                with torch.no_grad():
                    cca_val = cca(val_code_all, val_wav_all)
                    cka_val = cka(val_code_all, val_wav_all)

            current_lr = optimizer.param_groups[0]['lr']
            tq.set_description(
                "LR = %.2e | Train = %.4f | Val = %.4f (best %.4f) | CCA = %.4f | CKA = %.4f"
                % (current_lr, avg_train_loss, val_loss, best_val_loss, cca_val, cka_val)
            )

    return best_models, cca_val.item(), cka_val.item()

def test(args, dataloader, code_model, wav_model):
    code_model.eval()
    wav_model.eval()
    total_loss = 0
    code_all = []
    wav_all = []

    with torch.no_grad():
        for code_emb, wav_emb in dataloader:
            code_emb = code_emb.to(args.device)
            wav_emb = wav_emb.to(args.device)

            z1 = code_model(code_emb)
            z2 = wav_model(wav_emb)

            loss = info_nce_loss(z1, z2)
            total_loss += loss.item()

            code_all.append(z1)
            wav_all.append(z2)

    code_all = torch.cat(code_all, dim=0)
    wav_all = torch.cat(wav_all, dim=0)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, code_all, wav_all


if __name__ == "__main__":
    print(args)
    dataset = load_from_disk("../sonicpi_embeddings_full.arrow")

    code_embeddings = []
    wav_embeddings = []

    for sample in dataset:
        code_embeddings.append(sample['code_embedding'])
        wav_embeddings.append(sample['wav_embedding'])
        
    code_embeddings = torch.tensor(code_embeddings)
    wav_embeddings = torch.tensor(wav_embeddings)

    cca_pre = []
    cca_post = []
    cka_pre = []
    cka_post = []
    best_models = []
    for i in range(args.num_runs):
        train_idx, test_idx = train_test_split(np.arange(len(code_embeddings)), test_size = 0.2)
        val_idx, test_idx = train_test_split(test_idx, test_size = 0.5)
        train_idx = torch.LongTensor(train_idx)
        val_idx = torch.LongTensor(val_idx)
        test_idx = torch.LongTensor(test_idx)
        cka_pre.append(cka(code_embeddings[val_idx], wav_embeddings[val_idx]))
        cca_pre.append(cca(code_embeddings[val_idx], wav_embeddings[val_idx]))

        train_loader = DataLoader(EmbeddingDataset(code_embeddings[train_idx], wav_embeddings[train_idx]), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(EmbeddingDataset(code_embeddings[val_idx], wav_embeddings[val_idx]), batch_size=args.batch_size, shuffle=False)

        model, best_cca, best_cka = train(args, train_loader, val_loader, code_embeddings.shape[1], wav_embeddings.shape[1])
        cka_post.append(best_cka)
        cca_post.append(best_cca)
        best_models.append(model)
        
    cca_pre = np.array(cca_pre)
    cca_post = np.array(cca_post)
    cka_pre = np.array(cka_pre)
    cka_post = np.array(cka_post)

    avg_cca = float(cca_post.mean())
    avg_cka = float(cka_post.mean())

    tag = _args_tag(args)
    save_dir = os.path.join(BASE_SAVE, tag)
    os.makedirs(save_dir, exist_ok=True)

    # save each run's best models
    for i, (code_model, wav_model) in enumerate(best_models):
        fn = f"run{i+1}_bestCCA_{cca_post[i]:.4f}_bestCKA_{cka_post[i]:.4f}.pt"
        path = os.path.join(save_dir, _sanitize(fn))
        torch.save(
            {
                "code_model_state_dict": code_model.state_dict(),
                "wav_model_state_dict": wav_model.state_dict(),
                "args": vars(args),
                "metrics": {
                    "best_cca": float(cca_post[i]),
                    "best_cka": float(cka_post[i]),
                    "avg_cca_over_runs": avg_cca,
                    "avg_cka_over_runs": avg_cka,
                    "run_index": i + 1,
                },
            },
            path,
        )

    # write summary.json
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "avg_cca_over_runs": avg_cca,
                "cca_std": cca_post.std(),
                "avg_cka_over_runs": avg_cka,
                "cka_std": cka_post.std(),
                "per_run": [
                    {"run": i+1, "best_cca": float(cca_post[i]), "best_cka": float(cka_post[i])}
                    for i in range(len(best_models))
                ],
            },
            f,
            indent=2,
        )


    print("====================")
    print("Before aligning!!")
    print("====================")
    print(f"Average CKA: {cka_pre.mean():.4f} $\\pm$ {cka_pre.std():.4f}")
    print(f"Average CCA: {cca_pre.mean():.4f} $\\pm$ {cca_pre.std():.4f}")
    print("**********************")
    print("After aligning!!")
    print("====================")
    print(f"Average CKA: {cka_post.mean():.4f} $\\pm$ {cka_post.std():.4f}")
    print(f"Average CCA: {cca_post.mean():.4f} $\\pm$ {cca_post.std():.4f}")
    print("====================")
    print(f"[Models saved in] {path}")
