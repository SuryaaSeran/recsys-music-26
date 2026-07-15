"""
Neural listwise LTR — MLP with per-group softmax cross-entropy (ListNet top-1).

Input  : same NPZ as train_ltr_lightgbm.py
Output : models/ltr/<name>/{model.pt, scaler.json, meta.json}

Architecture (default):
    BN(n_feats) → Linear(n_feats, 256) → ReLU → Dropout(0.1)
              → Linear(256, 128) → ReLU → Dropout(0.1)
              → Linear(128, 64)  → ReLU
              → Linear(64, 1)    → scalar relevance score

Loss: ListNet top-1 = -log P(gold | scores) where P = softmax over group.
With soft labels (0/1/2), targets are proportional to label_gain[y] via softmax.

Evaluation: nDCG@20 computed exactly (same formula as LightGBM eval).

Usage:
    python scripts/train/train_ltr_neural.py \
        --features exp/analysis/ltr_phase_b_train_features.npz \
        --out models/ltr/neural_mlp_phase_b \
        --n_folds 5 --epochs 20 --lr 1e-3 --batch_turns 32

    # with soft labels (feature dump built with --soft_labels)
    python scripts/train/train_ltr_neural.py \
        --features exp/analysis/ltr_phase_b_soft_train_features.npz \
        --out models/ltr/neural_mlp_phase_b_soft \
        --soft_labels --n_folds 5 --epochs 20
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--features",       required=True)
parser.add_argument("--out",            required=True)
parser.add_argument("--n_folds",        type=int,   default=5)
parser.add_argument("--epochs",         type=int,   default=20)
parser.add_argument("--lr",             type=float, default=1e-3)
parser.add_argument("--weight_decay",   type=float, default=1e-4)
parser.add_argument("--batch_turns",    type=int,   default=32,
                    help="Number of groups (turns) per gradient update.")
parser.add_argument("--hidden",         type=str,   default="256,128,64",
                    help="Comma-separated hidden layer sizes.")
parser.add_argument("--dropout",        type=float, default=0.1)
parser.add_argument("--soft_labels",    action="store_true",
                    help="Use graded labels (0/1/2) with label_gain=[0,1,3].")
parser.add_argument("--poly_feats",     action="store_true",
                    help="Add pairwise interaction features at load time.")
parser.add_argument("--patience",       type=int,   default=5,
                    help="Early stopping patience in epochs.")
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--device",         default="cpu",
                    help="'cpu', 'cuda', or 'mps'.")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device)
hidden_sizes = [int(h) for h in args.hidden.split(",")]

# ── Label gain ────────────────────────────────────────────────────────────────
LABEL_GAIN = np.array([0.0, 1.0, 3.0] if args.soft_labels else [0.0, 1.0],
                      dtype=np.float32)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {args.features}...")
data         = np.load(args.features, allow_pickle=True)
X_all        = data["X"].astype(np.float32)
y_all        = data["y"].astype(np.int32)
group_all    = data["group"].astype(np.int32)
feature_cols = list(data["feature_cols"])
n_turns      = int(group_all.max()) + 1
print(f"  X: {X_all.shape}  turns: {n_turns}  features: {len(feature_cols)}")
print(f"  positives: {int(y_all.sum())}  pos_rate: {y_all.mean():.5f}  "
      f"mean_pool: {X_all.shape[0]/n_turns:.1f}")

sidecar = Path(args.features).with_suffix(".meta.json")
with open(sidecar) as f:
    meta = json.load(f)
turn_meta = meta["turn_meta"]

# ── Polynomial feature interactions ──────────────────────────────────────────
INTERACTION_PAIRS = [
    ("tt_cos",      "bm25_signal",     "tt_x_bm25"),
    ("tt_rank_sig", "bm25_origin",     "ttrank_x_bm25orig"),
    ("tt_cos",      "tt_rank_sig",     "tt_x_ttrank"),
    ("qm_cos",      "bm25_signal",     "qm_x_bm25"),
    ("artist_sig",  "artist_origin",   "artist_x_orig"),
    ("nn_sig",      "tt_cos",          "nn_x_tt"),
    ("collab_rank_sig", "collab_score","collab_rank_x_score"),
    ("popularity",  "tt_cos",          "pop_x_tt"),
    ("popularity",  "bm25_signal",     "pop_x_bm25"),
]
if args.poly_feats:
    col_idx = {name: i for i, name in enumerate(feature_cols)}
    added = []
    for fa, fb, name in INTERACTION_PAIRS:
        ia, ib = col_idx.get(fa), col_idx.get(fb)
        if ia is None or ib is None:
            continue
        added.append((name, X_all[:, ia] * X_all[:, ib]))
    if added:
        X_all = np.hstack([X_all] + [c[:, None] for _, c in added]).astype(np.float32)
        feature_cols = feature_cols + [n for n, _ in added]
        print(f"  poly_feats: +{len(added)} → {len(feature_cols)} total")

n_feats = X_all.shape[1]

# ── Pre-group data into list of (X_g, y_g) tensors ───────────────────────────
# Build per-turn slices. Groups are contiguous in the npz (turn order).
print("Pre-grouping turns...")
turn_sizes = np.bincount(group_all)   # (n_turns,)
boundaries = np.concatenate([[0], np.cumsum(turn_sizes)])

# Input normalisation: fit on all data
X_mean = X_all.mean(axis=0)
X_std  = X_all.std(axis=0) + 1e-8
X_norm = (X_all - X_mean) / X_std

# Keep group tensors on CPU; move each group to device on demand inside the
# loops. With 45.8M rows this avoids a ~5GB resident copy on the (unified) GPU.
groups: list[tuple[torch.Tensor, torch.Tensor]] = []
for t in range(n_turns):
    s, e = int(boundaries[t]), int(boundaries[t+1])
    Xg = torch.from_numpy(X_norm[s:e])                       # CPU (pool, n_feats)
    yg = torch.from_numpy(y_all[s:e].astype(np.int64))       # CPU (pool,)
    groups.append((Xg, yg))

# ── Session-stratified folds ──────────────────────────────────────────────────
session_to_turns: dict[str, list[int]] = {}
for t_idx, m in enumerate(turn_meta):
    session_to_turns.setdefault(m["session_id"], []).append(t_idx)
sessions = list(session_to_turns.keys())
rng = np.random.default_rng(args.seed)
rng.shuffle(sessions)
folds = [sessions[i::args.n_folds] for i in range(args.n_folds)]

# ── Model ─────────────────────────────────────────────────────────────────────

class ListwiseMLP(nn.Module):
    def __init__(self, n_feats: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_feats
        for out_dim in hidden:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (pool,)

# ── Loss: ListNet top-1 (= softmax cross-entropy over group) ─────────────────

def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Per-group ListNet top-1 loss.
    scores : (pool,) float
    labels : (pool,) int  — values in {0,1} or {0,1,2}
    Returns scalar loss, or None if no positives in group.
    """
    gains = torch.tensor(LABEL_GAIN, device=scores.device)[labels]  # (pool,)
    if gains.sum() == 0:
        return None
    # Target: soft probability proportional to gains (temperature=1)
    target_probs = torch.softmax(gains, dim=0)
    log_preds    = torch.log_softmax(scores, dim=0)
    return -(target_probs * log_preds).sum()

# ── nDCG@20 evaluation ───────────────────────────────────────────────────────

def ndcg_at_k(scores_np: np.ndarray, labels_np: np.ndarray, k: int = 20) -> float:
    order  = np.argsort(scores_np)[::-1][:k]
    gains  = LABEL_GAIN[np.clip(labels_np[order], 0, len(LABEL_GAIN)-1)]
    disc   = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg    = (gains * disc).sum()
    ideal  = np.sort(LABEL_GAIN[np.clip(labels_np, 0, len(LABEL_GAIN)-1)])[::-1][:k]
    idcg   = (ideal * disc[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate(model: ListwiseMLP, turn_idxs: list[int], batch_turns: int = 64) -> dict:
    model.eval()
    ndcg20s, ndcg10s, ndcg1s = [], [], []
    with torch.no_grad():
        for bs in range(0, len(turn_idxs), batch_turns):
            batch = turn_idxs[bs: bs + batch_turns]
            Xcat  = torch.cat([groups[t][0] for t in batch], dim=0).to(device)
            sizes = [groups[t][0].shape[0] for t in batch]
            scores_all = model(Xcat).cpu().numpy()
            off = 0
            for ti, t in enumerate(batch):
                sz = sizes[ti]
                s  = scores_all[off:off+sz]
                y  = groups[t][1].numpy()
                off += sz
                ndcg20s.append(ndcg_at_k(s, y, 20))
                ndcg10s.append(ndcg_at_k(s, y, 10))
                ndcg1s.append(ndcg_at_k(s, y, 1))
    return {
        "ndcg20": float(np.mean(ndcg20s)),
        "ndcg10": float(np.mean(ndcg10s)),
        "ndcg1":  float(np.mean(ndcg1s)),
    }

# ── Training loop ─────────────────────────────────────────────────────────────

def train_fold(train_turns: list[int], val_turns: list[int],
               fold_idx: int) -> tuple[ListwiseMLP, dict]:
    model = ListwiseMLP(n_feats, hidden_sizes, args.dropout).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_ndcg, best_state, patience_cnt = 0.0, None, 0
    n_batches = math.ceil(len(train_turns) / args.batch_turns)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng_ep = np.random.default_rng(args.seed + epoch + fold_idx * 1000)
        shuffled = rng_ep.permutation(train_turns).tolist()
        total_loss, n_groups = 0.0, 0

        for batch_start in range(0, len(shuffled), args.batch_turns):
            batch = shuffled[batch_start: batch_start + args.batch_turns]
            opt.zero_grad()
            # one forward pass over all candidates in the batch, then split per group
            Xcat   = torch.cat([groups[t][0] for t in batch], dim=0).to(device)
            sizes  = [groups[t][0].shape[0] for t in batch]
            scores_all = model(Xcat)
            batch_loss = torch.tensor(0.0, device=device)
            count, off = 0, 0
            for ti, t in enumerate(batch):
                sz = sizes[ti]
                loss = listnet_loss(scores_all[off:off+sz], groups[t][1].to(device))
                off += sz
                if loss is not None:
                    batch_loss = batch_loss + loss
                    count += 1
            if count > 0:
                (batch_loss / count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                total_loss += batch_loss.item()
                n_groups   += count

        sched.step()
        avg_loss = total_loss / max(n_groups, 1)

        val_scores = evaluate(model, val_turns)
        ndcg20 = val_scores["ndcg20"]
        print(f"  fold {fold_idx+1}  epoch {epoch:2d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  val ndcg@20={ndcg20:.4f}  "
              f"ndcg@10={val_scores['ndcg10']:.4f}  ndcg@1={val_scores['ndcg1']:.4f}")

        if ndcg20 > best_ndcg + 1e-5:
            best_ndcg  = ndcg20
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, {"ndcg20": best_ndcg, "best_epoch": epoch - patience_cnt}

# ── Cross-validation ──────────────────────────────────────────────────────────

fold_results = []
fold_models  = []

for fi in range(args.n_folds):
    val_sessions   = set(folds[fi])
    train_sessions = [s for f in (folds[:fi] + folds[fi+1:]) for s in f]
    train_turns = [t for s in train_sessions for t in session_to_turns[s]]
    val_turns   = [t for s in val_sessions   for t in session_to_turns[s]]

    print(f"\n{'='*60}")
    print(f"Fold {fi+1}/{args.n_folds}  "
          f"train_turns={len(train_turns)}  val_turns={len(val_turns)}")

    model, res = train_fold(train_turns, val_turns, fi)
    val_full = evaluate(model, val_turns)
    res.update(val_full)
    res["n_train"] = len(train_turns)
    res["n_val"]   = len(val_turns)
    fold_results.append(res)
    fold_models.append(model)
    print(f"Fold {fi+1} best  ndcg@20={res['ndcg20']:.4f}  "
          f"ndcg@10={res['ndcg10']:.4f}  ndcg@1={res['ndcg1']:.4f}")

cv_ndcg20 = float(np.mean([r["ndcg20"] for r in fold_results]))
cv_std    = float(np.std([r["ndcg20"] for r in fold_results]))
print(f"\nCV ndcg@20: {cv_ndcg20:.4f}  std={cv_std:.4f}")

# ── Final model: retrain on all turns ────────────────────────────────────────
print(f"\nFinal refit on all {n_turns} turns ...")
all_turns = list(range(n_turns))

# Warm-start from the best CV fold
best_fold = int(np.argmax([r["ndcg20"] for r in fold_results]))
final_model = ListwiseMLP(n_feats, hidden_sizes, args.dropout).to(device)
final_model.load_state_dict(
    {k: v.to(device) for k, v in fold_models[best_fold].state_dict().items()}
)
opt   = optim.Adam(final_model.parameters(), lr=args.lr * 0.5,
                   weight_decay=args.weight_decay)
sched = optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=max(fold_results, key=lambda r: r["ndcg20"])["best_epoch"], eta_min=1e-5
)
target_epochs = max(fold_results, key=lambda r: r["ndcg20"])["best_epoch"]

for epoch in range(1, target_epochs + 1):
    final_model.train()
    rng_ep = np.random.default_rng(args.seed + epoch + 9999)
    shuffled = rng_ep.permutation(all_turns).tolist()
    total_loss, n_groups = 0.0, 0
    for batch_start in range(0, len(shuffled), args.batch_turns):
        batch = shuffled[batch_start: batch_start + args.batch_turns]
        opt.zero_grad()
        Xcat   = torch.cat([groups[t][0] for t in batch], dim=0).to(device)
        sizes  = [groups[t][0].shape[0] for t in batch]
        scores_all = final_model(Xcat)
        batch_loss = torch.tensor(0.0, device=device)
        count, off = 0, 0
        for ti, t in enumerate(batch):
            sz = sizes[ti]
            loss = listnet_loss(scores_all[off:off+sz], groups[t][1].to(device))
            off += sz
            if loss is not None:
                batch_loss = batch_loss + loss
                count += 1
        if count > 0:
            (batch_loss / count).backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += batch_loss.item()
            n_groups   += count
    sched.step()
    if epoch % 5 == 0 or epoch == target_epochs:
        print(f"  refit epoch {epoch}/{target_epochs}  loss={total_loss/max(n_groups,1):.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

torch.save(final_model.state_dict(), out_dir / "model.pt")

scaler = {"mean": X_mean.tolist(), "std": X_std.tolist(), "feature_cols": feature_cols}
with open(out_dir / "scaler.json", "w") as f:
    json.dump(scaler, f)

imp = {}  # feature importance via input gradient magnitude on full dataset
final_model.eval()
X_sample = torch.from_numpy(X_norm[:min(500_000, len(X_norm))]).to(device).requires_grad_(True)
scores_s = final_model(X_sample).sum()
scores_s.backward()
grad_mag = X_sample.grad.abs().mean(dim=0).cpu().numpy()
imp = {feature_cols[i]: float(grad_mag[i]) for i in range(n_feats)}
imp_sorted = dict(sorted(imp.items(), key=lambda x: -x[1]))

with open(out_dir / "meta.json", "w") as f:
    json.dump({
        "feature_cols":   feature_cols,
        "n_feats":        n_feats,
        "hidden":         hidden_sizes,
        "dropout":        args.dropout,
        "soft_labels":    args.soft_labels,
        "poly_feats":     args.poly_feats,
        "cv_ndcg20_mean": cv_ndcg20,
        "cv_ndcg20_std":  cv_std,
        "fold_results":   fold_results,
        "target_epochs":  target_epochs,
        "feature_importance_grad": imp_sorted,
        "params": {
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "batch_turns":  args.batch_turns,
            "dropout":      args.dropout,
        },
    }, f, indent=2)

print(f"\nSaved: {out_dir}/model.pt")
print(f"       {out_dir}/scaler.json")
print(f"       {out_dir}/meta.json")
print(f"\nTop-10 features by gradient magnitude:")
for name, val in list(imp_sorted.items())[:10]:
    print(f"  {name:<28}  {val:.6f}")
