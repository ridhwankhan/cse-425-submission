"""Training entrypoint for Tasks 1–4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python -m src.train` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import MelCNN
from src.bert_encoder import BertEncoder, encode_texts, get_tokenizer
from src.contrastive import ContrastiveDualEncoder, info_nce_loss
from src.data_deam import DEAMDataset
from src.data_fma import FMAGraphDataset
from src.data_musiccaps import MusicCapsGraphDataset, MusicCapsTextDataset
from src.fusion_model import FusionModel
from src.gnn_model import MusicGNN
from src.utils import (
    ensure_dirs,
    get_device,
    load_config,
    load_json,
    multilabel_f1_auc,
    multiclass_metrics,
    save_json,
    set_seed,
)


def _limit_records(recs: List[Dict], n: int) -> List[Dict]:
    if n and n > 0:
        return recs[:n]
    return recs


def _node_dim_from_sample(sample_x: torch.Tensor) -> int:
    return int(sample_x.shape[-1])


def train_task1(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    splits = load_json(Path(cfg["paths"]["splits"]) / "musiccaps.json")
    vocab = splits["vocab"]
    n = int(cfg["debug"].get("n_tracks", 0))
    train_ds = MusicCapsTextDataset(_limit_records(splits["train"], n), require_audio=False)
    val_ds = MusicCapsTextDataset(_limit_records(splits["val"], n), require_audio=False)
    if len(train_ds) == 0:
        raise RuntimeError("No MusicCaps train records. Run data scripts first.")

    tokenizer = get_tokenizer(cfg["model"]["bert_name"])
    model = BertEncoder(
        cfg["model"]["bert_name"],
        num_labels=len(vocab),
        dropout=cfg["model"]["dropout"],
        freeze=True,
    ).to(device)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": [], "val_auc_pr": []}
    max_len = cfg["data"]["max_text_len"]
    epochs = cfg["train"]["epochs"]
    bs = cfg["train"]["batch_size"]

    for epoch in range(epochs):
        if epoch == cfg["model"].get("freeze_bert_epochs", 2):
            model.unfreeze_last_layers(cfg["model"].get("unfreeze_bert_layers", 2))
            opt = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": cfg["train"]["bert_lr"]},
                    {"params": model.classifier.parameters(), "lr": cfg["train"]["lr"]},
                ],
                weight_decay=cfg["train"]["weight_decay"],
            )
        model.train()
        losses = []
        order = np.random.permutation(len(train_ds))
        for start in tqdm(range(0, len(order), bs), desc=f"task1 ep{epoch+1}"):
            idxs = order[start : start + bs]
            batch = [train_ds[int(i)] for i in idxs]
            texts = [b["text"] for b in batch]
            y = torch.stack([b["label"] for b in batch]).to(device)
            enc = encode_texts(texts, tokenizer, max_len, device)
            out = model(enc["input_ids"], enc["attention_mask"])
            loss = criterion(out["logits"], y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            losses.append(loss.item())
        history["train_loss"].append(float(np.mean(losses)))

        # val
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for start in range(0, len(val_ds), bs):
                batch = [val_ds[i] for i in range(start, min(start + bs, len(val_ds)))]
                if not batch:
                    break
                enc = encode_texts([b["text"] for b in batch], tokenizer, max_len, device)
                logits = model(enc["input_ids"], enc["attention_mask"])["logits"]
                prob = torch.sigmoid(logits).cpu().numpy()
                ys.append(torch.stack([b["label"] for b in batch]).numpy())
                ps.append(prob)
        if ys:
            y_true = np.concatenate(ys, axis=0)
            y_prob = np.concatenate(ps, axis=0)
            m = multilabel_f1_auc(y_true, y_prob, cfg["eval"]["threshold"])
            history["val_macro_f1"].append(m["macro_f1"])
            history["val_micro_f1"].append(m["micro_f1"])
            history["val_auc_pr"].append(m["mean_auc_pr"])
        else:
            history["val_macro_f1"].append(0.0)
            history["val_micro_f1"].append(0.0)
            history["val_auc_pr"].append(0.0)

    ckpt = Path(cfg["paths"]["checkpoints"]) / "task1_bert.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab, "history": history}, ckpt)

    # 5 example predictions
    model.eval()
    examples = []
    with torch.no_grad():
        for i in range(min(5, len(val_ds))):
            b = val_ds[i]
            enc = encode_texts([b["text"]], tokenizer, max_len, device)
            prob = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])["logits"])[0].cpu().numpy()
            top = np.argsort(-prob)[:5]
            examples.append(
                {
                    "ytid": b["ytid"],
                    "text": b["text"][:200],
                    "pred_tags": [{"tag": vocab[j], "score": float(prob[j])} for j in top],
                    "true_tags": [vocab[j] for j, v in enumerate(b["label"].tolist()) if v > 0.5][:10],
                }
            )
    save_json(examples, Path(cfg["paths"]["results"]) / "task1_predictions.json")
    save_json(history, Path(cfg["paths"]["results"]) / "task1_history.json")
    return {"history": history, "checkpoint": str(ckpt)}


def train_task2(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    subset = cfg["data"]["fma_subset"]
    splits = load_json(Path(cfg["paths"]["splits"]) / f"fma_{subset}.json")
    n = int(cfg["debug"].get("n_tracks", 0))
    train_rec = _limit_records(splits["train"], n)
    val_rec = _limit_records(splits["val"], n)
    train_ds = FMAGraphDataset(train_rec, Path(cfg["paths"]["processed"]), subset)
    val_ds = FMAGraphDataset(val_rec, Path(cfg["paths"]["processed"]), subset)
    if len(train_ds) == 0:
        raise RuntimeError("No FMA train records with graphs. Run build_graphs.py first.")
    if len(val_ds) == 0:
        # Use a held-out slice of train graphs if val was not cached
        n_hold = max(1, min(32, len(train_ds) // 5))
        val_ds = torch.utils.data.Subset(train_ds, list(range(len(train_ds) - n_hold, len(train_ds))))
        # Subset breaks attribute access used below — wrap as list view via simple proxy
        class _ListDS:
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, i):
                return self.items[i]
        val_ds = _ListDS([train_ds[i] for i in range(len(train_ds) - n_hold, len(train_ds))])
        train_ds = _ListDS([train_ds[i] for i in range(0, len(train_ds) - n_hold)])
        print(f"Task2: empty val graphs; using holdout of {len(val_ds)} from train")

    sample = train_ds[0]
    in_dim = _node_dim_from_sample(sample["x"])
    num_classes = len(splits.get("genres", [])) or cfg["data"]["num_genres_small"]

    gnn = MusicGNN(
        in_dim=in_dim,
        hidden=cfg["model"]["gnn_hidden"],
        num_layers=cfg["model"]["gnn_layers"],
        num_classes=num_classes,
        gnn_type=cfg["model"]["gnn_type"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    cnn = MelCNN(n_mels=cfg["data"]["n_mels"], num_classes=num_classes, dropout=cfg["model"]["dropout"]).to(device)

    opt_g = torch.optim.AdamW(gnn.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    opt_c = torch.optim.AdamW(cnn.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    history = {"gnn_loss": [], "cnn_loss": [], "gnn_val_macro_f1": [], "cnn_val_macro_f1": []}

    for epoch in range(cfg["train"]["epochs"]):
        gnn.train()
        cnn.train()
        g_losses, c_losses = [], []
        order = np.random.permutation(len(train_ds))
        for i in tqdm(order, desc=f"task2 ep{epoch+1}"):
            b = train_ds[int(i)]
            x = b["x"].to(device)
            ei = b["edge_index"].to(device)
            y = torch.tensor([b["label"]], device=device)
            mel = b["mel"].unsqueeze(0).to(device)

            out_g = gnn(x, ei)
            loss_g = criterion(out_g["logits"].unsqueeze(0), y)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())

            out_c = cnn(mel)
            loss_c = criterion(out_c["logits"], y)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            c_losses.append(loss_c.item())

        history["gnn_loss"].append(float(np.mean(g_losses)))
        history["cnn_loss"].append(float(np.mean(c_losses)))

        # val
        gnn.eval()
        cnn.eval()
        yg, pg, yc, pc = [], [], [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                b = val_ds[i]
                lg = gnn(b["x"].to(device), b["edge_index"].to(device))["logits"]
                lc = cnn(b["mel"].unsqueeze(0).to(device))["logits"][0]
                yg.append(b["label"])
                pg.append(int(lg.argmax().cpu()))
                yc.append(b["label"])
                pc.append(int(lc.argmax().cpu()))
        history["gnn_val_macro_f1"].append(multiclass_metrics(np.array(yg), np.array(pg))["macro_f1"])
        history["cnn_val_macro_f1"].append(multiclass_metrics(np.array(yc), np.array(pc))["macro_f1"])

    ckpt = Path(cfg["paths"]["checkpoints"])
    torch.save({"model": gnn.state_dict(), "in_dim": in_dim, "num_classes": num_classes}, ckpt / "task2_gnn.pt")
    torch.save({"model": cnn.state_dict(), "num_classes": num_classes}, ckpt / "task2_cnn.pt")
    save_json(history, Path(cfg["paths"]["results"]) / "task2_history.json")
    return {"history": history}


def train_task3(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    # Prefer context_pairs (MusicCaps audio or FMA-backed pairs); fall back to musiccaps.json
    pairs_path = Path(cfg["paths"]["splits"]) / "context_pairs.json"
    mc_path = Path(cfg["paths"]["splits"]) / "musiccaps.json"
    splits = load_json(pairs_path if pairs_path.exists() else mc_path)
    vocab = splits["vocab"]
    n = int(cfg["debug"].get("n_tracks", 0))
    train_ds = MusicCapsGraphDataset(_limit_records(splits["train"], n), Path(cfg["paths"]["processed"]), require_audio=True)
    val_ds = MusicCapsGraphDataset(_limit_records(splits["val"], n), Path(cfg["paths"]["processed"]), require_audio=True)
    if len(train_ds) == 0:
        raise RuntimeError("No paired graph+text samples for Task 3. Build graphs and prepare_splits first.")

    sample = train_ds[0]
    in_dim = _node_dim_from_sample(sample["x"])
    modes = ["cross_attention", "concat", "bert_only", "gnn_only"]
    tokenizer = get_tokenizer(cfg["model"]["bert_name"])
    all_hist = {}
    # Positive-class upweight for sparse multi-label aspects
    y_stack = torch.stack([train_ds[i]["label"] for i in range(len(train_ds))])
    pos = y_stack.sum(dim=0).clamp(min=1.0)
    neg = (len(train_ds) - pos).clamp(min=1.0)
    pos_weight = (neg / pos).clamp(1.0, 20.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    max_len = cfg["data"]["max_text_len"]

    # Optional DEAM
    deam_path = Path(cfg["paths"]["splits"]) / "deam.json"
    deam_ds = None
    if deam_path.exists():
        deam = load_json(deam_path)
        if deam.get("train"):
            deam_ds = DEAMDataset(_limit_records(deam["train"], n), Path(cfg["paths"]["processed"]))

    for mode in modes:
        model = FusionModel(
            in_node_dim=in_dim,
            num_labels=len(vocab),
            bert_name=cfg["model"]["bert_name"],
            gnn_hidden=cfg["model"]["gnn_hidden"],
            gnn_layers=cfg["model"]["gnn_layers"],
            gnn_type=cfg["model"]["gnn_type"],
            fusion_dim=cfg["model"]["fusion_dim"],
            dropout=cfg["model"]["dropout"],
            mode=mode,
            emotion=True,
        ).to(device)
        model.bert.freeze_backbone()
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        hist = {"loss": [], "val_macro_f1": []}
        for epoch in range(cfg["train"]["epochs"]):
            if epoch == cfg["model"].get("freeze_bert_epochs", 2) and mode != "gnn_only":
                model.bert.unfreeze_last_layers(cfg["model"].get("unfreeze_bert_layers", 2))
            model.train()
            losses = []
            order = np.random.permutation(len(train_ds))
            for i in tqdm(order, desc=f"task3 {mode} ep{epoch+1}"):
                b = train_ds[int(i)]
                enc = encode_texts([b["text"]], tokenizer, max_len, device)
                y = b["label"].unsqueeze(0).to(device)
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    x=b["x"].to(device),
                    edge_index=b["edge_index"].to(device),
                )
                loss = criterion(out["logits"], y)
                if deam_ds is not None and len(deam_ds) > 0 and "emotion" in out:
                    db = deam_ds[int(i) % len(deam_ds)]
                    # Use same text encoding path for DEAM emotion aux on fused z from DEAM graph
                    denc = encode_texts([db["text"]], tokenizer, max_len, device)
                    dout = model(
                        input_ids=denc["input_ids"],
                        attention_mask=denc["attention_mask"],
                        x=db["x"].to(device),
                        edge_index=db["edge_index"].to(device),
                    )
                    target = db["emotion"].unsqueeze(0).to(device)
                    # scale DEAM targets roughly to similar range if needed
                    loss = loss + cfg["train"]["alpha_valence"] * nn.functional.l1_loss(
                        dout["emotion"][:, 0], target[:, 0]
                    ) + cfg["train"]["beta_arousal"] * nn.functional.l1_loss(dout["emotion"][:, 1], target[:, 1])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            hist["loss"].append(float(np.mean(losses)))

            model.eval()
            ys, ps = [], []
            zs, labels_for_tsne = [], []
            with torch.no_grad():
                for i in range(len(val_ds)):
                    b = val_ds[i]
                    enc = encode_texts([b["text"]], tokenizer, max_len, device)
                    out = model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        x=b["x"].to(device),
                        edge_index=b["edge_index"].to(device),
                    )
                    prob = torch.sigmoid(out["logits"]).cpu().numpy()
                    ys.append(b["label"].numpy())
                    ps.append(prob[0])
                    zs.append(out["z"].cpu().numpy()[0])
                    labels_for_tsne.append(int(b["label"].sum().item()))
            if ys:
                m = multilabel_f1_auc(np.stack(ys), np.stack(ps), cfg["eval"]["threshold"])
                hist["val_macro_f1"].append(m["macro_f1"])
            else:
                hist["val_macro_f1"].append(0.0)

        ckpt = Path(cfg["paths"]["checkpoints"]) / f"task3_fusion_{mode}.pt"
        torch.save({"model": model.state_dict(), "mode": mode, "in_dim": in_dim, "vocab": vocab}, ckpt)
        all_hist[mode] = hist

        # Save embeddings for t-SNE from last mode's last epoch (overwrite; final = cross_attention preferred)
        if mode == "cross_attention" and zs:
            np.savez(
                Path(cfg["paths"]["results"]) / "task3_embeddings.npz",
                z=np.stack(zs),
                label_count=np.array(labels_for_tsne),
            )

    save_json(all_hist, Path(cfg["paths"]["results"]) / "task3_ablation_history.json")

    # Case studies
    cases = []
    model = FusionModel(
        in_node_dim=in_dim,
        num_labels=len(vocab),
        bert_name=cfg["model"]["bert_name"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        mode="cross_attention",
    ).to(device)
    ck = Path(cfg["paths"]["checkpoints"]) / "task3_fusion_cross_attention.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(val_ds))):
            b = val_ds[i]
            enc = encode_texts([b["text"]], tokenizer, max_len, device)
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                x=b["x"].to(device),
                edge_index=b["edge_index"].to(device),
            )
            prob = torch.sigmoid(out["logits"])[0].cpu().numpy()
            top = np.argsort(-prob)[:5]
            cases.append(
                {
                    "ytid": b["ytid"],
                    "caption": b["text"][:300],
                    "num_nodes": int(b["x"].shape[0]),
                    "num_edges": int(b["edge_index"].shape[1]),
                    "pred_aspects": [vocab[j] for j in top],
                    "note": "Graph segment nodes connected by temporal/similarity edges; caption aligned via cross-attention.",
                }
            )
    save_json(cases, Path(cfg["paths"]["results"]) / "task3_case_studies.json")
    return {"ablations": all_hist}


def train_task4(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    pairs_path = Path(cfg["paths"]["splits"]) / "context_pairs.json"
    mc_path = Path(cfg["paths"]["splits"]) / "musiccaps.json"
    splits = load_json(pairs_path if pairs_path.exists() else mc_path)
    n = int(cfg["debug"].get("n_tracks", 0))
    train_ds = MusicCapsGraphDataset(_limit_records(splits["train"], n), Path(cfg["paths"]["processed"]), require_audio=True)
    val_ds = MusicCapsGraphDataset(_limit_records(splits["val"], n), Path(cfg["paths"]["processed"]), require_audio=True)
    if len(train_ds) == 0:
        raise RuntimeError("No paired graph+text samples for Task 4.")
    sample = train_ds[0]
    in_dim = _node_dim_from_sample(sample["x"])
    model = ContrastiveDualEncoder(
        in_node_dim=in_dim,
        bert_name=cfg["model"]["bert_name"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        gnn_layers=cfg["model"]["gnn_layers"],
        gnn_type=cfg["model"]["gnn_type"],
        proj_dim=cfg["model"]["contrastive_dim"],
        temperature=cfg["model"]["temperature"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.bert.freeze_backbone()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["train"]["lr"])
    tokenizer = get_tokenizer(cfg["model"]["bert_name"])
    max_len = cfg["data"]["max_text_len"]
    bs = max(2, cfg["train"]["batch_size"])
    history = {"loss": []}

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        losses = []
        order = np.random.permutation(len(train_ds))
        for start in tqdm(range(0, len(order), bs), desc=f"task4 ep{epoch+1}"):
            idxs = order[start : start + bs]
            if len(idxs) < 2:
                continue
            batch = [train_ds[int(i)] for i in idxs]
            enc = encode_texts([b["text"] for b in batch], tokenizer, max_len, device)
            graph_zs = []
            for b in batch:
                gz = model.encode_graph(b["x"].to(device), b["edge_index"].to(device))
                graph_zs.append(gz)
            graph_z = torch.stack(graph_zs, dim=0)
            text_z = model.encode_text(enc["input_ids"], enc["attention_mask"])
            logits = graph_z @ text_z.t() / model.temperature
            loss = info_nce_loss(logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)

    ckpt = Path(cfg["paths"]["checkpoints"]) / "task4_contrastive.pt"
    torch.save({"model": model.state_dict(), "in_dim": in_dim}, ckpt)
    save_json(history, Path(cfg["paths"]["results"]) / "task4_history.json")

    # Retrieval pool: val + test (+ train head) so small val sets still yield ≥10 examples
    model.eval()
    from src.utils import recall_at_k

    test_ds = MusicCapsGraphDataset(
        _limit_records(splits.get("test", []), n),
        Path(cfg["paths"]["processed"]),
        require_audio=True,
    )
    pool_recs = list(range(len(val_ds)))
    # Build a combined dataset view for qualitative retrieval
    combined = []
    for ds in (val_ds, test_ds, train_ds):
        for i in range(len(ds)):
            combined.append(ds[i])
            if len(combined) >= 32:
                break
        if len(combined) >= 32:
            break

    with torch.no_grad():
        gz_list, tz_list, meta = [], [], []
        for b in combined:
            enc = encode_texts([b["text"]], tokenizer, max_len, device)
            gz_list.append(model.encode_graph(b["x"].to(device), b["edge_index"].to(device)).cpu().numpy())
            tz_list.append(model.encode_text(enc["input_ids"], enc["attention_mask"])[0].cpu().numpy())
            meta.append({"ytid": b["ytid"], "text": b["text"][:200]})
        G = np.stack(gz_list)
        T = np.stack(tz_list)
        sim = G @ T.T
        metrics = {
            "caption_to_audio": {f"R@{k}": recall_at_k(sim.T, k) for k in (1, 5, 10)},
            "audio_to_caption": {f"R@{k}": recall_at_k(sim, k) for k in (1, 5, 10)},
            "pool_size": len(meta),
        }
        # qualitative examples (always aim for 10)
        examples = []
        out_dir = Path(cfg["paths"]["results"]) / "retrieval_examples"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(10, len(meta))):
            top = np.argsort(-sim.T[i])[:3]  # caption i -> top audio
            ex = {
                "query_caption": meta[i]["text"],
                "query_ytid": meta[i]["ytid"],
                "top3_audio": [
                    {"ytid": meta[j]["ytid"], "score": float(sim.T[i, j]), "caption": meta[j]["text"]}
                    for j in top
                ],
            }
            examples.append(ex)
            save_json(ex, out_dir / f"example_{i+1:02d}.json")
        save_json(examples, out_dir / "all_examples.json")
        save_json(metrics, Path(cfg["paths"]["results"]) / "task4_retrieval.json")
    return {"history": history, "retrieval": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="1|2|3|4|all")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)
    print(f"Device: {device}")

    tasks = ["1", "2", "3", "4"] if args.task == "all" else [args.task]
    results = {}
    for t in tasks:
        print(f"\n=== Training Task {t} ===")
        if t == "1":
            results["task1"] = train_task1(cfg, device)
        elif t == "2":
            results["task2"] = train_task2(cfg, device)
        elif t == "3":
            results["task3"] = train_task3(cfg, device)
        elif t == "4":
            results["task4"] = train_task4(cfg, device)
        else:
            raise ValueError(f"Unknown task {t}")
    save_json({k: "ok" for k in results}, Path(cfg["paths"]["results"]) / "train_status.json")
    print("Done.")


if __name__ == "__main__":
    main()
