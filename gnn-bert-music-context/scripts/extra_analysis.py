"""Extra analyses required by the assignment: zero-shot tags, DEAM emotion, graph coherence, plots."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bert_encoder import encode_texts, get_tokenizer
from src.data_deam import DEAMDataset
from src.fusion_model import FusionModel
from src.utils import ensure_dirs, get_device, load_config, load_json, mae_r2, multilabel_f1_auc, save_json, set_seed


def zero_shot_tag_prediction(cfg: dict, device) -> dict:
    """Predict tags by cosine similarity between caption CLS and tag-name embeddings (no task head)."""
    splits = load_json(Path(cfg["paths"]["splits"]) / "musiccaps.json")
    vocab = splits["vocab"]
    n = int(cfg["debug"].get("n_tracks", 0)) or None
    test = splits["test"][:n]
    if not test:
        test = splits["val"][:n]
    tokenizer = get_tokenizer(cfg["model"]["bert_name"])
    from transformers import AutoModel

    bert = AutoModel.from_pretrained(cfg["model"]["bert_name"]).to(device)
    bert.eval()
    with torch.no_grad():
        tag_enc = encode_texts(vocab, tokenizer, cfg["data"]["max_text_len"], device)
        tag_h = bert(tag_enc["input_ids"], attention_mask=tag_enc["attention_mask"]).last_hidden_state[:, 0]
        tag_h = F.normalize(tag_h, dim=-1)

        ys, ps = [], []
        for rec in test:
            enc = encode_texts([rec.get("text") or rec.get("caption", "")], tokenizer, cfg["data"]["max_text_len"], device)
            cls = bert(enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
            cls = F.normalize(cls, dim=-1)
            sim = (cls @ tag_h.t()).squeeze(0).cpu().numpy()
            # map similarity to [0,1] via sigmoid-scaled shift
            prob = 1.0 / (1.0 + np.exp(-5.0 * (sim - 0.2)))
            ys.append(np.asarray(rec["label"], dtype=np.float32))
            ps.append(prob.astype(np.float32))
    y_true = np.stack(ys)
    y_prob = np.stack(ps)
    zs = multilabel_f1_auc(y_true, y_prob, cfg["eval"]["threshold"])

    # Supervised Task 3 (if checkpoint exists)
    supervised = {}
    ck = Path(cfg["paths"]["checkpoints"]) / "task3_fusion_cross_attention.pt"
    if ck.exists():
        sample_dim = 2 * (cfg["data"]["n_mels"] + cfg["data"]["n_chroma"] + cfg["audio"]["n_mfcc"])
        model = FusionModel(
            in_node_dim=sample_dim,
            num_labels=len(vocab),
            bert_name=cfg["model"]["bert_name"],
            gnn_hidden=cfg["model"]["gnn_hidden"],
            mode="cross_attention",
        ).to(device)
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval()
        from src.data_musiccaps import MusicCapsGraphDataset

        ds = MusicCapsGraphDataset(test, Path(cfg["paths"]["processed"]), require_audio=False)
        ys2, ps2 = [], []
        with torch.no_grad():
            for i in range(len(ds)):
                b = ds[i]
                enc = encode_texts([b["text"]], tokenizer, cfg["data"]["max_text_len"], device)
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    x=b["x"].to(device),
                    edge_index=b["edge_index"].to(device),
                )
                ys2.append(b["label"].numpy())
                ps2.append(torch.sigmoid(out["logits"])[0].cpu().numpy())
        if ys2:
            supervised = multilabel_f1_auc(np.stack(ys2), np.stack(ps2), cfg["eval"]["threshold"])

    out = {"zero_shot_caption_to_tag": zs, "task3_supervised": supervised, "n_eval": len(test)}
    save_json(out, Path(cfg["paths"]["results"]) / "task4_zeroshot.json")

    # bar plot
    plt.figure(figsize=(6, 4))
    labels = ["Zero-shot\n(caption→tag)", "Task 3\nsupervised"]
    vals = [zs.get("macro_f1", 0.0), supervised.get("macro_f1", 0.0)]
    plt.bar(labels, vals, color=["#4C78A8", "#F58518"])
    plt.ylabel("Macro-F1")
    plt.title("Zero-shot vs supervised tag prediction")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots"]) / "task4_zeroshot.png", dpi=150)
    plt.close()
    return out


def deam_emotion_metrics(cfg: dict, device) -> dict:
    splits_path = Path(cfg["paths"]["splits"]) / "deam.json"
    if not splits_path.exists():
        return {}
    deam = load_json(splits_path)
    recs = deam.get("test") or deam.get("val") or []
    n = int(cfg["debug"].get("n_tracks", 0))
    if n:
        recs = recs[:n]
    if not recs:
        return {}
    ds = DEAMDataset(recs, Path(cfg["paths"]["processed"]))
    if len(ds) == 0:
        return {}
    ck = Path(cfg["paths"]["checkpoints"]) / "task3_fusion_cross_attention.pt"
    if not ck.exists():
        return {}
    sample = ds[0]
    model = FusionModel(
        in_node_dim=sample["x"].shape[-1],
        num_labels=50,
        bert_name=cfg["model"]["bert_name"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        mode="cross_attention",
        emotion=True,
    ).to(device)
    state = torch.load(ck, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    tokenizer = get_tokenizer(cfg["model"]["bert_name"])
    yt, yp = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            enc = encode_texts([b["text"]], tokenizer, cfg["data"]["max_text_len"], device)
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                x=b["x"].to(device),
                edge_index=b["edge_index"].to(device),
            )
            if "emotion" not in out:
                break
            yt.append(b["emotion"].numpy())
            yp.append(out["emotion"][0].cpu().numpy())
    if not yt:
        return {}
    yt = np.stack(yt)
    yp = np.stack(yp)
    out = {
        "valence": mae_r2(yt[:, 0], yp[:, 0]),
        "arousal": mae_r2(yt[:, 1], yp[:, 1]),
        "n": len(yt),
    }
    save_json(out, Path(cfg["paths"]["results"]) / "task3_emotion.json")
    return out


def graph_coherence(cfg: dict) -> dict:
    """Optional S_graph: fraction of edges with cosine(node_i, node_j) > tau."""
    ex = Path(cfg["paths"]["examples"])
    tau = float(cfg["graph"]["similarity_tau"])
    scores = []
    for pt in sorted(ex.glob("*.pt"))[:20]:
        g = torch.load(pt, map_location="cpu", weights_only=False)
        x = g["x"].float()
        ei = g["edge_index"].long()
        if ei.numel() == 0 or x.size(0) < 2:
            continue
        x = F.normalize(x, dim=-1)
        src, dst = ei[0], ei[1]
        cos = (x[src] * x[dst]).sum(dim=-1)
        scores.append(float((cos > tau).float().mean().item()))
    out = {"mean_S_graph": float(np.mean(scores)) if scores else 0.0, "n_graphs": len(scores), "tau": tau}
    save_json(out, Path(cfg["paths"]["results"]) / "graph_coherence.json")
    return out


def plot_auc_pr_bars(cfg: dict) -> None:
    metrics = load_json(Path(cfg["paths"]["results"]) / "metrics.json")
    items = []
    if "baselines" in metrics:
        items.append(("Random", metrics["baselines"].get("random_auc_pr")))
        items.append(("Majority", metrics["baselines"].get("majority_auc_pr")))
    t1 = metrics.get("model", {}).get("task1_bert", {})
    items.append(("BERT Task1", t1.get("auc_pr")))
    labels = [a for a, b in items if b is not None]
    vals = [b for a, b in items if b is not None]
    if not vals:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color="#54A24B")
    plt.ylabel("Mean AUC-PR")
    plt.title("Tag ranking quality (AUC-PR)")
    plt.tight_layout()
    plt.savefig(Path(cfg["paths"]["plots"]) / "auc_pr_bars.png", dpi=150)
    plt.close()


def plot_architecture(cfg: dict) -> None:
    """Simple block diagram of the system."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=1.2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, weight="bold")

    box(0.2, 1.7, 1.8, 0.9, "Audio\nmel/chroma", "#A0C4FF")
    box(0.2, 0.4, 1.8, 0.9, "Text\ncaption/tags", "#FFADAD")
    box(2.4, 1.7, 1.8, 0.9, "Segment\ngraph", "#BDB2FF")
    box(2.4, 0.4, 1.8, 0.9, "DistilBERT", "#FFC6FF")
    box(4.6, 1.7, 1.8, 0.9, "GraphSAGE\nGNN", "#CAFFBF")
    box(4.6, 0.4, 1.8, 0.9, "CLS / tokens", "#FDFFB6")
    box(6.8, 0.85, 1.6, 1.2, "Fusion\ncross-attn", "#FFD6A5")
    box(8.6, 0.85, 1.2, 1.2, "Heads\ntags/\nemotion/\nretrieve", "#E0E0E0")

    for x0, y0, x1, y1 in [
        (2.0, 2.15, 2.4, 2.15),
        (2.0, 0.85, 2.4, 0.85),
        (4.2, 2.15, 4.6, 2.15),
        (4.2, 0.85, 4.6, 0.85),
        (6.4, 2.15, 6.8, 1.6),
        (6.4, 0.85, 6.8, 1.3),
        (8.4, 1.45, 8.6, 1.45),
    ]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.set_title("GNN–BERT music context pipeline", fontsize=12, pad=8)
    fig.tight_layout()
    out = Path(cfg["paths"]["plots"]) / "architecture.png"
    fig.savefig(out, dpi=200)
    plt.close()
    # also copy into report/figures
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(out, fig_dir / "architecture.png")
    for name in [
        "task1_f1_curves.png",
        "task2_f1_curves.png",
        "task3_ablations.png",
        "task3_tsne.png",
        "task4_retrieval.png",
        "auc_pr_bars.png",
        "task4_zeroshot.png",
    ]:
        src = Path(cfg["paths"]["plots"]) / name
        if src.exists():
            shutil.copy2(src, fig_dir / name)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)
    print("device", device)
    print("zero-shot:", zero_shot_tag_prediction(cfg, device))
    print("deam:", deam_emotion_metrics(cfg, device))
    print("coherence:", graph_coherence(cfg))
    plot_auc_pr_bars(cfg)
    # merge into metrics.json
    metrics_path = Path(cfg["paths"]["results"]) / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    zs = Path(cfg["paths"]["results"]) / "task4_zeroshot.json"
    emo = Path(cfg["paths"]["results"]) / "task3_emotion.json"
    coh = Path(cfg["paths"]["results"]) / "graph_coherence.json"
    if zs.exists():
        metrics["zero_shot"] = load_json(zs)
    if emo.exists():
        metrics["emotion"] = load_json(emo)
        # fill comparison table MAE
        mae = load_json(emo).get("valence", {}).get("mae")
        if "comparison_table" in metrics and mae is not None:
            metrics["comparison_table"].setdefault("Task 3: GNN–BERT", {})["MAE (emotion)"] = mae
    if coh.exists():
        metrics["graph_coherence"] = load_json(coh)
    save_json(metrics, metrics_path)
    plot_architecture(cfg)
    print("Wrote extra analyses and figures.")


if __name__ == "__main__":
    main()
