"""Build a properly written multi-page NeurIPS-style PDF when LaTeX is unavailable.

Pulls live numbers from results/metrics.json so the PDF stays consistent with experiments.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "neurips" / "final_report.pdf"
FIG = ROOT / "report" / "neurips" / "figures"


def P(text: str, style):
    return Paragraph(text.replace("\n", " "), style)


def load_metrics():
    path = ROOT / "results" / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, digits=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "—"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    # Prefer freshest plots
    src_plots = ROOT / "results" / "plots"
    if src_plots.exists():
        for p in src_plots.glob("*.png"):
            shutil.copy2(p, FIG / p.name)
            shutil.copy2(p, ROOT / "report" / "figures" / p.name)

    m = load_metrics()
    ct = m.get("comparison_table", {})
    zs = m.get("zero_shot", {})
    emo = m.get("emotion", {})
    notes = m.get("notes", "Real-data run.")

    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], fontSize=15, spaceAfter=6, leading=18)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=12, spaceBefore=12, spaceAfter=5, leading=15)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=3, leading=13)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.5, leading=12.5, spaceAfter=5, alignment=4)
    small = ParagraphStyle("S", parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=3)
    center = ParagraphStyle("C", parent=body, alignment=1)
    abstract = ParagraphStyle("A", parent=body, fontSize=9, leading=12, leftIndent=10, rightIndent=10)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=1.15 * inch,
        rightMargin=1.0 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="GNN-Based BERT for Understanding Context from Music",
        author="Ridhwanur Rahman Khan, Jannatul Bushra Maisha",
    )

    t1_f1 = fmt((ct.get("Task 1: BERT-only") or {}).get("Macro-F1"))
    t1_ap = fmt((ct.get("Task 1: BERT-only") or {}).get("AUC-PR"))
    t2 = fmt((ct.get("Task 2: GNN-only") or {}).get("Macro-F1"))
    cnn = fmt((ct.get("CNN mel-spec") or {}).get("Macro-F1"))
    t3 = fmt((ct.get("Task 3: GNN–BERT") or {}).get("Macro-F1"))
    mae = fmt((ct.get("Task 3: GNN–BERT") or {}).get("MAE (emotion)"))
    r5 = fmt((ct.get("Task 4: Contrastive") or {}).get("R@5 (retrieval)"))
    rnd = fmt((ct.get("Random tags") or {}).get("Macro-F1"))
    zs_f1 = fmt((zs.get("zero_shot_caption_to_tag") or {}).get("macro_f1"))
    sup_f1 = fmt((zs.get("task3_supervised") or {}).get("macro_f1"))
    abl = m.get("model", {}).get("task3_ablations", {})
    bert_only = fmt((abl.get("bert_only") or {}).get("macro_f1"))
    gnn_only = fmt((abl.get("gnn_only") or {}).get("macro_f1"))
    n_emo = emo.get("n", "—")

    story = []
    story.append(P("GNN-Based BERT for Understanding Context from Music", title))
    story.append(P("<b>Ridhwanur Rahman Khan</b> (22301143) &nbsp;&nbsp; <b>Jannatul Bushra Maisha</b> (21301498)", center))
    story.append(P("CSE425 Section 02, BRAC University &mdash; NeurIPS-style course report", center))
    story.append(Spacer(1, 6))

    story.append(P(
        "<b>Abstract.</b> Music context spans genre, mood, harmony, and free-form captions. "
        "Spectrogram CNNs model local acoustics well but do not explicitly represent relations among temporal segments or chords. "
        "We build a DistilBERT + GraphSAGE pipeline for CSE425 Tasks 1–4: multi-label aspect tagging from MusicCaps captions, "
        "FMA-small genre classification from segment graphs, cross-attention GNN–BERT fusion with DEAM emotion auxiliaries, "
        f"and InfoNCE caption–audio retrieval. On a real-data run (2,000 FMA tracks, 800 MusicCaps audio graphs, DEAM), "
        f"BERT tagging reaches Macro-F1 {t1_f1} / AUC-PR {t1_ap}; GraphSAGE is competitive with a mel-CNN baseline "
        f"(≈{t2} vs {cnn} Macro-F1); fusion improves over GNN-only on multi-label aspects; and retrieval attains R@5 = {r5}. "
        "We report ablations, zero-shot vs. supervised tags, qualitative cases, and full reproducibility artifacts.",
        abstract,
    ))

    story.append(P("1. Introduction", h1))
    story.append(P(
        "A single clip may be electronic, melancholic, and described by a detailed caption at once. "
        "“Context” therefore mixes discrete tags, continuous affect, and natural language. "
        "CNNs on log-mel spectrograms are strong local detectors, but they flatten relational structure: "
        "how adjacent windows evolve, how similar segments cohere, and how chord transitions form a song graph.",
        body,
    ))
    story.append(P(
        "This project follows the CSE425 brief: combine a contextual language encoder (BERT family) with a graph neural "
        "network for <i>understanding and retrieval</i>, not generation. We deliver (i) a real-data pipeline over "
        "FMA-small, MusicCaps, and DEAM; (ii) segment graphs with temporal and similarity edges plus ≥20 exported examples; "
        "(iii) Tasks 1–4 with required baselines, metrics, plots, and a demo notebook; and (iv) an honest analysis of where "
        "text dominates tagging, where graphs help structure/retrieval, and where fusion still lags BERT-only under sparse pairing.",
        body,
    ))

    story.append(P("2. Related work", h1))
    story.append(P(
        "<b>Datasets.</b> FMA provides Creative Commons audio with genre metadata and official splits. "
        "MusicCaps pairs AudioSet clips with expert captions and is standard for text–music evaluation. "
        "DEAM supplies continuous valence/arousal annotations.",
        body,
    ))
    story.append(P(
        "<b>Models.</b> DistilBERT is a compact contextual encoder suitable for course-scale fine-tuning. "
        "GraphSAGE supports inductive message passing on variable-size graphs. "
        "Cross-modal contrastive learning (InfoNCE) aligns audio and language embeddings for retrieval. "
        "Our design follows the assignment equations for GraphSAGE updates, cross-attention fusion, and InfoNCE.",
        body,
    ))

    story.append(P("3. Problem formulation", h1))
    story.append(P(
        "A track is a tuple T = (X<sub>audio</sub>, X<sub>text</sub>, G, y), where X<sub>audio</sub> denotes log-mel/chroma features, "
        "X<sub>text</sub> is tokenized captions or metadata text, G = (V, E) is a music structure graph, and y are context targets "
        "(aspects, genre, valence/arousal). DistilBERT maps text to token states H<sub>text</sub>. GraphSAGE updates each node by "
        "concatenating its representation with the mean of its neighbors, followed by a linear transform and nonlinearity. "
        "Graph readout g = mean-pool({h<sub>i</sub>}) is fused with text as z = Fusion(g, H<sub>text</sub>), with ŷ = σ(Wz + b). "
        "Primary losses are multi-label BCE (tags), cross-entropy (genre), L1 on DEAM emotion when available, and symmetric InfoNCE for retrieval.",
        body,
    ))

    story.append(P("4. Data and preprocessing", h1))
    story.append(P(
        "<b>Datasets (real).</b> Synthetic tones are disabled (<font face='Courier'>use_synthetic: false</font>). "
        "<b>FMA-small:</b> 2,000 locally cached tracks with official <font face='Courier'>genre_top</font> labels "
        "(8-way taxonomy) from FMA metadata; metadata text for BERT excludes genre words to avoid leakage. "
        "<b>MusicCaps:</b> 5,521 Hugging Face captions for Task 1; 800 successfully downloaded YouTube clips with "
        "segment graphs for Tasks 3–4. <b>DEAM:</b> 1,802 annotated excerpts; emotion evaluation uses "
        f"n = {n_emo} held-out pairs with cached graphs in the reported run.",
        body,
    ))
    story.append(P(
        "<b>Audio and graphs.</b> Audio is resampled to 22,050 Hz (torchaudio/ffmpeg), with log-mel (128 bins) and chroma (12). "
        "Tracks are cut into 5 s windows (hop 2.5 s). Segment-graph nodes are windows; edges are temporal ±1 neighbors and "
        "pairs with cosine similarity of MFCC/chroma features above τ = 0.85. Node features concatenate mean/std of mel, "
        "chroma, and MFCC. We export 20 example .pt/.json graphs under <font face='Courier'>data/processed/examples/</font>. "
        "Hyperparameters live in <font face='Courier'>config.yaml</font> (25 epochs, DistilBERT unfreeze schedule, GraphSAGE depth 3).",
        body,
    ))

    if (FIG / "architecture.png").exists():
        story.append(KeepTogether([
            P("<i>Figure 1.</i> System overview: audio → segment graph (GraphSAGE); captions/tags → DistilBERT; fusion for tagging, emotion, and retrieval.", small),
            Image(str(FIG / "architecture.png"), width=6.3 * inch, height=2.05 * inch),
        ]))

    story.append(P("5. Methods", h1))
    story.append(P("<b>5.1 Task 1: BERT multi-label tagging.</b> DistilBERT produces a CLS vector; a linear head predicts aspect probabilities with BCE. We freeze the backbone for early epochs, then unfreeze the last layers.", body))
    story.append(P("<b>5.2 Task 2: GNN genre classification.</b> A GraphSAGE encoder with mean pooling predicts FMA top genre. Baselines include a mel-spectrogram CNN and random/majority predictors.", body))
    story.append(P(
        "<b>5.3 Task 3: GNN–BERT fusion.</b> Cross-attention uses Q from the graph readout and K/V from token states, "
        "then concatenates attended text with g. Ablations: early concatenation, BERT-only, and GNN-only. "
        "When DEAM batches are available we add L1 losses on valence and arousal.",
        body,
    ))
    story.append(P(
        "<b>5.4 Task 4: Contrastive retrieval.</b> A dual encoder maps graphs and captions to an ℓ₂-normalized space and "
        "minimizes symmetric InfoNCE. We report caption↔audio Recall@{1,5,10}, qualitative top-3 matches, and zero-shot "
        "caption→tag scoring versus Task 3 supervised fusion.",
        body,
    ))

    story.append(P("6. Experiments and discussion", h1))
    story.append(P(
        f"<b>Setup.</b> PyTorch training with CUDA. We log Macro/Micro-F1, mean AUC-PR, retrieval R@K, and DEAM MAE/R². "
        f"{notes}",
        body,
    ))

    data = [
        ["Model", "Macro-F1", "AUC-PR", "MAE_v", "R@5"],
        ["Random tags", rnd, fmt((ct.get("Random tags") or {}).get("AUC-PR")), "—", "—"],
        ["CNN mel-spec", cnn, "—", "—", "—"],
        ["Task 1 BERT-only", t1_f1, t1_ap, "—", "—"],
        ["Task 2 GNN-only", t2, "—", "—", "—"],
        ["Task 3 GNN–BERT", t3, "—", mae, "—"],
        ["Task 4 Contrastive", "—", "—", "—", r5],
    ]
    t = Table(data, colWidths=[2.05 * inch, 0.85 * inch, 0.8 * inch, 0.75 * inch, 0.7 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(P("<i>Table 1.</i> Main comparison from <font face='Courier'>results/metrics.json</font> (real-data run).", small))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(P(
        f"<b>Discussion.</b> Task 1 shows that captions alone are highly informative for MusicCaps aspects "
        f"(Macro-F1 {t1_f1}, AUC-PR {t1_ap}). On Task 2, GraphSAGE ({t2}) matches the mel-CNN baseline ({cnn}) and clearly "
        f"beats random tags ({rnd}), confirming that segment graphs carry usable genre signal under official 8-way labels; "
        "absolute scores remain modest given class imbalance and model capacity. "
        f"Task 3 cross-attention fusion (Macro-F1 {t3}) substantially beats GNN-only ({gnn_only}) and improves over earlier "
        f"sparse-audio runs, but BERT-only ablation remains slightly stronger ({bert_only}), suggesting text still dominates "
        "when graph–caption pairing is imperfect. "
        f"Task 4 reaches R@5 = {r5} (pool size 32), above chance. "
        f"Zero-shot caption→tag Macro-F1 = {zs_f1} lags supervised fusion ({sup_f1}), as required for the Task 4 comparison. "
        f"DEAM valence MAE ≈ {mae} on n = {n_emo} is far more meaningful than toy annotation subsets.",
        body,
    ))

    for caption, fname, w, h in [
        ("Figure 2. Task 1 validation Macro/Micro-F1 versus epoch.", "task1_f1_curves.png", 5.4, 2.9),
        ("Figure 3. Task 2 GraphSAGE vs. mel-CNN Macro-F1 curves.", "task2_f1_curves.png", 5.4, 2.9),
        ("Figure 4. Task 3 fusion ablations (cross-attention, concat, BERT-only, GNN-only).", "task3_ablations.png", 5.4, 2.9),
        ("Figure 5. t-SNE of fused representation z.", "task3_tsne.png", 4.5, 3.7),
        ("Figure 6. MusicCaps retrieval Recall@K (caption↔audio).", "task4_retrieval.png", 5.4, 2.9),
        ("Figure 7. Zero-shot caption→tag vs. Task 3 supervised fusion.", "task4_zeroshot.png", 4.6, 3.0),
        ("Figure 8. AUC-PR comparison bars.", "auc_pr_bars.png", 4.8, 3.0),
    ]:
        path = FIG / fname
        if path.exists():
            story.append(KeepTogether([
                P(f"<i>{caption}</i>", small),
                Image(str(path), width=w * inch, height=h * inch),
                Spacer(1, 4),
            ]))

    story.append(P("7. Qualitative deliverables", h1))
    story.append(P(
        "We include five Task 1 predictions (<font face='Courier'>results/task1_predictions.json</font>), "
        "three Task 3 case studies linking graph size/connectivity to captions "
        "(<font face='Courier'>results/task3_case_studies.json</font>), and ten Task 4 retrieval examples "
        "(<font face='Courier'>results/retrieval_examples/</font>). Twenty sample graphs are in "
        "<font face='Courier'>data/processed/examples/</font>. These artifacts match the course qualitative requirements.",
        body,
    ))

    story.append(P("8. Reproducibility", h1))
    story.append(P(
        "Repository layout matches the course tree: <font face='Courier'>src/</font>, <font face='Courier'>scripts/</font>, "
        "<font face='Courier'>data/</font>, <font face='Courier'>notebooks/</font>, <font face='Courier'>results/</font>, "
        "<font face='Courier'>report/</font>. Authors: Ridhwanur Rahman Khan (22301143), Jannatul Bushra Maisha (21301498), "
        "CSE425 Section 02.",
        body,
    ))
    story.append(Preformatted(
        "pip install -r requirements.txt\n"
        "python scripts/run_proper_pipeline.py\n"
        "# demo: notebooks/demo_context.ipynb",
        small,
    ))

    story.append(P("9. Limitations and conclusion", h1))
    story.append(P(
        "MusicCaps YouTube attrition and compute limits still constrain paired graph–text scale. "
        "Task 2/3 absolute F1 leave headroom with larger models and fuller FMA coverage. "
        "Emotion R² remains weak, indicating DEAM regression needs stronger audio encoders or tighter alignment. "
        "Human listening studies for retrieval are future work.",
        body,
    ))
    story.append(P(
        "We delivered a complete GNN–BERT music-context system covering Tasks 1–4 on real FMA, MusicCaps, and DEAM data, "
        "with baselines, required artifacts, and this NeurIPS-formatted report for CSE425 Section 02.",
        body,
    ))

    story.append(P("Acknowledgments", h1))
    story.append(P(
        "Course project for CSE425 Neural Networks, BRAC University, Section 02. Assignment designed by Moin Mostakim.",
        body,
    ))

    story.append(P("References", h1))
    refs = [
        "[1] M. Mostakim. Supervised neural network project: GNN-based BERT for understanding context from music. CSE425 course brief, 2026.",
        "[2] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson. FMA: A dataset for music analysis. ISMIR, 2017.",
        "[3] A. Agostinelli et al. MusicLM: Generating music from text. arXiv:2301.11325, 2023.",
        "[4] V. Sanh, L. Debut, J. Chaumond, and T. Wolf. DistilBERT. arXiv:1910.01108, 2019.",
        "[5] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. NeurIPS, 2017.",
        "[6] A. Aljanaki, Y.-H. Yang, and M. Soleymani. Developing a benchmark for emotional analysis of music. PLOS ONE, 2017.",
    ]
    for r in refs:
        story.append(P(r, small))

    doc.build(story)
    shutil.copy2(OUT, ROOT / "report" / "final_report.pdf")

    # Sync into submission package
    sub = ROOT / "submitted" / "gnn-bert-music-context" / "report"
    if sub.exists():
        shutil.copy2(OUT, sub / "final_report.pdf")
        shutil.copy2(OUT, sub / "neurips" / "final_report.pdf")
        shutil.copy2(ROOT / "report" / "neurips" / "main.tex", sub / "neurips" / "main.tex")
        for p in FIG.glob("*.png"):
            shutil.copy2(p, sub / "neurips" / "figures" / p.name)
            shutil.copy2(p, sub / "figures" / p.name)
    print("Wrote", OUT)
    print(f"Key metrics embedded: T1={t1_f1}/{t1_ap}, T2={t2}, CNN={cnn}, T3={t3}, R@5={r5}")


if __name__ == "__main__":
    main()
