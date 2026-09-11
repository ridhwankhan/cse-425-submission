# CSE425 Submission Package

**Course:** CSE425 Neural Networks, Section 02, BRAC University  
**Project:** GNN-Based BERT for Understanding Context from Music

| Name | Student ID |
|------|------------|
| Ridhwanur Rahman Khan | 22301143 |
| Jannatul Bushra Maisha | 21301498 |

## Contents

- `gnn-bert-music-context/` — full project (source, examples, results, report, notebooks)
- `CSE425_Sec02_22301143_21301498_GNN_BERT_Music.zip` — packaged archive of the same materials

## Assignment checklist

1. Full source: `gnn-bert-music-context/src/`, `scripts/`, `config.yaml`, `requirements.txt`
2. Example graphs (≥20): `gnn-bert-music-context/data/processed/examples/`
3. Metrics + plots + retrieval: `gnn-bert-music-context/results/`
4. Final report: `gnn-bert-music-context/report/main.pdf` (LaTeX source: `main.tex`)
5. Demo notebook: `gnn-bert-music-context/notebooks/demo_context.ipynb`

## Report

- LaTeX: `gnn-bert-music-context/report/main.tex`
- Compiled PDF: `gnn-bert-music-context/report/main.pdf`
- Figures: `gnn-bert-music-context/report/figures/`

To recompile: from `report/`, run `pdflatex main.tex` twice (or upload `main.tex`, `report.sty`, and `figures/` to Overleaf).

## Quick run

```bash
cd gnn-bert-music-context
pip install -r requirements.txt
python scripts/run_all.py
python -m src.evaluate --task all
python scripts/extra_analysis.py
```

Note: large model checkpoints under `results/checkpoints/*.pt` are omitted from git (GitHub 100 MB limit).
