# GNN-BERT Music Context Understanding

CSE425 Neural Networks (Section 02) - BRAC University

**Group members**

| Name |  |
|------|-----|
| Ridhwanur Rahman Khan | |
| Jannatul Bushra Maisha | |

This project implements a hybrid DistilBERT + GraphSAGE model for music context understanding: multi-label tagging, genre classification, multimodal fusion, and caption-audio retrieval (MusicCaps).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Edit `config.yaml` for paths, model size, and training settings.

## Data

```bash
python scripts/download_fma.py --subset small
python scripts/download_musiccaps.py
python scripts/download_deam.py
python scripts/prepare_splits.py
python scripts/build_graphs.py --dataset all
python scripts/export_example_graphs.py --n 20
```

If a dataset is temporarily unavailable, `scripts/make_dev_subset.py` can build a small local subset for debugging.

## Training and evaluation

```bash
python -m src.train --task 1
python -m src.train --task 2
python -m src.train --task 3
python -m src.train --task 4
# or
python scripts/run_all.py
python -m src.evaluate --task all
```

## Project layout

```
src/           model code (BERT, GNN, fusion, contrastive, train, evaluate)
scripts/       download, preprocessing, graph export
data/          splits, processed example graphs
notebooks/     eda.ipynb, demo_context.ipynb
results/       metrics.json, plots, retrieval examples
report/        main.tex + main.pdf + figures/
```

## Results

See `results/metrics.json` and figures under `results/plots/`.  
Demo: `notebooks/demo_context.ipynb`.  
Report: `report/main.pdf`.

