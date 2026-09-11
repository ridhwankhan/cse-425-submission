"""GNN encoders (GraphSAGE / GAT) with pure-PyTorch fallback if PyG missing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _has_pyg() -> bool:
    try:
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


class SageConvPure(nn.Module):
    """GraphSAGE mean-aggregate: h' = σ(W · concat(h, mean(N(h))))."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        if edge_index.numel() > 0:
            agg.index_add_(0, dst, x[src])
            ones = torch.ones(src.shape[0], device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, ones)
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        mean_n = agg / deg
        out = self.lin(torch.cat([x, mean_n], dim=-1))
        return F.relu(out)


class GATConvPure(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.zeros(1, heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_dim)
        src, dst = edge_index[0], edge_index[1]
        if edge_index.numel() == 0:
            return h.mean(dim=1)
        h_src, h_dst = h[src], h[dst]
        cat = torch.cat([h_src, h_dst], dim=-1)
        e = (cat * self.attn).sum(dim=-1)
        e = F.leaky_relu(e, 0.2)
        # softmax per destination per head
        out = torch.zeros(n, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        for head in range(self.heads):
            eh = e[:, head]
            # numerical stability per dst
            max_per_dst = torch.full((n,), -1e9, device=x.device, dtype=x.dtype)
            max_per_dst.scatter_reduce_(0, dst, eh, reduce="amax", include_self=False)
            eh = eh - max_per_dst[dst]
            alpha = eh.exp()
            denom = torch.zeros(n, device=x.device, dtype=x.dtype)
            denom.index_add_(0, dst, alpha)
            alpha = alpha / (denom[dst] + 1e-8)
            weighted = h_src[:, head] * alpha.unsqueeze(-1)
            out[:, head].index_add_(0, dst, weighted)
        return out.mean(dim=1)


class MusicGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        gnn_type: str = "sage",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.use_pyg = _has_pyg()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.use_pyg:
                from torch_geometric.nn import GATConv, SAGEConv

                if gnn_type == "gat":
                    self.layers.append(GATConv(hidden, hidden // 2, heads=2, concat=True))
                else:
                    self.layers.append(SAGEConv(hidden, hidden))
            else:
                if gnn_type == "gat":
                    self.layers.append(GATConvPure(hidden, hidden))
                else:
                    self.layers.append(SageConvPure(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.hidden = hidden

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv in self.layers:
            h = conv(h, edge_index)
            h = self.dropout(h)
        # mean pool
        g = h.mean(dim=0, keepdim=True)
        return g.squeeze(0), h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        g, node_h = self.encode(x, edge_index)
        logits = self.classifier(self.dropout(g))
        return {"logits": logits, "graph_emb": g, "node_emb": node_h}


def collate_graphs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Simple list collate — train loops handle graphs one-by-one or padded."""
    return {"items": batch}
