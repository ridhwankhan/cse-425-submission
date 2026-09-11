"""Contrastive dual-encoder GNN–BERT (InfoNCE) for MusicCaps."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import BertEncoder
from .gnn_model import MusicGNN


class ContrastiveDualEncoder(nn.Module):
    def __init__(
        self,
        in_node_dim: int,
        bert_name: str = "distilbert-base-uncased",
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        gnn_type: str = "sage",
        proj_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.bert = BertEncoder(bert_name, num_labels=1, dropout=dropout)
        # discard classifier usage
        self.gnn = MusicGNN(
            in_dim=in_node_dim,
            hidden=gnn_hidden,
            num_layers=gnn_layers,
            num_classes=1,
            gnn_type=gnn_type,
            dropout=dropout,
        )
        self.graph_proj = nn.Linear(gnn_hidden, proj_dim)
        self.text_proj = nn.Linear(self.bert.hidden_size, proj_dim)

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        g, _ = self.gnn.encode(x, edge_index)
        z = self.graph_proj(g)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls, _ = self.bert.forward_hidden(input_ids, attention_mask)
        z = self.text_proj(cls)
        return F.normalize(z, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_embs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """graph_embs: [B, D] already projected+normalized or raw gnn hidden before proj.

        Prefer calling encode_* externally for variable graphs; this assumes
        graph_embs are L2-normalized projections [B, proj_dim].
        """
        text_z = self.encode_text(input_ids, attention_mask)
        # If graph_embs are raw gnn hidden:
        if graph_embs.size(-1) != text_z.size(-1):
            graph_z = F.normalize(self.graph_proj(graph_embs), dim=-1)
        else:
            graph_z = graph_embs
        logits = graph_z @ text_z.t() / self.temperature
        return {"logits": logits, "graph_z": graph_z, "text_z": text_z}


def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE: graph→text + text→graph."""
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)
