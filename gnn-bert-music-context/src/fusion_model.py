"""GNN–BERT fusion: early concat and cross-attention."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import BertEncoder
from .gnn_model import MusicGNN


class CrossAttentionFusion(nn.Module):
    """Q from graph readout; K/V from BERT token states (PDF Task 3)."""

    def __init__(self, graph_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.Wq = nn.Linear(graph_dim, out_dim, bias=False)
        self.Wk = nn.Linear(text_dim, out_dim, bias=False)
        self.Wv = nn.Linear(text_dim, out_dim, bias=False)
        self.out_dim = out_dim

    def forward(
        self,
        g: torch.Tensor,
        token_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # g: [B, Dg] or [Dg]; token_states: [B, L, Dt]
        if g.dim() == 1:
            g = g.unsqueeze(0)
        if token_states.dim() == 2:
            token_states = token_states.unsqueeze(0)
        Q = self.Wq(g).unsqueeze(1)  # [B,1,D]
        K = self.Wk(token_states)  # [B,L,D]
        V = self.Wv(token_states)
        scale = self.out_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale  # [B,1,L]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e9)
        A = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(A, V).squeeze(1)  # [B,D]
        z = torch.cat([g, ctx], dim=-1)
        return z


class FusionModel(nn.Module):
    def __init__(
        self,
        in_node_dim: int,
        num_labels: int,
        bert_name: str = "distilbert-base-uncased",
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        gnn_type: str = "sage",
        fusion_dim: int = 256,
        dropout: float = 0.2,
        mode: str = "cross_attention",  # cross_attention | concat | bert_only | gnn_only
        emotion: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.bert = BertEncoder(bert_name, num_labels=num_labels, dropout=dropout)
        self.gnn = MusicGNN(
            in_dim=in_node_dim,
            hidden=gnn_hidden,
            num_layers=gnn_layers,
            num_classes=num_labels,
            gnn_type=gnn_type,
            dropout=dropout,
        )
        text_dim = self.bert.hidden_size
        self.cross = CrossAttentionFusion(gnn_hidden, text_dim, fusion_dim)
        if mode == "cross_attention":
            fused_in = gnn_hidden + fusion_dim
        elif mode == "concat":
            fused_in = gnn_hidden + text_dim
        elif mode == "bert_only":
            fused_in = text_dim
        else:
            fused_in = gnn_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_labels),
        )
        self.emotion = emotion
        if emotion:
            self.emotion_head = nn.Linear(fused_in, 2)  # valence, arousal

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        g, _ = self.gnn.encode(x, edge_index)
        return g

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        cls: Optional[torch.Tensor] = None,
        token_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.mode in ("cross_attention", "concat", "bert_only") and cls is None:
            assert input_ids is not None and attention_mask is not None
            bert_out = self.bert(input_ids, attention_mask)
            cls, token_states = bert_out["cls"], bert_out["token_states"]

        if self.mode in ("cross_attention", "concat", "gnn_only") and graph_emb is None:
            assert x is not None and edge_index is not None
            graph_emb = self.encode_graph(x, edge_index)

        if self.mode == "bert_only":
            z = cls
        elif self.mode == "gnn_only":
            z = graph_emb if graph_emb.dim() == 2 else graph_emb.unsqueeze(0)
            if cls is not None and z.size(0) != cls.size(0):
                z = z.expand(cls.size(0), -1)
        elif self.mode == "concat":
            if graph_emb.dim() == 1:
                graph_emb = graph_emb.unsqueeze(0)
            if graph_emb.size(0) == 1 and cls.size(0) > 1:
                graph_emb = graph_emb.expand(cls.size(0), -1)
            z = torch.cat([graph_emb, cls], dim=-1)
        else:
            if graph_emb.dim() == 1:
                graph_emb = graph_emb.unsqueeze(0)
            z = self.cross(graph_emb, token_states, attention_mask)

        logits = self.head(z)
        out: Dict[str, torch.Tensor] = {"logits": logits, "z": z}
        if self.emotion and hasattr(self, "emotion_head"):
            out["emotion"] = self.emotion_head(z)
        return out
