"""BERT / DistilBERT text encoder with multi-label classification head."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 50,
        dropout: float = 0.2,
        freeze: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.hidden_size = hidden
        self.num_labels = num_labels
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, n: int = 2) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        # DistilBERT: transformer.layer
        layers = None
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "layer"):
            layers = self.backbone.transformer.layer
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer
        if layers is not None:
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # Always train embeddings lightly optional — keep frozen

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls [B,H], token_states [B,L,H])."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls = hidden[:, 0]
        return cls, hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cls, token_states = self.forward_hidden(input_ids, attention_mask)
        logits = self.classifier(self.dropout(cls))
        return {"logits": logits, "cls": cls, "token_states": token_states}


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def encode_texts(
    texts: List[str],
    tokenizer,
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
