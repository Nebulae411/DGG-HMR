"""Depth-related decoder modules."""

import torch
import torch.nn as nn

from utils.misc import inverse_sigmoid
from .decoder import TransformerDecoder
from .position_encoding import position_encoding_xy


class DepthOnlyTzDecoder(nn.Module):
    """Depth-only Tz decoder conditioned on bbox prior queries.

    This module reuses the shared TransformerDecoder (DAB-DETR style) but
    removes any bbox prediction head. It takes:
        - depth features: [B, C, H, W] (already adapted by a depth interface)
        - prior_boxes: [B, Nq, 4] normalized cxcywh from bbox prior (detached)
        - prior_tokens: [B, Nq, C_prior] query tokens from bbox prior (detached)

    It returns a per-person depth prediction and updated query tokens for
    optional injection into the main HMR decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        nheads: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        min_tz: float = 0.3,
        max_tz: float = 50.0,
        prior_token_dim: int = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.min_tz = min_tz
        self.max_tz = max_tz

        # Lightweight DETR-style decoder without bbox head (no refpoint updates).
        self.decoder = TransformerDecoder(
            d_model=hidden_dim,
            nhead=nheads,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            return_intermediate_dec=True,
            query_dim=4,
            keep_query_pos=False,
            query_scale_type="cond_elewise",
            modulate_hw_attn=False,
            bbox_embed_diff_each_layer=False,
        )

        if prior_token_dim is None or prior_token_dim == hidden_dim:
            self.prior_query_proj = None
        else:
            self.prior_query_proj = nn.Linear(prior_token_dim, hidden_dim)

        # Depth regression head operating on final-layer query tokens.
        self.tz_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        feat: torch.Tensor,
        prior_boxes: torch.Tensor,
        prior_tokens: torch.Tensor,
    ):
        """Forward pass.

        Args:
            feat: depth features [B, C, H, W] from a depth interface.
            prior_boxes: [B, Nq, 4] normalized cxcywh from bbox prior.
            prior_tokens: [B, Nq, C_prior] query tokens from bbox prior.
        Returns:
            outputs: dict with keys:
                - 'pred_tz': [B, Nq, 1]
                - 'query_tokens': [B, Nq, hidden_dim]
        """
        if feat is None or prior_boxes is None or prior_tokens is None:
            return None

        B, C, H, W = feat.shape
        B_boxes, Nq, _ = prior_boxes.shape
        B_tokens, Nq_tokens, _ = prior_tokens.shape
        assert B_boxes == B and B_tokens == B, "Batch dimension mismatch"
        assert Nq_tokens == Nq, "Number of queries must match between boxes and tokens"
        assert C == self.hidden_dim, "Depth feature channels must match hidden_dim"

        # Flatten spatial dimensions into a sequence for the decoder memory.
        memory = feat.flatten(2).permute(2, 0, 1).flatten(0, 1)  # [B*H*W, C]
        memory_lens = [H * W] * B

        # Positional encodings for depth features.
        ys, xs = torch.meshgrid(
            torch.arange(H, device=feat.device),
            torch.arange(W, device=feat.device),
            indexing="ij",
        )
        pos_y = (ys.flatten().to(feat.dtype) + 0.5) / float(H)
        pos_x = (xs.flatten().to(feat.dtype) + 0.5) / float(W)
        pos_embed = position_encoding_xy(pos_x, pos_y, embedding_dim=self.hidden_dim)  # [H*W, C]
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1).flatten(0, 1)  # [B*H*W, C]

        # Use bbox prior outputs as fixed reference points and queries (no gradients
        # flow back into the prior branch).
        prior_boxes = prior_boxes.detach()
        prior_tokens = prior_tokens.detach()

        refpoints_unsigmoid = inverse_sigmoid(prior_boxes).flatten(0, 1)  # [B*Nq, 4]

        tgt = prior_tokens
        if self.prior_query_proj is not None:
            tgt = self.prior_query_proj(tgt)
        tgt = tgt.flatten(0, 1)  # [B*Nq, hidden_dim]
        tgt_lens = [Nq] * B

        hs, _ = self.decoder(
            memory=memory,
            memory_lens=memory_lens,
            tgt=tgt,
            tgt_lens=tgt_lens,
            refpoint_embed=refpoints_unsigmoid,
            pos_embed=pos_embed,
            self_attn_mask=None,
        )

        # hs: [num_layers, B, Nq, hidden_dim]
        if hs.dim() == 3:
            # Fallback for potential legacy shapes [num_layers, B*Nq, C].
            hs = hs.view(hs.shape[0], B, Nq, self.hidden_dim)

        last_hs = hs[-1]  # [B, Nq, hidden_dim]

        tz_raw = self.tz_head(last_hs)  # [B, Nq, 1]
        tz = torch.sigmoid(tz_raw) * (self.max_tz - self.min_tz) + self.min_tz

        return {
            'pred_tz': tz,
            'query_tokens': last_hs,
        }
