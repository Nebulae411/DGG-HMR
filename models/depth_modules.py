'''
"""
bbox+roi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
 

class ROITzHead(nn.Module):
    """ROI-based Tz head operating on DAV2 path_1 features.

    This module takes a batch of spatial feature maps and a set of normalized
    per-person bounding boxes (cx, cy, w, h in [0, 1]) and returns:
        - pred_tz: [B, Nq, 1] per-person root depth predictions
        - query_tokens: [B, Nq, hidden_dim] per-person tokens for HMR query injection
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        roi_size: int = 56,
        num_layers: int = 2,
        nheads: int = 4,
        dim_feedforward: int = 1024,
        min_tz: float = 0.3,
        max_tz: float = 50.0,
        context_scale: float = 1.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size
        self.min_tz = min_tz
        self.max_tz = max_tz
        self.context_scale = context_scale

        # 1) Channel adapter on DAV2 spatial features
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 2) Lightweight transformer decoder over flattened ROI tokens (cross-attention per ROI)
        self.query = nn.Embedding(1, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3) Per-ROI depth regression head (uses pooled token + bbox geometry)
        self.geom_dim = 4
        self.tz_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + self.geom_dim),
            nn.Linear(hidden_dim + self.geom_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for all submodules."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, feat: torch.Tensor, boxes: torch.Tensor):
        """Forward pass.

        Args:
            feat: DAV2 path_1 feature map, shape [B, C_in, H, W].
            boxes: normalized bounding boxes in cxcywh format, shape [B, Nq, 4],
                   with values in [0, 1] (same normalization as training targets).
        Returns:
            outputs: dict with keys:
                - 'pred_tz': [B, Nq, 1]
                - 'query_tokens': [B, Nq, hidden_dim]
        """
        if feat is None or boxes is None:
            return None

        B, C, H, W = feat.shape
        assert C == self.in_channels, "Input feature channels must match in_channels"
        B_boxes, Nq, _ = boxes.shape
        assert B_boxes == B, "Batch size of boxes must match feature maps"

        # Adapt features to hidden_dim
        feat_adapt = self.spatial_adapter(feat)  # [B, hidden_dim, H, W]

        # Convert normalized cxcywh boxes to xyxy in feature-map coordinates
        boxes = boxes.clamp(0.0, 1.0)
        cx = boxes[..., 0] * W
        cy = boxes[..., 1] * H
        bw = boxes[..., 2] * W
        bh = boxes[..., 3] * H

        # Optionally enlarge ROI with extra context around each box
        context_scale = getattr(self, "context_scale", 1.0)
        bw_ctx = bw * context_scale
        bh_ctx = bh * context_scale

        x1 = (cx - 0.5 * bw_ctx).clamp(0.0, float(W - 1))
        y1 = (cy - 0.5 * bh_ctx).clamp(0.0, float(H - 1))
        x2 = (cx + 0.5 * bw_ctx).clamp(0.0, float(W - 1))
        y2 = (cy + 0.5 * bh_ctx).clamp(0.0, float(H - 1))

        batch_indices = torch.arange(B, device=feat_adapt.device).view(B, 1).repeat(1, Nq)
        rois = torch.stack([batch_indices, x1, y1, x2, y2], dim=-1).reshape(-1, 5)

        # RoIAlign over adapted features
        roi_feats = roi_align(
            feat_adapt,
            rois,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )  # [B*Nq, hidden_dim, roi_size, roi_size]

        # Flatten spatial dimensions into a token sequence for each ROI
        roi_feats = roi_feats.flatten(2).transpose(1, 2)  # [B*Nq, L, hidden_dim]

        # Cross-attention: a single learnable query attends to all ROI tokens
        bs_rois = roi_feats.size(0)
        query = self.query.weight.unsqueeze(0).repeat(bs_rois, 1, 1)  # [B*Nq, 1, hidden_dim]
        decoder_out = self.decoder(query, roi_feats)  # [B*Nq, 1, hidden_dim]
        pooled = decoder_out[:, 0, :]  # [B*Nq, hidden_dim]

        # Depth regression in inverse-depth style range [min_tz, max_tz]
        # Concatenate pooled ROI token with normalized bbox geometry (cx, cy, w, h)
        geom = boxes.view(B * Nq, 4)  # [B*Nq, 4] in [0, 1]
        tz_input = torch.cat([pooled, geom], dim=-1)  # [B*Nq, hidden_dim + 4]
        tz_raw = self.tz_head(tz_input)  # [B*Nq, 1]
        tz = torch.sigmoid(tz_raw) * (self.max_tz - self.min_tz) + self.min_tz

        pred_tz = tz.view(B, Nq, 1)
        query_tokens = pooled.view(B, Nq, self.hidden_dim)

        return {
            'pred_tz': pred_tz,
            'query_tokens': query_tokens,
        }
'''


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
