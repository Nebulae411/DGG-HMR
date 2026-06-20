# Active DGG-HMR model.
# Standard DINOv2 + DETR decoder for human mesh recovery,
# with optional DAV2 depth and bbox/depth-prior branches.


import os
import math
from math import tan, pi
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import copy

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from utils.transforms import rot6d_to_axis_angle, img2patch_flat
from utils import constants
from configs.paths import smpl_mean_path

from models.encoders import build_encoder
from .matcher import build_matcher
from .decoder import build_decoder, TransformerDecoder
from .position_encoding import position_encoding_xy
from .criterion import SetCriterion
from .dn_components import prepare_for_cdn, dn_post_process

from configs.paths import smpl_model_path
from models.human_models import SMPL_Layer
from .depth_modules import DepthOnlyTzDecoder
from configs.paths import dav2_pretrained_paths

# Import DAV2 components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2
from .dav2_feature_extractor import patch_dav2_for_feature_extraction


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Model(nn.Module):
    """DGG-HMR multi-person human mesh estimation model."""
    def __init__(self, encoder, decoder,
                    num_queries,
                    input_size,
                    dn_cfg = {'use_dn': False},
                    train_pos_embed = True,
                    aux_loss=True, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=True,
                    random_refpoints_xy=False,
                    num_poses=24,
                    dim_shape=10,
                    FOV=pi/3,
                    use_dav2_depth=False,
                    dav2_encoder='vitb',
                    dav2_frozen=True,
                    use_bbox_prior=False,
                    bbox_prior_decoder_layers=3,
                    bbox_prior_hidden_dim=256,
                    bbox_prior_dim_feedforward=1024,
                    # Depth-only Tz decoder (DAV2 + bbox prior)
                    depth_only_tz_decoder=False,
                    depth_only_decoder_layers=2,
                    depth_only_nheads=4,
                    depth_only_dim_feedforward=1024,
                    depth_tz_hidden_dim=None,
                    depth_stage1_only=False,
                    ):
        """Initializes the DGG-HMR model.
        Parameters:
            encoder: torch module of the encoder to be used. See ./encoders.
            decoder: torch module of the decoder architecture. See decoder.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.depth_stage1_only = depth_stage1_only

        # ========== Start of common settings =============
        self.input_size = input_size
        hidden_dim = decoder.d_model
        num_dec_layers = decoder.dec_layers
        self.hidden_dim = hidden_dim
        # Make decoder available to all subsequent branches (including bbox prior)
        self.decoder = decoder
        
        # camera model
        self.focal = input_size/(2*tan(FOV/2))
        self.FOV = FOV
        cam_intrinsics = torch.tensor([[self.focal,0.,self.input_size/2],
                                            [0.,self.focal,self.input_size/2],
                                            [0.,0.,1.]])
        self.register_buffer('cam_intrinsics', cam_intrinsics)
        
        # human model
        self.num_poses = num_poses
        self.dim_shape = dim_shape
        self.human_model = SMPL_Layer(model_path = smpl_model_path, with_genders = False)
        
        # init params (following multi-hmr)
        smpl_mean_params = np.load(smpl_mean_path, allow_pickle = True)
        self.register_buffer('mean_pose', torch.from_numpy(smpl_mean_params['pose']))
        self.register_buffer('mean_shape', torch.from_numpy(smpl_mean_params['shape']))
        # ========== End of common settings =============

        # ========== Start of DGG encoder settings =============
        self.encoder = encoder
        self.patch_size = encoder.patch_size
        assert self.patch_size == 14
        
        # Single-level feature processing.
        assert self.input_size % self.patch_size == 0
        self.feature_size = self.input_size // self.patch_size
        
        # Copy patch embedding components from encoder
        self.encoder_patch_proj = copy.deepcopy(encoder.patch_embed.proj)
        self.encoder_patch_norm = copy.deepcopy(encoder.patch_embed.norm)
        
        # cls_token and register tokens
        encoder_cr_token = self.encoder.cls_token.view(1,-1) + self.encoder.pos_embed.float()[:,0].view(1,-1)
        if self.encoder.register_tokens is not None:
            encoder_cr_token = torch.cat([encoder_cr_token, self.encoder.register_tokens.view(self.encoder.num_register_tokens,-1)], dim=0)
        self.encoder_cr_token = nn.Parameter(encoder_cr_token)
        
        # Positional embeddings for single level
        self.encoder_pos_embeds = nn.Parameter(self.encoder.interpolate_pos_encoding3(self.feature_size).detach())
        if not train_pos_embed:
            self.encoder_pos_embeds.requires_grad = False
        
        # delete unwanted params from original encoder
        del(self.encoder.mask_token)
        del(self.encoder.pos_embed)
        del(self.encoder.patch_embed)
        del(self.encoder.cls_token)
        del(self.encoder.register_tokens)
        # ========== End of DGG encoder settings =============

        # ========== DAV2 Depth Backbone (Frozen Feature Extractor) =============
        self.use_dav2_depth = use_dav2_depth
        self.dav2_frozen = dav2_frozen
        if self.use_dav2_depth:
            # Initialize complete DepthAnythingV2 model (DINOv2 + DPT head)
            # Following BLADE's configuration
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            self.dav2_backbone = DepthAnythingV2(**model_configs[dav2_encoder])
            
            # Patch DAV2 to add optimized feature-only forward method
            # This avoids computing the final depth map when we only need features
            patch_dav2_for_feature_extraction(self.dav2_backbone)

            # Load DAV2 pretrained weights (same convention as official Depth-Anything-V2)
            ckpt_path = dav2_pretrained_paths[dav2_encoder]
            try:
                state = torch.load(ckpt_path, map_location='cpu')
                self.dav2_backbone.load_state_dict(state, strict=True)
                print(f"Loaded DAV2 pretrained weights from {ckpt_path}")
            except FileNotFoundError:
                print(f"[WARN] DAV2 checkpoint not found at {ckpt_path}, using random initialization.")
            except RuntimeError as e:
                print(f"[WARN] Failed to load DAV2 checkpoint from {ckpt_path}: {e}")
            
            # Freeze all DAV2 parameters if required (backbone + depth_head)
            if self.dav2_frozen:
                for param in self.dav2_backbone.parameters():
                    param.requires_grad = False
            
            # Store encoder type for accessing intermediate layers
            self.dav2_encoder = dav2_encoder
            
            # Get DAV2 output feature dimension
            dav2_feat_dim = {'vits': 64, 'vitb': 128, 'vitl': 256, 'vitg': 384}[dav2_encoder]
            self.dav2_feat_dim = dav2_feat_dim
        else:
            self.dav2_backbone = None
            self.dav2_feat_dim = None
        # ========== End of DAV2 settings =============

        # ========== Depth Tz hidden dimension (decoder feature space) =============
        # If not specified, default to DAV2 feature dimension.
        self.depth_tz_hidden_dim = depth_tz_hidden_dim if depth_tz_hidden_dim is not None else self.dav2_feat_dim

        # ========== Bbox-only prior DETR Branch (HMR encoder-based) =============
        self.use_bbox_prior = use_bbox_prior
        self.bbox_prior_hidden_dim = bbox_prior_hidden_dim
        self.bbox_prior_dim_feedforward = bbox_prior_dim_feedforward
        if self.use_bbox_prior:
            # Feature projection from main hidden_dim to bbox_prior_hidden_dim
            self.bbox_prior_feature_proj = nn.Linear(hidden_dim, self.bbox_prior_hidden_dim)
            # Positional embedding projection from main hidden_dim to bbox_prior_hidden_dim
            self.bbox_prior_pos_proj = nn.Linear(hidden_dim, self.bbox_prior_hidden_dim)

            # Lightweight transformer decoder in bbox_prior_hidden_dim space
            self.bbox_prior_decoder = TransformerDecoder(
                d_model=self.bbox_prior_hidden_dim,
                nhead=self.decoder.nhead,
                num_queries=num_queries,
                num_decoder_layers=bbox_prior_decoder_layers,
                dim_feedforward=self.bbox_prior_dim_feedforward,
                dropout=0.0,
                activation="relu",
                return_intermediate_dec=True,
                query_dim=query_dim,
                keep_query_pos=False,
                query_scale_type='cond_elewise',
                modulate_hw_attn=self.decoder.decoder.modulate_hw_attn,
                bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
            )

            # Bbox prediction heads for the prior branch (working in bbox_prior_hidden_dim space)
            if bbox_embed_diff_each_layer:
                self.bbox_prior_bbox_embed = nn.ModuleList([
                    MLP(self.bbox_prior_hidden_dim, self.bbox_prior_hidden_dim, 4, 3) for _ in range(bbox_prior_decoder_layers)
                ])
            else:
                self.bbox_prior_bbox_embed = MLP(self.bbox_prior_hidden_dim, self.bbox_prior_hidden_dim, 4, 3)

            # Attach bbox heads to the prior decoder for iterative reference updates
            self.bbox_prior_decoder.decoder.bbox_embed = self.bbox_prior_bbox_embed

            # Zero-initialize final bbox prediction layer(s) in the prior branch
            if bbox_embed_diff_each_layer:
                for bbox_embed in self.bbox_prior_bbox_embed:
                    nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                    nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
            else:
                nn.init.constant_(self.bbox_prior_bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(self.bbox_prior_bbox_embed.layers[-1].bias.data, 0)

            # Confidence head for bbox-only prior branch
            self.bbox_prior_conf_head = nn.Linear(self.bbox_prior_hidden_dim, 1)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.bbox_prior_conf_head.bias.data = torch.ones(1) * bias_value

            # Independent query and reference point embeddings for the prior branch
            self.bbox_prior_refpoint_embed = nn.Embedding(num_queries, query_dim)
            self.bbox_prior_tgt_embed = nn.Embedding(num_queries, self.bbox_prior_hidden_dim)
        else:
            self.bbox_prior_feature_proj = None
            self.bbox_prior_pos_proj = None
            self.bbox_prior_decoder = None
            self.bbox_prior_bbox_embed = None
            self.bbox_prior_conf_head = None
            self.bbox_prior_refpoint_embed = None
            self.bbox_prior_tgt_embed = None
        # ========== End of bbox-only prior settings =============

        # ========== Depth-only Tz decoder (DAV2 + bbox prior) =============
        self.use_depth_only_tz_decoder = depth_only_tz_decoder
        self.depth_tz_interface = None
        if self.use_depth_only_tz_decoder:
            assert self.use_dav2_depth, "Depth-only Tz decoder requires DAV2 depth features"
            assert self.use_bbox_prior, "Depth-only Tz decoder requires bbox prior branch"
            # Feature adaptation from DAV2 path_1 channels to depth decoder hidden space
            if self.dav2_feat_dim is not None and self.depth_tz_hidden_dim is not None:
                self.depth_tz_interface = nn.Sequential(
                    nn.Conv2d(self.dav2_feat_dim, self.depth_tz_hidden_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.depth_tz_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.depth_tz_hidden_dim, self.depth_tz_hidden_dim, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(self.depth_tz_hidden_dim),
                    nn.ReLU(inplace=True),
                )
            self.depth_only_tz_decoder = DepthOnlyTzDecoder(
                hidden_dim=self.depth_tz_hidden_dim,
                num_queries=num_queries,
                nheads=depth_only_nheads,
                num_decoder_layers=depth_only_decoder_layers,
                dim_feedforward=depth_only_dim_feedforward,
                min_tz=0.3,
                max_tz=50.0,
                prior_token_dim=self.bbox_prior_hidden_dim,
            )
        else:
            self.depth_only_tz_decoder = None
            self.depth_tz_interface = None
        # ========== End of depth-only Tz decoder settings =============

        # ========== Fusion projections for injecting priors into HMR decoder =============
        if self.use_bbox_prior:
            self.bbox_prior_fusion_proj = nn.Linear(query_dim, query_dim)
            nn.init.zeros_(self.bbox_prior_fusion_proj.weight)
            nn.init.zeros_(self.bbox_prior_fusion_proj.bias)
            self.bbox_prior_query_proj = nn.Linear(self.bbox_prior_hidden_dim, hidden_dim)
            self.bbox_prior_query_fusion_proj = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.bbox_prior_query_fusion_proj.weight)
            nn.init.zeros_(self.bbox_prior_query_fusion_proj.bias)
            self.prior_geo_token_mlp = MLP(6, hidden_dim, hidden_dim, 2)
        else:
            self.bbox_prior_fusion_proj = None
            self.bbox_prior_query_proj = None
            self.bbox_prior_query_fusion_proj = None
            self.prior_geo_token_mlp = None

        if getattr(self, "use_depth_only_tz_decoder", False) and self.depth_tz_hidden_dim is not None:
            self.depth_tz_query_proj = nn.Linear(self.depth_tz_hidden_dim, hidden_dim)
            self.depth_tz_query_fusion_proj = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.depth_tz_query_fusion_proj.weight)
            nn.init.zeros_(self.depth_tz_query_fusion_proj.bias)
        else:
            self.depth_tz_query_proj = None
            self.depth_tz_query_fusion_proj = None
        # ========== End of fusion projections =============

        # ========== Start of decoder settings =============
        self.num_queries = num_queries
        
        # embed_dim between encoder and decoder can be different
        self.feature_proj = nn.Linear(encoder.embed_dim, hidden_dim)

        # bbox
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # poses (use 6D rotation)
        self.pose_head = MLP(hidden_dim, hidden_dim, num_poses*6, 6)
        # shape
        self.shape_head = MLP(hidden_dim, hidden_dim, dim_shape, 5)
        # cam_trans
        self.cam_head = MLP(hidden_dim, hidden_dim//2, 3, 3)
        # confidence score
        self.conf_head = nn.Linear(hidden_dim, 1)
        
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.conf_head.bias.data = torch.ones(1) * bias_value

        # for iter update
        self.pose_head = _get_clones(self.pose_head, num_dec_layers)
        self.shape_head = _get_clones(self.shape_head, num_dec_layers)
        
        # setting query dim (bboxes as queries)
        self.query_dim = query_dim
        assert query_dim == 4
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.aux_loss = aux_loss
        self.iter_update = iter_update
        assert iter_update
        if self.iter_update:
            self.decoder.decoder.bbox_embed = self.bbox_embed

        assert bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # ========== End of decoder settings =============

        if self.depth_stage1_only:
            for p in self.parameters():
                p.requires_grad = False
            stage1_modules = []
            if self.use_bbox_prior:
                stage1_modules.extend([
                    self.bbox_prior_feature_proj,
                    self.bbox_prior_pos_proj,
                    self.bbox_prior_decoder,
                    self.bbox_prior_bbox_embed,
                    self.bbox_prior_conf_head,
                    self.bbox_prior_refpoint_embed,
                    self.bbox_prior_tgt_embed,
                ])
            if getattr(self, "use_depth_only_tz_decoder", False):
                stage1_modules.extend([
                    self.depth_only_tz_decoder,
                    self.depth_tz_interface,
                ])
            # Keep HMR ViT encoder trainable in Stage-1 so that bbox prior can adapt
            stage1_modules.extend([
                self.encoder,
                self.encoder_patch_proj,
                self.encoder_patch_norm,
                self.feature_proj,
            ])
            for m in stage1_modules:
                if m is None:
                    continue
                for p in m.parameters():
                    p.requires_grad = True
            self.encoder_cr_token.requires_grad = True
            self.encoder_pos_embeds.requires_grad = True

        # for dn training
        self.use_dn = dn_cfg['use_dn']
        if self.depth_stage1_only:
            self.use_dn = False
        self.dn_cfg = dn_cfg
        if self.use_dn:
            assert dn_cfg['dn_number'] > 0
            if dn_cfg['tgt_embed_type'] == 'labels':
                self.dn_enc = nn.Embedding(dn_cfg['dn_labelbook_size'], hidden_dim)
            elif dn_cfg['tgt_embed_type'] == 'params':
                self.dn_enc = nn.Linear(num_poses*3 + dim_shape, hidden_dim)
            else:
                raise NotImplementedError

    def forward_dav2_features(self, samples):
        """Extract depth features from DAV2 backbone (frozen), following BLADE's approach
        Supports arbitrary aspect ratios and batch processing via padding.
        
        Args:
            samples: list of [C, H, W] tensors (may have different H, W)
        Returns:
            depth_feat: list of 4 feature maps from DPT head, each [B, C_i, H_i, W_i]
                       [path_4, path_3, path_2, path_1] where path_1 is the finest scale
        """
        if not self.use_dav2_depth or self.dav2_backbone is None:
            return None
        
        # Find max dimensions in batch for padding
        max_h = max(img.shape[1] for img in samples)
        max_w = max(img.shape[2] for img in samples)
        
        # Pad all images to uniform size for batch processing
        padded_images = []
        for img in samples:
            h, w = img.shape[1], img.shape[2]
            # Create padded tensor
            padded = torch.zeros(3, max_h, max_w, device=img.device, dtype=img.dtype)
            padded[:, :h, :w] = img
            padded_images.append(padded)
        
        images = torch.stack(padded_images, dim=0)  # [B, 3, max_h, max_w]
        
        # Calculate patch dimensions (DAV2 uses patch_size=14)
        patch_h, patch_w = max_h // 14, max_w // 14

        # Extract features using DAV2, following BLADE's implementation
        if self.dav2_frozen:
            # 1) Run DINOv2 backbone in eval mode without gradients (frozen encoder)
            self.dav2_backbone.pretrained.eval()
            with torch.no_grad():
                features = self.dav2_backbone.pretrained.get_intermediate_layers(
                    images,
                    self.dav2_backbone.intermediate_layer_idx[self.dav2_encoder],
                    return_class_token=True
                )

            # 2) Run DPT depth head to get intermediate features
            # Use forward_features_only since output_conv1/2 are frozen anyway
            depth_feat = self.dav2_backbone.depth_head.forward_features_only(features, patch_h, patch_w)
        else:
            # Full DAV2 trainable: allow gradients through both backbone and depth head
            features = self.dav2_backbone.pretrained.get_intermediate_layers(
                images,
                self.dav2_backbone.intermediate_layer_idx[self.dav2_encoder],
                return_class_token=True
            )
            depth_feat = self.dav2_backbone.depth_head.forward_features_only(features, patch_h, patch_w)
        
        # depth_feat is a list of 4 feature maps at different scales: [path_4, path_3, path_2, path_1]
        return depth_feat

    def forward_encoder(self, samples, targets):
        """Encoder forward."""
        B = len(samples)
        C = self.encoder.embed_dim
        cr_token_list = [self.encoder_cr_token]*len(samples)

        
        # Standard ViT processing: img2token
        lvl0_feature_hw = [(img.shape[1]//self.patch_size, img.shape[2]//self.patch_size) for img in samples]
        lvl0_token_lens = [h*w for (h,w) in lvl0_feature_hw]
        lvl0_img_patches = torch.cat([img2patch_flat(img, patch_size = self.patch_size)\
                                    for img in samples], dim=0)
        lvl0_tokens = self.encoder_patch_norm(self.encoder_patch_proj(lvl0_img_patches).flatten(1))
        

        # token position information
        full_grids = torch.meshgrid(torch.arange(self.feature_size), torch.arange(self.feature_size), indexing='ij')
        lvl0_pos_y = torch.cat([full_grids[0][:h,:w].flatten() for (h,w) in lvl0_feature_hw]).to(device = lvl0_tokens.device)
        lvl0_pos_x = torch.cat([full_grids[1][:h,:w].flatten() for (h,w) in lvl0_feature_hw]).to(device = lvl0_tokens.device)

        # pos_embed
        full_pos_embed = self.encoder_pos_embeds
        lvl0_pos_embed = torch.cat([full_pos_embed[:h,:w].flatten(0,1)\
                                    for (h,w) in lvl0_feature_hw], dim=0)
        lvl0_tokens = lvl0_tokens + lvl0_pos_embed

        # convert to list for DINOv2 input
        x_list = [torch.cat([cr, lvl0],dim=0).unsqueeze(0)\
                            for (cr, lvl0) \
                            in zip(cr_token_list, lvl0_tokens.split(lvl0_token_lens))]
        
        # normalize positions
        lvl0_pos_y_norm = (lvl0_pos_y.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size
        lvl0_pos_x_norm = (lvl0_pos_x.to(dtype=lvl0_tokens.dtype) + 0.5) / self.feature_size
        pos_y_list = list(lvl0_pos_y_norm.split(lvl0_token_lens))
        pos_x_list = list(lvl0_pos_x_norm.split(lvl0_token_lens))

        # Standard ViT forward pass
        full_x_list, final_feature_list = self.encoder.forward_specific_layers_list(x_list, start = 0, norm=True)
        

        # Construct spatial HMR encoder feature maps for potential fusion with
        # DAV2 features. These maps are built from the final transformer layer
        # patch tokens and reshaped to [B, hidden_dim, H/14, W/14].
        hmr_fmaps = []

        processed_feature_list = []
        encoder_dim = self.encoder.embed_dim
        for i in range(len(final_feature_list)):
            tokens_i = final_feature_list[i][0]  # [N_patches, C_enc]
            patch_h, patch_w = lvl0_feature_hw[i]

            # Build spatial fmap from raw encoder tokens (B=1 here)
            fmap_enc = tokens_i.transpose(0, 1).reshape(1, encoder_dim, patch_h, patch_w)

            # Convert spatial fmap to decoder hidden_dim for downstream fusion
            if encoder_dim == self.hidden_dim:
                fmap_hidden = fmap_enc
            else:
                # Project per-patch tokens to decoder dim, then reshape back to feature map
                tokens_proj = self.feature_proj(fmap_enc.reshape(encoder_dim, -1).transpose(0, 1))
                fmap_hidden = tokens_proj.transpose(0, 1).reshape(1, self.hidden_dim, patch_h, patch_w)

            hmr_fmaps.append(fmap_hidden)

            tokens_out = fmap_enc.reshape(encoder_dim, patch_h * patch_w).transpose(0, 1)
            processed_feature_list.append(tokens_out.unsqueeze(0))

        # proj
        token_lens = [feature.shape[1] for feature in processed_feature_list]
        final_features = self.feature_proj(torch.cat(processed_feature_list, dim=1).squeeze(0)) # (sum(L), C)
        assert tuple(final_features.shape) == (sum(token_lens), self.hidden_dim)
        
        # positional encoding
        pos_embeds = position_encoding_xy(torch.cat(pos_x_list,dim=0), torch.cat(pos_y_list,dim=0), embedding_dim=self.hidden_dim)

        # Stack HMR spatial feature maps to [B, hidden_dim, H/14, W/14]
        if len(hmr_fmaps) > 0:
            # Each fmap has shape [1, C, h_i, w_i]; pad to max(h_i, w_i) within the batch
            patch_h_list = [f.shape[2] for f in hmr_fmaps]
            patch_w_list = [f.shape[3] for f in hmr_fmaps]
            max_h = max(patch_h_list)
            max_w = max(patch_w_list)

            hmr_spatial = hmr_fmaps[0].new_zeros(len(hmr_fmaps), self.hidden_dim, max_h, max_w)
            for b, fmap in enumerate(hmr_fmaps):
                _, _, h, w = fmap.shape
                hmr_spatial[b, :, :h, :w] = fmap[0]
        else:
            hmr_spatial = None

        return final_features, pos_embeds, token_lens, lvl0_feature_hw, hmr_spatial

    

    def process_smpl(self, poses, shapes, cam_xys, cam_intrinsics, detach_j3ds = False):
        bs, num_queries, _ = poses.shape # should be (bs,n_q,num_poses*3)

        # flatten and compute
        poses = poses.flatten(0,1) # (bs*n_q,24*3)
        shapes = shapes.flatten(0,1) # (bs*n_q,10)
        verts, joints = self.human_model(poses=poses,
                                         betas=shapes)
        num_verts = verts.shape[1]
        num_joints = joints.shape[1]
        verts = verts.reshape(bs,num_queries,num_verts,3)
        joints = joints.reshape(bs,num_queries,num_joints,3)

        # apply cam_trans and projection
        scale = 2*cam_xys[:,:,2:].sigmoid() + 1e-6
        t_xy = cam_xys[:,:,:2]/scale
        t_z = (2*self.focal)/(scale*self.input_size)    # (bs,num_queries,1)
        transl = torch.cat([t_xy,t_z],dim=2)[:,:,None,:]    # (bs,nq,1,3)

        verts_cam = verts + transl # only for visualization and evaluation
        j3ds_cam = joints + transl

        if detach_j3ds:
            j2ds_homo = torch.matmul(joints.detach() + transl, cam_intrinsics.transpose(2,3))
        else:
            j2ds_homo = torch.matmul(j3ds_cam, cam_intrinsics.transpose(2,3))
        j2ds_img = (j2ds_homo[..., :2] / (j2ds_homo[..., 2, None] + 1e-6)).reshape(bs,num_queries,num_joints,2)

        depths = j3ds_cam[:,:,0,2:]   # (bs, n_q, 1)
        depths = torch.cat([depths, depths/self.focal], dim=-1) # (bs, n_q, 2)

        return verts_cam, j3ds_cam, j2ds_img, depths, transl.flatten(2)

    def forward(self, samples: NestedTensor, targets, detach_j3ds = False):
        """Forward pass."""
        
        assert isinstance(samples, (list, torch.Tensor))
        bs = len(targets)

        # get cam_intrinsics base (preset K with principal point updated by valid ratio)
        img_size = torch.stack([t['img_size'].flip(0) for t in targets])
        valid_ratio = img_size/self.input_size
        base_intrinsics = self.cam_intrinsics.repeat(bs, 1, 1, 1)
        base_intrinsics[...,:2,2] = base_intrinsics[...,:2,2] * valid_ratio[:, None, :]

        # encoder forward
        final_features, pos_embeds, token_lens, feature_hw_list, hmr_spatial = self.forward_encoder(samples, targets)

        # ========== Bbox-only prior branch (optional) =============
        bbox_prior_outputs = None
        if self.use_bbox_prior and self.bbox_prior_decoder is not None:
            # Initialize independent queries and reference points for bbox-only prior
            bbox_tgt = (self.bbox_prior_tgt_embed.weight).unsqueeze(0).repeat(bs, 1, 1)
            bbox_refpoints = (self.bbox_prior_refpoint_embed.weight).unsqueeze(0).repeat(bs, 1, 1)
            bbox_tgt_lens = [bbox_tgt.shape[1]] * bs

            # Project main decoder memory into bbox prior hidden space if projection is defined.
            bbox_features = final_features
            if self.bbox_prior_feature_proj is not None:
                bbox_memory = self.bbox_prior_feature_proj(bbox_features)
            else:
                bbox_memory = bbox_features

            # Project positional encodings into bbox prior hidden space to match d_model
            bbox_positions = pos_embeds
            if getattr(self, 'bbox_prior_pos_proj', None) is not None:
                bbox_pos_embeds = self.bbox_prior_pos_proj(bbox_positions)
            else:
                bbox_pos_embeds = bbox_positions

            bbox_hs, bbox_references = self.bbox_prior_decoder(
                memory=bbox_memory,
                memory_lens=token_lens,
                tgt=bbox_tgt.flatten(0, 1),
                tgt_lens=bbox_tgt_lens,
                refpoint_embed=bbox_refpoints.flatten(0, 1),
                pos_embed=bbox_pos_embeds,
                self_attn_mask=None,
            )

            bbox_reference_before_sigmoid = inverse_sigmoid(bbox_references)
            bbox_outputs_coords = []
            for lvl in range(bbox_hs.shape[0]):
                tmp = self.bbox_prior_bbox_embed[lvl](bbox_hs[lvl])
                tmp[..., :self.query_dim] += bbox_reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                bbox_outputs_coords.append(outputs_coord)
            bbox_pred_boxes = torch.stack(bbox_outputs_coords)

            # Confidence predictions per decoder layer
            bbox_outputs_confs = []
            for lvl in range(bbox_hs.shape[0]):
                bbox_outputs_confs.append(self.bbox_prior_conf_head(bbox_hs[lvl]).sigmoid())
            bbox_pred_confs = torch.stack(bbox_outputs_confs)

            bbox_prior_outputs = {
                'pred_boxes': bbox_pred_boxes,
                'pred_confs': bbox_pred_confs,
                'query_tokens': bbox_hs,
            }

        # ========== Depth Tz branch (DAV2 + bbox prior) =============
        depth_tz_outputs = None
        if (
            bbox_prior_outputs is not None
            and self.use_dav2_depth
            and getattr(self, "use_depth_only_tz_decoder", False)
            and self.depth_only_tz_decoder is not None
        ):
            depth_feat = self.forward_dav2_features(samples)
            if depth_feat is not None:
                path_1 = depth_feat[-1]
                if getattr(self, "depth_tz_interface", None) is not None:
                    path_1 = self.depth_tz_interface(path_1)
                depth_tz_outputs = self.depth_only_tz_decoder(
                    path_1,
                    bbox_prior_outputs['pred_boxes'][-1],
                    bbox_prior_outputs['query_tokens'][-1],
                )

        if getattr(self, "depth_stage1_only", False):
            out = {}
            if bbox_prior_outputs is not None:
                out['pred_boxes'] = bbox_prior_outputs['pred_boxes'][-1]
                out['pred_confs'] = bbox_prior_outputs['pred_confs'][-1]
                out['bbox_prior_outputs'] = bbox_prior_outputs
            else:
                dummy_boxes = final_features.new_zeros(bs, self.num_queries, 4)
                dummy_confs = final_features.new_zeros(bs, self.num_queries, 1)
                out['pred_boxes'] = dummy_boxes
                out['pred_confs'] = dummy_confs
            if depth_tz_outputs is not None:
                out['depth_tz_outputs'] = depth_tz_outputs
            return out

        if (
            self.use_bbox_prior
            and bbox_prior_outputs is not None
            and getattr(self, "prior_geo_token_mlp", None) is not None
            and depth_tz_outputs is not None
        ):
            bbox_boxes_last = bbox_prior_outputs['pred_boxes'][-1]
            bbox_confs_last = bbox_prior_outputs['pred_confs'][-1]
            tz_pred = depth_tz_outputs.get('pred_tz', None)
            if tz_pred is not None:
                tz_pred = tz_pred.detach()
                prior_geo = torch.cat([bbox_boxes_last, bbox_confs_last, tz_pred], dim=-1)
                prior_tokens = self.prior_geo_token_mlp(prior_geo)

                memory_list = list(final_features.split(token_lens, dim=0))
                pos_list = list(pos_embeds.split(token_lens, dim=0))

                updated_memory_list = []
                updated_pos_list = []
                for b in range(bs):
                    mem_b = memory_list[b]
                    pos_b = pos_list[b]
                    prior_tokens_b = prior_tokens[b]

                    cx = bbox_boxes_last[b, :, 0]
                    cy = bbox_boxes_last[b, :, 1]
                    pos_prior_b = position_encoding_xy(cx, cy, embedding_dim=self.hidden_dim)

                    mem_b = torch.cat([mem_b, prior_tokens_b], dim=0)
                    pos_b = torch.cat([pos_b, pos_prior_b], dim=0)

                    token_lens[b] = mem_b.shape[0]
                    updated_memory_list.append(mem_b)
                    updated_pos_list.append(pos_b)

                final_features = torch.cat(updated_memory_list, dim=0)
                pos_embeds = torch.cat(updated_pos_list, dim=0)

        pred_intrinsics = base_intrinsics.clone()

        # Initialize queries and reference points with learned embeddings
        embedweight = (self.refpoint_embed.weight).unsqueeze(0).repeat(bs, 1, 1)
        tgt = (self.tgt_embed.weight).unsqueeze(0).repeat(bs, 1, 1)

        # Inject bbox-only prior query tokens into HMR decoder tgt via zero-initialized residual
        if (
            self.use_bbox_prior
            and bbox_prior_outputs is not None
            and self.bbox_prior_query_proj is not None
            and self.bbox_prior_query_fusion_proj is not None
        ):
            prior_tokens = bbox_prior_outputs['query_tokens'][-1]  # [B, Nq, bbox_prior_hidden_dim]
            prior_proj = self.bbox_prior_query_proj(prior_tokens)  # [B, Nq, hidden_dim]
            prior_residual = self.bbox_prior_query_fusion_proj(prior_proj)  # [B, Nq, hidden_dim]
            tgt = tgt + prior_residual

        if (
            depth_tz_outputs is not None
            and getattr(self, "depth_tz_query_proj", None) is not None
            and getattr(self, "depth_tz_query_fusion_proj", None) is not None
        ):
            depth_tokens = depth_tz_outputs.get('query_tokens', None)
            if depth_tokens is not None:
                depth_tokens = depth_tokens.detach()
                depth_proj = self.depth_tz_query_proj(depth_tokens)
                depth_residual = self.depth_tz_query_fusion_proj(depth_proj)
                tgt = tgt + depth_residual

        # Inject bbox-only prior boxes into reference points via zero-initialized residual
        if (
            self.use_bbox_prior
            and bbox_prior_outputs is not None
            and self.bbox_prior_fusion_proj is not None
        ):
            prior_boxes = bbox_prior_outputs['pred_boxes'][-1]  # [B, Nq, 4]
            bbox_residual = self.bbox_prior_fusion_proj(prior_boxes)  # [B, Nq, 4]
            embedweight = embedweight + bbox_residual

        if self.training and self.use_dn:
            input_query_tgt, input_query_bbox, attn_mask, dn_meta =\
                            prepare_for_cdn(targets = targets, dn_cfg = self.dn_cfg, 
                                        num_queries = self.num_queries, hidden_dim = self.hidden_dim, dn_enc = self.dn_enc)
            tgt = torch.cat([input_query_tgt, tgt], dim=1)
            embedweight = torch.cat([input_query_bbox, embedweight], dim=1)
        else:
            attn_mask = None

        tgt_lens = [tgt.shape[1]]*bs

        hs, reference = self.decoder(memory=final_features, memory_lens=token_lens,
                                         tgt=tgt.flatten(0,1), tgt_lens=tgt_lens,
                                         refpoint_embed=embedweight.flatten(0,1),
                                         pos_embed=pos_embeds,
                                         self_attn_mask = attn_mask)
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        pred_boxes = torch.stack(outputs_coords)


        outputs_poses = []
        outputs_shapes = []
        outputs_confs = []
        outputs_j3ds = []
        outputs_j2ds = []
        outputs_depths = []

        # shape of hs: (lvl, bs, num_queries, dim)
        outputs_pose_6d = self.mean_pose.view(1, 1, -1)
        outputs_shape = self.mean_shape.view(1, 1, -1)
        for lvl in range(hs.shape[0]):

            outputs_pose_6d = outputs_pose_6d + self.pose_head[lvl](hs[lvl])
            outputs_shape = outputs_shape + self.shape_head[lvl](hs[lvl])

            if self.training or lvl == hs.shape[0] - 1:
                outputs_pose = rot6d_to_axis_angle(outputs_pose_6d)

                outputs_conf = self.conf_head(hs[lvl]).sigmoid()

                # cam
                cam_xys = self.cam_head(hs[lvl])

                outputs_vert, outputs_j3d, outputs_j2d, depth, transl\
                = self.process_smpl(poses = outputs_pose,
                                    shapes = outputs_shape,
                                    cam_xys = cam_xys,
                                    cam_intrinsics = pred_intrinsics,
                                    detach_j3ds = detach_j3ds)
                
                outputs_poses.append(outputs_pose)
                outputs_shapes.append(outputs_shape)
                outputs_confs.append(outputs_conf)
                outputs_j3ds.append(outputs_j3d)
                outputs_j2ds.append(outputs_j2d)
                outputs_depths.append(depth)
        
        pred_poses = torch.stack(outputs_poses)
        pred_betas = torch.stack(outputs_shapes)
        pred_confs = torch.stack(outputs_confs)
        pred_verts = outputs_vert
        pred_transl = transl
        pred_intrinsics = pred_intrinsics
        pred_j3ds = torch.stack(outputs_j3ds)
        pred_j2ds = torch.stack(outputs_j2ds)
        pred_depths = torch.stack(outputs_depths)

        if self.training > 0 and self.use_dn:
            pred_poses, pred_betas,\
            pred_boxes, pred_confs,\
            pred_j3ds, pred_j2ds, pred_depths,\
            pred_verts, pred_transl =\
                dn_post_process(pred_poses, pred_betas,
                                pred_boxes, pred_confs,
                                pred_j3ds, pred_j2ds, pred_depths,
                                pred_verts, pred_transl,
                                dn_meta, self.aux_loss, self._set_aux_loss)

        out = {'pred_poses': pred_poses[-1], 'pred_betas': pred_betas[-1],
                'pred_boxes': pred_boxes[-1], 'pred_confs': pred_confs[-1], 
               'pred_j3ds': pred_j3ds[-1], 'pred_j2ds': pred_j2ds[-1],
               'pred_verts': pred_verts, 'pred_intrinsics': pred_intrinsics, 
               'pred_depths': pred_depths[-1], 'pred_transl': pred_transl}

        # Add bbox-only prior outputs if available (not yet used in losses)
        if bbox_prior_outputs is not None:
            out['bbox_prior_outputs'] = bbox_prior_outputs

        # Add depth-prior Tz outputs if available (depth-only Tz decoder)
        if depth_tz_outputs is not None:
            out['depth_tz_outputs'] = depth_tz_outputs

        if self.aux_loss and self.training:
            out['aux_outputs'] = self._set_aux_loss(pred_poses, pred_betas,
                                                    pred_boxes, pred_confs,
                                                    pred_j3ds, pred_j2ds, pred_depths)

        if self.training > 0 and self.use_dn:
            out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_poses, pred_betas, pred_boxes, 
                        pred_confs, pred_j3ds, 
                        pred_j2ds, pred_depths):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_poses': a, 'pred_betas': b,
                    'pred_boxes': c, 'pred_confs': d, 
                'pred_j3ds': e, 'pred_j2ds': f, 'pred_depths': g}
                    for a, b, c, d, e, f, g in zip(pred_poses[:-1], pred_betas[:-1], 
                    pred_boxes[:-1], pred_confs[:-1], pred_j3ds[:-1], pred_j2ds[:-1], pred_depths[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_dgg_model(args, set_criterion=True):
    """Build the active DGG-HMR model."""
    encoder = build_encoder(args)
    decoder = build_decoder(args)

    model = Model(
        encoder,
        decoder,
        num_queries=args.num_queries,
        input_size=args.input_size,
        dn_cfg=args.dn_cfg,
        train_pos_embed=getattr(args, 'train_pos_embed', True),
        use_dav2_depth=getattr(args, 'use_dav2_depth', False),
        dav2_encoder=getattr(args, 'dav2_encoder', 'vitb'),
        dav2_frozen=getattr(args, 'dav2_frozen', True),
        use_bbox_prior=getattr(args, 'use_bbox_prior', False),
        bbox_prior_decoder_layers=getattr(args, 'bbox_prior_decoder_layers', 3),
        bbox_prior_hidden_dim=getattr(args, 'bbox_prior_hidden_dim', 256),
        bbox_prior_dim_feedforward=getattr(args, 'bbox_prior_dim_feedforward', 1024),
        depth_only_tz_decoder=getattr(args, 'depth_only_tz_decoder', False),
        depth_only_decoder_layers=getattr(args, 'depth_only_decoder_layers', 2),
        depth_only_nheads=getattr(args, 'depth_only_nheads', 4),
        depth_only_dim_feedforward=getattr(args, 'depth_only_dim_feedforward', 1024),
        depth_tz_hidden_dim=getattr(args, 'depth_tz_hidden_dim', None),
        depth_stage1_only=getattr(args, 'depth_stage1_only', False),
    )
    
    if set_criterion:
        matcher = build_matcher(args)
        weight_dict = args.weight_dict
        losses = args.losses

        if args.dn_cfg['use_dn']:
            dn_weight_dict = {}
            dn_weight_dict.update({f'{k}_dn': v for k, v in weight_dict.items()})
            weight_dict.update(dn_weight_dict)

        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f'{k}.{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        # Add auxiliary weights for bbox-only prior DETR decoder layers
        # Keys follow the pattern: bbox_prior_boxes.prior.{i}, bbox_prior_giou.prior.{i}, bbox_prior_conf.prior.{i}
        if getattr(args, 'use_bbox_prior', False) and getattr(args, 'bbox_prior_decoder_layers', 1) > 1:
            prior_aux_weight_dict = {}
            num_prior_layers = args.bbox_prior_decoder_layers
            base_prior_keys = ['bbox_prior_boxes', 'bbox_prior_giou', 'bbox_prior_conf']
            for i in range(num_prior_layers - 1):
                for k in base_prior_keys:
                    if k in weight_dict:
                        prior_aux_weight_dict[f'{k}.prior.{i}'] = weight_dict[k]
            weight_dict.update(prior_aux_weight_dict)

        criterion = SetCriterion(
            matcher, 
            weight_dict, 
            losses=losses, 
            j2ds_norm_scale=args.input_size,
            input_size=args.input_size, 
            FOV=getattr(args, 'FOV', pi/3),
        )
        return model, criterion
    else:
        return model, None
