"""
Optimized DAV2 feature extractor that only computes intermediate features
without the final depth prediction head (to save computation)
"""
import torch
import torch.nn as nn


def forward_features_only(self, out_features, patch_h, patch_w):
    """
    Modified DPT head forward that only returns intermediate features,
    skipping the final depth prediction computation to save overhead.
    
    This is more efficient than the original forward when we only need
    the intermediate features (path_1/2/3/4) and not the depth map.
    
    Args:
        out_features: intermediate features from DINOv2 backbone
        patch_h, patch_w: patch dimensions
        
    Returns:
        [path_4, path_3, path_2, path_1]: list of 4 feature maps
    """
    out = []
    for i, x in enumerate(out_features):
        if self.use_clstoken:
            x, cls_token = x[0], x[1]
            readout = cls_token.unsqueeze(1).expand_as(x)
            x = self.readout_projects[i](torch.cat((x, readout), -1))
        else:
            x = x[0]
        
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        
        x = self.projects[i](x)
        x = self.resize_layers[i](x)
        
        out.append(x)
    
    layer_1, layer_2, layer_3, layer_4 = out
    
    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)
    
    path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
    path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
    
    # Skip the depth prediction computation (output_conv1, interpolate, output_conv2)
    # This saves computation when we only need the features
    
    return [path_4, path_3, path_2, path_1]


def forward_with_intermediate_features(self, out_features, patch_h, patch_w):
    """
    Modified DPT head forward that returns both the final depth prediction
    AND intermediate features (path_1/2/3/4). This ensures all parameters
    receive gradients (avoiding DDP unused-parameter errors) while still
    providing the multi-scale features we need for Depth-DETR.
    
    Args:
        out_features: intermediate features from DINOv2 backbone
        patch_h, patch_w: patch dimensions
        
    Returns:
        depth_pred: final depth prediction [B, H, W]
        features: [path_4, path_3, path_2, path_1] list of 4 feature maps
    """
    out = []
    for i, x in enumerate(out_features):
        if self.use_clstoken:
            x, cls_token = x[0], x[1]
            readout = cls_token.unsqueeze(1).expand_as(x)
            x = self.readout_projects[i](torch.cat((x, readout), -1))
        else:
            x = x[0]
        
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        
        x = self.projects[i](x)
        x = self.resize_layers[i](x)
        
        out.append(x)
    
    layer_1, layer_2, layer_3, layer_4 = out
    
    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)
    
    path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
    path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
    
    # Complete the depth prediction (ensures all parameters receive gradients)
    import torch.nn.functional as F
    out = self.scratch.output_conv1(path_1)
    out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
    depth_pred = self.scratch.output_conv2(out)
    
    return depth_pred, [path_4, path_3, path_2, path_1]


def patch_dav2_for_feature_extraction(dav2_backbone):
    """
    Patch the DAV2 backbone's depth_head to add methods for feature extraction.
    
    Args:
        dav2_backbone: DepthAnythingV2 model instance
    """
    # Add the feature-only forward method (skips output_conv1/2)
    dav2_backbone.depth_head.forward_features_only = forward_features_only.__get__(
        dav2_backbone.depth_head, 
        dav2_backbone.depth_head.__class__
    )
    
    # Add the method that returns both depth and features
    dav2_backbone.depth_head.forward_with_intermediate_features = forward_with_intermediate_features.__get__(
        dav2_backbone.depth_head, 
        dav2_backbone.depth_head.__class__
    )
    
    return dav2_backbone
