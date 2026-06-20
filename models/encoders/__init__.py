# DGG-HMR encoder
from models.encoders.dinov2.models.vision_transformer import vit_base, vit_large
import torch
from configs.paths import dinov2_pretrained_paths

def build_encoder(args):
    """Build standard DINOv2 encoder."""
    weights = None
    if args.encoder == 'vitb':
        model = vit_base(img_size = 518,
            patch_size  = 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            num_register_tokens = 0,
            interpolate_antialias = False,
            interpolate_offset = 0.1,
            num_additional_blocks = 0)  # No additional blocks
        if args.mode.lower() == 'train':
            weights = torch.load(dinov2_pretrained_paths[args.encoder], weights_only=False)
    elif args.encoder == 'vitl':
        model = vit_large(img_size = 518,
            patch_size  = 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            num_register_tokens = 0,
            interpolate_antialias = False,
            interpolate_offset = 0.1,
            num_additional_blocks = 0)  # No additional blocks
        if args.mode.lower() == 'train':
            weights = torch.load(dinov2_pretrained_paths[args.encoder], weights_only=False)
    else:
        raise NotImplementedError
    
    if weights is not None:
        print('Loading pretrained DINOv2...')
        model.load_state_dict(weights, strict=True)

    return model
