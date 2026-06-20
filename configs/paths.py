dataset_root = './data'
weights_root = './weights'

smpl_model_path = f'{weights_root}/smpl_data'
smpl_mean_path = f'{smpl_model_path}/smpl/smpl_mean_params.npz'

dinov2_root = f'{weights_root}/dinov2'
dinov2_pretrained_paths = {
    'vitb': f'{dinov2_root}/dinov2_vitb14_pretrain.pth',
    'vitl': f'{dinov2_root}/dinov2_vitl14_pretrain.pth',
}

dav2_root = f'{weights_root}/dav2'
dav2_pretrained_paths = {
    'vits': f'{dav2_root}/depth_anything_v2_vits.pth',
    'vitb': f'{dav2_root}/depth_anything_v2_vitb.pth',
    'vitl': f'{dav2_root}/depth_anything_v2_vitl.pth',
    'vitg': f'{dav2_root}/depth_anything_v2_vitg.pth',
}
